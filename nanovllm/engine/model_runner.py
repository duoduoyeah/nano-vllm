import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            if config.decode_varlen:
                self.capture_cudagraph_varlen()   # ragged variable-K decode (one graph, fixed total_q)
            else:
                self.capture_cudagraph()           # uniform K-over-S decode (graph per batch size)
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name=config.shm_name, create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=config.shm_name)
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if hasattr(self, "varlen_graph"):   # release the captured graph (holds NCCL comms) BEFORE
                del self.varlen_graph, self.varlen_graph_vars   # destroy_process_group, else teardown hangs
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_decode_k(self, seqs: list[Sequence], k: int):
        # Multi-token (K-over-S) decode: k query positions/seq at [L, L+k) attending CAUSALLY to
        # [0, L+k) via the DECODE path (flash_attn_with_kvcache, seqlen_q=k) — which is cudagraph-able,
        # unlike the eager prefill/varlen path. The k new positions' KV is stored (slot_mapping) and
        # cache_seqlens=L+k. Placeholder ids (not a quality test); the driver advances seq.num_tokens
        # by k only after the S steps, so during the S steps the same [L, L+k) slots are re-written.
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            L = seq.num_tokens                       # committed length; KV cached for [0, L)
            input_ids.extend([seq.last_token] * k)   # placeholder query ids
            positions.extend(range(L, L + k))
            context_lens.append(L + k)               # queries attend to [0, L+k)
            for pos in range(L, L + k):
                slot_mapping.append(seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    @torch.inference_mode()
    def run_block_decode(self, seqs: list[Sequence], k: int):
        # One K-over-S forward over k query positions/seq, via run_model -> cudagraph replay when the
        # graphs were captured for decode_k=k. Logits are computed and discarded (timing only; no sampling).
        input_ids, positions = self.prepare_decode_k(seqs, k)
        self.run_model(input_ids, positions, False)
        reset_context()
        return None

    def prepare_decode_varlen(self, seqs: list[Sequence], ks: list[int]):
        # Ragged multi-token decode (variable K per request, S=1): seq i contributes ks[i] query
        # positions [L_i, L_i+ks[i]) attending CAUSALLY to [0, L_i+ks[i]) via the paged VARLEN path
        # (the is_prefill attention branch -> flash_attn_varlen_func + block_table; the only flash-attn
        # API that takes ragged per-seq query lengths). total_q = sum(ks) is kept CONSTANT across steps
        # by the balanced schedule so the single varlen cudagraph applies. Placeholder ids (cost only).
        input_ids = []
        positions = []
        slot_mapping = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        for seq, k in zip(seqs, ks):
            L = seq.num_tokens                       # committed length; KV cached for [0, L)
            input_ids.extend([seq.last_token] * k)   # placeholder query ids
            positions.extend(range(L, L + k))
            cu_seqlens_q.append(cu_seqlens_q[-1] + k)
            cu_seqlens_k.append(cu_seqlens_k[-1] + L + k)
            max_seqlen_q = max(max_seqlen_q, k)
            max_seqlen_k = max(max_seqlen_k, L + k)
            for pos in range(L, L + k):
                slot_mapping.append(seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        # is_prefill=True routes attention to the varlen+paged branch; max_seqlen_q/k are read at
        # capture and baked into the graph grid (replay stays within k_max / max_model_len).
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    @torch.inference_mode()
    def run_block_decode_varlen(self, seqs: list[Sequence], ks: list[int]):
        # One ragged S=1 forward over sum(ks) query rows. Replays the varlen cudagraph when total_q/bs
        # match the captured shapes (balanced-schedule case); else eager. compute_logits runs AFTER
        # reset_context() (is_prefill=False) so it scores ALL sum(ks) rows -> same lm_head work as the
        # uniform K path's bs*K rows (fair). Logits discarded (timing only; no sampling).
        input_ids, positions = self.prepare_decode_varlen(seqs, ks)
        total_q = input_ids.size(0)
        graph = getattr(self, "varlen_graph", None)
        use_graph = (not self.enforce_eager and graph is not None
                     and total_q == self.varlen_graph_vars["total_q"]
                     and len(seqs) == self.varlen_graph_vars["bs"])
        if use_graph:
            context = get_context()
            gv = self.varlen_graph_vars
            gv["input_ids"].copy_(input_ids)
            gv["positions"].copy_(positions)
            gv["slot_mapping"].copy_(context.slot_mapping)
            gv["cu_seqlens_q"].copy_(context.cu_seqlens_q)
            gv["cu_seqlens_k"].copy_(context.cu_seqlens_k)
            gv["block_tables"].zero_()
            gv["block_tables"][:, :context.block_tables.size(1)].copy_(context.block_tables)
            graph.replay()
            hidden = gv["outputs"]
        else:
            hidden = self.model(input_ids, positions)
        reset_context()
        self.model.compute_logits(hidden)
        return None

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        k = self.config.decode_k                 # query positions per seq (1 = vanilla AR)
        bs = input_ids.size(0) // k              # actual batch (graphs are keyed by bs, sized bs*k)
        if is_prefill or self.enforce_eager or bs > self.graph_bs[-1]:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs * k] = input_ids
            graph_vars["positions"][:bs * k] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs * k] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs * k])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        k = config.decode_k                       # query positions/seq; graphs capture seqlen_q=k (1 = vanilla)
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs * k, dtype=torch.int64)
        positions = torch.zeros(max_bs * k, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs * k, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs * k, hf_config.hidden_size)
        # Cap captured graphs by total rows (bs*k): large-K graphs (e.g. K=256 -> bs*256) would blow
        # up capture memory. Batches beyond the cap fall back to eager in run_model — fine for large K
        # (few forwards, so per-launch overhead is negligible vs the big forward itself).
        max_graph_rows = 8192
        self.graph_bs = [bs for bs in ([1, 2, 4, 8] + list(range(16, max_bs + 1, 16))) if bs * k <= max_graph_rows] or [1]
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs * k], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs * k] = self.model(input_ids[:bs * k], positions[:bs * k])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs * k] = self.model(input_ids[:bs * k], positions[:bs * k])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_cudagraph_varlen(self):
        # ONE cudagraph for the RAGGED variable-K decode (S=1) via the paged VARLEN kernel
        # (flash_attn_varlen_func + block_table = the is_prefill attention branch). Capturable because
        # the balanced schedule fixes total query rows: q is a constant (total_q, H, D) buffer, the
        # cu_seqlens_q/k are fixed-shape int buffers whose CONTENTS we overwrite per step, and we capture
        # with max_seqlen_q = k_max (and max_seqlen_k = max_model_len) so the launch grid covers the
        # widest step. The kernel's per-seq key loop reads cu_seqlens from GPU memory at replay, so
        # growing context + varying K are handled inside the captured kernel -- the same property that
        # lets the kvcache decode graph grow. If varlen can't be captured (host-side sync), the engine
        # falls back to eager varlen in run_block_decode_varlen.
        config = self.config
        hf_config = config.hf_config
        bs = min(config.max_num_seqs, 512)                  # fixed batch B (graph is keyed to it)
        total_q = config.varlen_total_q or bs               # fixed query rows/step (balanced column sum)
        k_max = config.varlen_k_max                          # widest per-seq K (grid bound)
        max_ctx = config.max_model_len                       # widest L+K any step reaches (grid bound)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        input_ids = torch.zeros(total_q, dtype=torch.int64)
        positions = torch.zeros(total_q, dtype=torch.int64)
        slot_mapping = torch.zeros(total_q, dtype=torch.int32)
        cu_seqlens_q = torch.zeros(bs + 1, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32)
        block_tables = torch.zeros(bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(total_q, hf_config.hidden_size)

        # capture-time contents: a valid balanced split of total_q across the bs seqs (keys = q here;
        # replay overwrites all of it -- only the SHAPES and the max_seqlen ints get baked into the graph).
        cu = 0
        for i in range(bs):
            q_i = total_q // bs + (1 if i < total_q - (total_q // bs) * bs else 0)
            positions[cu:cu + q_i] = torch.arange(q_i)
            cu_seqlens_q[i + 1] = cu_seqlens_q[i] + q_i
            cu_seqlens_k[i + 1] = cu_seqlens_k[i] + q_i
            cu += q_i

        self.graphs = {}                                     # kept so exit()'s `del self.graphs` holds
        self.graph_pool = None
        graph = torch.cuda.CUDAGraph()
        set_context(True, cu_seqlens_q, cu_seqlens_k, k_max, max_ctx, slot_mapping, None, block_tables)
        outputs[:] = self.model(input_ids, positions)        # warmup
        with torch.cuda.graph(graph):
            outputs[:] = self.model(input_ids, positions)    # capture
        self.graph_pool = graph.pool()
        self.varlen_graph = graph
        torch.cuda.synchronize()
        reset_context()
        self.varlen_graph_vars = dict(
            input_ids=input_ids, positions=positions, slot_mapping=slot_mapping,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables, outputs=outputs, total_q=total_q, bs=bs,
        )
