import atexit
import os
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        config.shm_name = f"nanovllm-{os.getpid()}"   # unique per engine -> no /dev/shm collision/landmine across jobs
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def kovers_decode(self, k: int, s: int, output_len: int):
        """Multi-token (K-over-S) decode benchmark driver (see experiments/kovers_design.md).

        Assumes requests are already added. Runs prefill UNTIMED (reusing the scheduler's chunked
        prefill; no token is appended), then TIMES committing K tokens per block over S forward
        passes — KV persisted only on the last of the S steps. Static batch (all seqs lockstep).
        Token values are placeholders (not a quality test). Returns (decode_time_s, peak_mem_gb_rank0).
        """
        assert output_len % k == 0, "output_len must be divisible by K"
        bm = self.scheduler.block_manager

        # Up-front KV-capacity check: the OOM corner (e.g. B=64 x prefix 10K) must be recorded as a
        # CLEAN OOM, not a mid-forward hang / TP all_reduce desync / IndexError mislabel (review HIGH +
        # mediums). num_kvcache_blocks already accounts for weights; if the whole run can't fit, bail now.
        block_size = bm.block_size
        need = sum((seq.num_tokens + output_len + block_size - 1) // block_size for seq in self.scheduler.waiting)
        capacity = len(bm.free_block_ids) + len(bm.used_block_ids)
        if need > capacity:
            raise torch.cuda.OutOfMemoryError(f"KV cache too small: need {need} blocks, have {capacity}")

        # 1. prefill (untimed) — reuse the scheduler's chunked prefill; do NOT append a token.
        while self.scheduler.waiting:
            seqs, is_prefill = self.scheduler.schedule()
            if not is_prefill:   # KV pressure made prefill fall through to the decode branch -> clean OOM
                raise torch.cuda.OutOfMemoryError("prefill could not be scheduled (KV pressure)")
            self.model_runner.call("run", seqs, True)
            for seq in seqs:
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
        seqs = list(self.scheduler.running)
        assert seqs
        for seq in seqs:
            # decode-mode pickling: send only last_token to TP workers (not the whole id list),
            # and block_decode reads num_tokens/block_table/last_token — never token_ids.
            seq.is_prefill = False

        # 2. timed K-over-S block decode
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = perf_counter()
        for _ in range(output_len // k):
            for seq in seqs:
                bm.append_blocks(seq, k)                       # blocks for [L, L+k) before the writes
            for _ in range(s):
                self.model_runner.call("run_block_decode", seqs, k)   # cudagraph-replayed (decode_k=k)
                # TP lockstep: nano-vLLM broadcasts commands through a SINGLE shm buffer; this loop
                # isn't throttled by sampling, so without a sync rank-0 overwrites the buffer before
                # the workers read each call -> NCCL collective desync/hang. Sync each step (graph
                # replay keeps it cheap; this is what makes K>1 fast yet TP-correct).
                if self.ps:
                    torch.cuda.synchronize()
            for seq in seqs:                                   # commit: advance L by k placeholder tokens
                for _ in range(k):
                    seq.append_token(seq.last_token)
        torch.cuda.synchronize()
        decode_time = perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        return decode_time, peak_gb

    def reset(self):
        """Free all KV blocks and clear the scheduler queues so the engine can run another fresh
        config in-process (used by the per-(K,S) sweep to amortize the model load). Idempotent: a
        finished vanilla run already self-cleans; kovers/errored runs leave seqs to deallocate here."""
        bm = self.scheduler.block_manager
        for seq in list(self.scheduler.running) + list(self.scheduler.waiting):
            if seq.block_table:
                bm.deallocate(seq)
        self.scheduler.running.clear()
        self.scheduler.waiting.clear()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
