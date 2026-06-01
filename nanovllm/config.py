import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    decode_k: int = 1   # query positions per seq per decode forward (K-over-S multi-token; 1 = vanilla AR)
    shm_name: str = "nanovllm"   # TP command-buffer name; LLMEngine makes it unique per process
    # Ragged multi-token decode: variable K per request, S=1, via the paged VARLEN attention path
    # (flash_attn_varlen_func + block_table). When set, ModelRunner captures ONE cudagraph for a FIXED
    # total query-row count (varlen_total_q) and batch (max_num_seqs), per-seq K <= varlen_k_max. The
    # balanced schedule keeps per-step total rows constant so the single graph covers every step.
    decode_varlen: bool = False
    varlen_total_q: int = 0     # fixed sum of per-seq K across the batch each step (0 -> = max_num_seqs)
    varlen_k_max: int = 8       # max per-seq K (bounds the captured launch grid)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
