# K-over-S Implementation — Stages & Mechanism (Milestone 2)

Implements the decode path designed in `kovers_design.md`, **reusing nano-vLLM's chunked-prefill
attention** instead of writing a new attention kernel.

## Key insight

A K-over-S **block step is just a chunked-prefill step over K query positions**:
- `flash_attn_varlen_func(causal=True, block_table=…)` already does "K causal queries attend to
  `[0, L+K)` paged KV" (this is nano-vLLM's prefix-cache prefill path).
- `store_kvcache` already **skips** cache slots set to `-1`.

So K-over-S needs **no attention change**: feed K placeholder positions `[L, L+K)`; set
`slot_mapping` = real slots on the **last** of the S steps (persist + commit) and `-1` otherwise
(the K positions stay "scratch", recomputed each step). Causal intra-block falls out of `causal=True`.

## Code changes (all reuse existing machinery)

- **`block_manager.append_blocks(seq, k)`** — allocate blocks covering `num_tokens + k` (raises if KV
  exhausted → caught as OOM by `bench_decode`).
- **`model_runner.prepare_block_decode(seqs, k, persist)`** — build a prefill-style `Context` for K
  positions/seq: `positions=[L,L+k)`, `cu_seqlens_k = L+k`, `block_tables` set, `slot_mapping` = real
  iff `persist` else `-1`. **`run_block(seqs, k, persist)`** — forward only (no sampling; timing).
- **`llm_engine.kovers_decode(k, s, output_len)`** — prefill **untimed** (reuse the scheduler's chunked
  prefill; do not append a token), then time `output_len/k` blocks × `s` forwards each; on the last of
  the s steps persist KV and commit (append k placeholder tokens → advance `L += k`). Returns
  `(decode_time_s, peak_mem_gb)`. TP-safe: each step is a `model_runner.call("run_block", …)` broadcast.
- **`bench_decode.py`** — `K>1` → `llm.kovers_decode(K,S,T)`; `K=1,S=1` → production `step()` decode.

## Stages

1. **(done)** demand.md harness — schema, two sweeps, 4-figure plotter, OOM/skip reporting.
2. **(this stage)** the engine path above — **implemented, UNVALIDATED until a GPU is available.**
3. **(next, on GPU once flash-attn builds)** smoke a tiny config (e.g. `K=2,S=1,B=1,prefix=128`),
   fix any bookkeeping, then run `sweep_full.sh`.

## Caveats to validate on GPU

- **K=1 baseline** uses the production **cudagraph** decode; **K>1** uses the **eager** block path. This
  is the fair comparison (best-case AR vs multi-token), but note the path discontinuity at K=1.
- For `s < S` the K positions read **garbage** within-block KV (not persisted) — timing-correct, values
  irrelevant (not a quality test).
- `peak_mem_gb` is rank-0 only; TP timing uses a rank-0 `cuda.synchronize()` (TP all-reduce keeps ranks tight).
- Prefill appends no token here; block-decode starts at `L = prefix_len`. Immaterial for timing.
- Memory: the worst corner (B=64 × prefix 10K) may exceed the KV budget. `kovers_decode` does an
  **up-front capacity check** (needed blocks vs `num_kvcache_blocks`) and raises
  `torch.cuda.OutOfMemoryError` before any forward, so it's recorded as a clean `OOM` rather than a
  mid-forward hang / TP-desync / `IndexError` (adversarial-review HIGH + mediums, fixed).
- `peak_mem_gb` is measured over the **decode phase** in both paths (vanilla resets the peak at the
  first decode step) so the K=1 vs K>1 memory column is comparable.
