# K-over-S Multi-Token Decode — Design (Milestone 2)

Emulate "**commit K tokens every S steps**" on vanilla Qwen3-32B to find when `K>1` beats the
`(1,1)` AR baseline. **Decode-stage system cost only** (throughput/latency), *not* quality — so the
token ids are arbitrary. Same model / TP=4 / random per-request prefix (1K, 10K) / **256 committed**
output tokens as the vanilla sweep (see `experiment_plan.md`).

## Mechanism (block-structured decode)

Let `L` = committed length (starts right after prefill). Until 256 tokens are committed:

1. A **block** = `K` query positions at `[L, L+K)`.
2. Run **S forward passes** over those same `K` positions (so `B×K` query rows per forward, vs `B×1` for AR):
   - Queries attend to the committed KV cache `[0, L)` and, **within the block, causally** (query *i*
     sees block positions `0..i`). *[decision 1]*
   - For `s < S`: the block's K/V are computed for the attention but **NOT persisted** to the paged KV
     cache — they are scratch, recomputed each step. *("these K tokens do not use their KV cache")*
   - For `s == S` (last step): **persist** the block's K/V into the cache and **advance `L += K`**.
3. Next block; repeat. Fed token ids = **fixed placeholder** for all S steps. *[decision 2]*

Effective committed rate = **K/S tokens per step**. `(K=1, S=1)` is exactly the AR baseline.
Constraint `K/S ≥ 1`; `T=256` is divisible by every K so blocks divide evenly (`256/K` blocks × S steps).

## Decisions (confirmed)

1. **Intra-block mask = causal** (AR-within-block), not bidirectional. Maps directly onto flash-attn's
   `causal=True` varlen path / the merged chunked-prefill primitive; a bidirectional-suffix mask is not
   a simple flash-attn flag.
2. **Fixed placeholder ids** fed to the K positions for all S steps (no feedback of per-step
   predictions). Quality is irrelevant, and this keeps each of the S steps identical compute.

## What changes (engine modification — reuses chunked-prefill)

- **Scheduler** (`nanovllm/engine/scheduler.py`): schedule `K` query positions/seq/step; track each
  block's step counter `s ∈ [1, S]`; **commit** (persist KV + `L += K`) only at `s == S`; finish when
  256 committed.
- **Attention decode branch** (`nanovllm/layers/attention.py`): process the K block like a *causal
  prefill chunk* against cache `[0, L)`; **suppress the KV-cache store for `s < S`, store at `s == S`**.
- **`bench_decode.py`**: add `--k`, `--s`; loop `256/K` blocks × `S` steps; still time only the decode
  window; throughput = committed `256·B / time`; per-token latency. Add columns `k, s, eff (=K/S)`.
- **Sweep**: extend to the **12 valid (K,S) pairs × B × prefix** (96 runs). `(1,1)` is shared with the
  vanilla baseline (`vanilla.csv`).

## Why this is the measurement

Per step does `B×K` query rows × S forwards to land K tokens (**more FLOPs**), but the **KV cache grows
the same as AR** (only `+K` per committed block; scratch positions never persist). Small-batch decode is
memory-bandwidth-bound and underuses the GPU, so packing `B×K` rows per forward can be near-free —
until compute saturates. **Where `K>1` beats `(1,1)` across `(B, K, S, prefix)` is the result.**

## Open / later

- Exact "scratch" KV handling for `s < S` (recompute-in-attention vs temporary cache slots) — pick the
  simplest the flash-attn path allows.
- The S forwards are identical compute (fixed ids), but we run them for real to capture per-step overhead.
- Status: **not yet implemented** — depends on a working `.venv` (flash-attn) + a validated vanilla baseline.
