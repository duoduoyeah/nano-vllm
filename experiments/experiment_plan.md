# Experiment Plan — 1 vs K (Multi-Token Decoding)

**Project:** When does multi-token (K-over-S) decoding beat 1-token AR batching?
**Team:** Shiyuan Li, Zhaoling Chen · CS213, Spring 2026
**Runtime:** modified nano-vLLM, branch `project`
**Scope:** decode-stage performance only — per-request latency and system throughput. *Not* text quality.

## Idea

We impose the **(B, K, S)** decode pattern on a model and measure system cost:

- **AR baseline** = `(K=1, S=1)`: 1 token/step/request.
- **Multi-token** = `K>1`: carry K positions, resolve over S steps → `K/S` tokens/request/step.

Since we measure throughput/latency (not quality), we can **emulate** the pattern on a vanilla AR model:

- **Practical model:** **Qwen3-32B** (dense AR, runs on nano-vLLM as-is) driven in (B,K,S) mode.
- **Reference:** **LLaDA2.0-flash (100B)** native masked diffusion — if runnable.

Question: under which `(B, K, S, prefix, output)` regime does `K>1` beat `(1,1)`?

## Fixed setup

- **Model:** Qwen3-32B (practical) · LLaDA2.0-flash 100B (reference). Both GQA — see `memory_math.md`.
- **Hardware:** 4× RTX PRO 6000 Blackwell, 96 GB each = 384 GB.
- **Deployment:** **Scenario 2 — one model, tensor-parallel `TP=4`** across all 4 cards. (Qwen3-32B's `batch64×10K` KV ≈ 167 GB needs pooled memory; one fixed TP=4 config also keeps every sweep point comparable.)
- **Batching:** **static** — B equal-length requests started together (decode microbenchmark).
- **Prefill:** fixed content, **excluded from timing**; distinct random token-ids per request → **no prefix-cache reuse**. We only control prefix length.

## Swept parameters

| Parameter | Symbol | Values |
|---|---|---|
| Batch size | B | 1, 4, 16, 64 |
| Multi-token positions | K | 1, 2, 4, 8, 16 |
| Steps per block | S | 1, 2, 4  (constraint: **K/S ≥ 1**) |
| Prefix length | P | 1K, 10K |
| Output length | T | **256** (fixed; = 16×16, divisible by every K) |

### Valid (K, S) combinations

Constraint **K/S ≥ 1** (≥ 1 committed token per step). Cell = effective tokens/step `K/S`; "–" = not tested.

| K \ S | S=1 | S=2 | S=4 |
|---|---|---|---|
| **K=1**  | 1  | –  | –  |
| **K=2**  | 2  | 1  | –  |
| **K=4**  | 4  | 2  | 1  |
| **K=8**  | 8  | 4  | 2  |
| **K=16** | 16 | 8  | 4  |

→ **12 valid (K, S) pairs**; `(1,1)` is the AR baseline.

## Metrics

- **Per-request latency** — time per output token.
- **System throughput** — total output tokens / second.

## Run count

- Full sweep: `4 (B) × 12 (K,S) × 2 (prefix) = 96` runs at T=256.
- **Milestone 1 (now): vanilla `K=S=1`** → `4 (B) × 2 (prefix) = 8` runs. No engine changes.

## VRAM

Fits 384 GB — see `memory_math.md`. Qwen3-32B worst corner `batch64×10K` ≈ 167 GB KV → the reason for TP=4.

## Status

- **Decided:** model (Qwen3-32B + flash reference) · TP=4 / Scenario 2 · static batching · T=256 · GQA confirmed.
- **Milestone-1 harness:** `bench_decode.py` + `sweep.sh` + `run.slurm` (run it via `HPCC.md`).
- **Open:** run flash-100B too? · exact prefix token distribution.
- **Later:** implement K>1 / S-step decode (engine modification) — localizes to the scheduler + the decode branch of `layers/attention.py`.
