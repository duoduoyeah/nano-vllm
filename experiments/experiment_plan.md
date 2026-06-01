# Experiment Plan — 1 vs K (Multi-Token Decoding)

**Project:** When does multi-token (K-over-S) decoding beat 1-token AR batching?
**Team:** Shiyuan Li, Zhaoling Chen · CS213, Spring 2026
**Runtime:** modified nano-vLLM, branch `project`
**Scope:** decode-stage performance only — per-request latency and system throughput. *Not* text quality.

## Idea

We impose the **(B, K, S)** decode pattern on a **vanilla autoregressive model** and measure system cost:

- **AR baseline** = `(K=1, S=1)`: 1 token/step/request.
- **Multi-token** = `K>1`: carry K positions, resolve over S steps → `K/S` committed tokens/request/step.

Because we measure throughput/latency (**not** quality), we don't need a real diffusion / multi-token
model. We **emulate** the decode pattern on a normal AR model that nano-vLLM already supports — the
generated text can be meaningless; only the compute/memory pattern matters. The recently-merged
**chunked-prefill** path gives us the multi-position-per-forward primitive, so no diffusion code is needed.

- **Model:** **Qwen3-32B** (dense AR, GQA, runs on nano-vLLM as-is) driven in (B, K, S) mode.

Question: under which `(B, K, S, prefix, output)` regime does `K>1` beat `(1, 1)`?

## Fixed setup

- **Model:** Qwen3-32B — 64 layers, 64 Q / 8 KV heads (GQA), head_dim 128, BF16, native ctx 40960. See `memory_math.md`.
- **Hardware:** 4× RTX PRO 6000 Blackwell, 96 GB each = 384 GB (UCR HPCC, nodes gpu13/gpu14).
- **Deployment:** **one model, tensor-parallel `TP=4`** across all 4 cards. (Qwen3-32B's `batch64×10K` KV ≈ 172 GB needs pooled memory; one fixed TP=4 config also keeps every sweep point comparable.)
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

Fits 384 GB — see `memory_math.md`. Qwen3-32B worst corner `batch64×10K` ≈ 172 GB KV → the reason for TP=4.

## How to run (UCR HPCC)

All weights + library caches route into repo-local **`.cache/`** (gitignored, disposable); secrets live in **`.env`** (`HF_TOKEN`). One central `experiments/env.sh` wires this up.

1. **Provision once:** `bash experiments/setup_env.sh` — uv venv (Python 3.12), torch cu128 (Blackwell sm_120), then compile flash-attn 2.8.3 from source (Rocky 8 / glibc 2.28 has no usable prebuilt wheel).
2. **Download weights:** `bash experiments/download_model.sh` → `.cache/models/Qwen3-32B`.
3. **Smoke test:** `cd experiments && sbatch run.slurm` (or `SMOKE=1 sbatch run.slurm`).
4. **Sweep:** `cd experiments && sbatch run.slurm` → `experiments/results/vanilla.csv`.
5. **Summarize:** `python experiments/plot_results.py experiments/results/vanilla.csv` → tables + plots.

Queue: `short_gpu` / `--qos=short_gpu` / `--gres=gpu:blackwell6000:4` / `--time=2:00:00`. See `HPCC.md`.

## Status

- **Decided:** model **Qwen3-32B** · TP=4 · static batching · T=256 · GQA confirmed (8 KV heads, 256 KB KV/token).
- **Milestone-1 harness:** `bench_decode.py` + `sweep.sh` + `run.slurm` (env via `env.sh`; runbook in `HPCC.md`).
- **Open:** exact prefix token distribution · whether the 2h `short_gpu` window fits all 8 baseline runs (each reloads the 65.5 GB model) — if not, request the account-gated `raise` QOS (30d on gpu13/14) or sweep inside one process.
- **K>1 / S-step decode (Milestone 2):** **implemented** (reuses the chunked-prefill attention path; no new kernel) — `block_manager.append_blocks`, `model_runner.prepare_block_decode`/`run_block`, `LLMEngine.kovers_decode`, driven by `bench_decode.py`. Design + decisions in `kovers_design.md`; mechanism + stages + caveats in `kovers_impl.md`. **UNVALIDATED until the first GPU run** (needs flash-attn).
