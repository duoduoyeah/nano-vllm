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
- **K>1 / S-step decode (Milestone 2): DONE + GPU-validated.** Implemented via the **cudagraph decode path with `seqlen_q=K`** (`model_runner.prepare_decode_k`/`run_block_decode` + cudagraph capture for `decode_k=K`, `LLMEngine.kovers_decode`, `block_manager.append_blocks`) so K=1 and K>1 are compared **fairly** (both graph-replayed, not eager). Design: `kovers_design.md`; mechanism + caveats: `kovers_impl.md`.
- **Ragged variable-K decode (Milestone 3): DONE + GPU-validated.** Variable K∈[1,8] *per request*, S=1, via a **single varlen cudagraph** (`model_runner.prepare_decode_varlen`/`run_block_decode_varlen` + `capture_cudagraph_varlen`, `LLMEngine.kovers_decode_vark`, balanced schedule `vark_schedule.py`, `bench_vark.py`). Result: **raggedness is free** (≈ uniform same-average-K). Also filled the K-sweep with **K=2, 4** (`run_ks_array.slurm`). NB: a captured varlen graph holds NCCL comms — `ModelRunner.exit` now frees it before `destroy_process_group` (else TP teardown hangs).

## Results (response to `demand.md`)

Qwen3-32B · TP=4 on 4× RTX PRO 6000 Blackwell (**PCIe, no NVLink**) · T=256 decode · distinct random per-request prefixes · decode-only (prefill untimed) · **fair: both K=1 and K>1 are cudagraph-replayed**.

**Deliverables:** `results/vanilla.csv` (K=1 baseline, 8 rows) · `results/sweep.csv` (47 rows: 42 ok + 5 OOM) · `results/vark.csv` (ragged variable-K vs uniform-K=4, 4 rows) · `results/fig1–4.png`.
Grid (uniform K-over-S): K ∈ {1, 2, 4, 16, 32, 64, 256}, S ∈ {1, 2, 16} (for K=16), B ∈ {1, 64, 128} @ prefix 1K / B ∈ {1, 64} @ prefix 10K. Plus a **ragged variable-K** study (below): K∈[1,8] per request, S=1, B=64, both prefixes.

**Pull the outputs to local (scp).** `results/*.csv` and `*.png` are gitignored (not committed), so copy
them from HPCC. Use the **`hpcc-sli-login`** SSH alias — the **login** node (always use the `*-login`
alias for transfers, not the compute-node one; both are defined in `~/.ssh/config`). Quote remote globs
so they expand on HPCC, not locally:

```bash
R=/bigdata/blilab/sli588/cs213/nano-vllm/experiments/results
scp "hpcc-sli-login:$R/vanilla.csv" "hpcc-sli-login:$R/sweep.csv" "hpcc-sli-login:$R/vark.csv" "hpcc-sli-login:$R/fig*.png" ./results/
```

### Throughput (tok/s), uniform K, S=1 (eff = K)
| B | K=1 | K=2 | K=4 | K=16 | K=32 | K=64 | K=256 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 · P=1K | 65 | 114 | 225 | 730 | 1,163 | 1,531 | **3,497** |
| 64 · P=1K | 1,427 | 2,217 | 3,163 | 3,799 | 3,993 | 4,096 | OOM |
| 128 · P=1K | 2,318 | 2,762 | 3,310 | 3,893 | 4,043 | 4,111 | OOM |
| 1 · P=10K | 61 | 110 | 214 | 700 | 1,109 | 1,473 | **3,244** |
| 64 · P=10K | 909 | 878 | 1,518 | 2,863 | 3,405 | 3,754 | OOM |

### Per-request latency = seconds to generate the 256 tokens (S=1, P=1K)
| B | K=1 | K=16 | K=32 | K=64 | K=256 |
|--:|--:|--:|--:|--:|--:|
| 1 | 3.91 | 0.35 | 0.22 | 0.17 | **0.073** |
| 64 | 11.5 | 4.31 | 4.10 | 4.00 | OOM |

### Ragged variable-K decode (variable K per request, S=1, B=64) — `vark.csv`
Each request commits a **different** number of tokens per step, K∈[1,8], on a **balanced** schedule
(`vark_schedule.py`) that keeps the per-step total constant (avg 4 → 256 query rows/step × 64 steps,
each request emits exactly 256) so a **single varlen cudagraph** (`flash_attn_varlen_func` + `cu_seqlens`
+ paged `block_table`) covers every step. Compared against **uniform K=4** (same average work, every
request commits exactly 4/step). Both cudagraph-replayed (`cudagraph=yes`).

| prefix | ragged avg-4 (tok/s) | uniform K=4 (tok/s) | ragged / uniform |
|--:|--:|--:|--:|
| 1K | 3,154.8 | 3,161.0 | 99.8% |
| 10K | 1,546.1 | 1,514.2 | 102.1% |

**Raggedness is free.** Variable per-request K lands within ±2% of uniform-K=4 at equal average work —
the varlen `cu_seqlens` indirection costs nothing measurable over the rectangular kvcache path, so decode
throughput is set by the **average** tokens/step, not by whether the per-request K is uniform or ragged.
(Cross-check: uniform-K=4 B=64 appears in both `vark.csv` and `sweep.csv` and agrees to 0.3%.)

### Findings — when does multi-token (K-over-S) decode beat 1-token AR?
1. **K>1 wins when eff = K/S > 1 (maximized at S=1).** The win is **largest at small batch** (idle, memory-bound GPU): at B=1, K=256 = one forward → 256 tokens in **0.073 s ≈ 54×** the AR baseline.
2. **High batch is compute-bound** (~**4,100 tok/s** ceiling here): K=16 already nearly saturates it, bigger K barely helps.
3. **At fixed K, throughput ∝ 1/S exactly** (S=2 = ½, S=16 = 1/16): the S steps are identical repeats in this cost-only emulation, so S is pure overhead — **eff = 1 (K=16/S=16) loses to AR**.
4. **Memory:** K=256 fits only at B=1; B=128 fits only at the 1K prefix (B=128×10K and K=256 at B≥64 OOM — recorded, never dropped).
5. **Filling K=2/4 shows the low-K shape:** at B=64/P=1K throughput climbs smoothly with diminishing returns (1,427 → 2,217 → 3,163 → 3,799 → … ceiling); at P=10K the payoff only starts at K≥4 (K=1≈K=2≈900, long-KV-read-bound). And **ragged avg-4 ≈ uniform-K=4** — only the *average* tokens/step matters, not the per-request uniformity.
