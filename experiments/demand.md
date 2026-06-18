# Experiment Demands — Decode Benchmark

What we need delivered from HPCC. Run the configs, record the measurements, hand back
the data and figures below. The per-cluster runbook stays in `HPCC.md`; the study
question, sweep grid and status stay in `experiment_plan.md`. Everything else (file
index, VRAM math, K-over-S design + implementation notes) has been consolidated into
the **Reference / Background** part at the bottom of this file.

## Target
- **Model:** Qwen3-32B, **TP=4** (one model across 4 GPUs). Optional reference: LLaDA2.0-flash 100B, if time allows.
- **Decode-only:** prefill excluded from timing; **static batch**; **distinct random token-id prefixes** per request (no prefix-cache reuse); **T = 256** output tokens.

## Configs to run
- **Batch B:** 1, 4, 16, 64
- **Prefix P:** 1024, 10240
- **(K, S):** the 12 valid pairs with `K/S ≥ 1` — K∈{1,2,4,8,16}, S∈{1,2,4}.

Priority order:
1. **K=S=1** — 8 runs (`B × P`). Runnable now with `bench_decode.py` / `sweep.sh`.
2. **K>1** — remaining 11 (K,S) pairs `× B × P`. Requires the multi-token decode path
   (K query positions per forward × S forwards per block); see the K-over-S sections below.

## Per-run measurements → one CSV row
`model, tp, K, S, batch, prefix_len, output_len, decode_time_s, throughput_tok_s, latency_ms_per_tok, peak_mem_gb, eff_tok_per_step(=K/S)`

- `throughput_tok_s` = system: `B·T / decode_time`
- `latency_ms_per_tok` = per request: `decode_time / T`
- Deliver: `results/vanilla.csv` (priority 1) and `results/sweep.csv` (full).

## Figures to deliver (PNG + the data behind each)
1. Throughput (tok/s) vs batch size — one line per K, one panel per prefix.
2. Latency (ms/token) vs batch size — one line per K, one panel per prefix.
3. Throughput vs K at fixed batch — show where K>1 overtakes K=1.
4. At equal `K/S`, K-vs-S comparison — does wider K behave differently from more steps?

## Rules
- **Report every config that OOMs or is skipped — never silently drop one.**
- Note the interconnect context (TP=4 over PCIe, no NVLink) with the numbers.
- Hand back: the CSV(s), the PNGs, and raw run logs, in a location we can pull (e.g. `results/`).

---

# Reference / Background

The sections below were standalone notes (`README.md`, `memory_math.md`,
`kovers_design.md`, `kovers_impl.md`) and have been merged here verbatim-ish so this
file is the single source of context. Cross-references that pointed at those files now
point at the corresponding section here.

## What one data point is

One data point = take a batch of **B** requests, each with its **own random prefix**
(`prefix_len` ids drawn under the model's vocab → no prefix-cache reuse), run prefill
**untimed**, then **time the decode of 256 tokens** for the whole batch. We measure
**system cost (throughput/latency), not output quality** — the generated token ids may be
meaningless; only the emulated compute/memory pattern matters.

---

## Memory math — VRAM for the sweep

How much GPU memory the model needs, and whether it fits on
**4× RTX PRO 6000 Blackwell (96 GB each = 384 GB total)**.

### 1. Equations

```
Total VRAM  ≈  Weights  +  KV cache  +  overhead
```

**Weights**

```
W = N_total × b
```

- `N_total` = total parameter count. Qwen3-32B is **dense**, so every parameter is resident and active.
- `b` = bytes per parameter:

| precision | b (bytes/param) |
|---|---|
| FP32 | 4 |
| FP16 / BF16 | 2 |
| INT8 | 1 |
| INT4 | 0.5 |

**KV cache**

```
KV = 2 · L · H_kv · d_head · L_seq · B · b_kv
```

- `2` = Key + Value
- `L` = layers · `H_kv` = KV heads (GQA: ≪ attention heads) · `d_head` = head dim
- `L_seq` = prefix + generated (≈ prefix here) · `B` = batch · `b_kv` = bytes/elem (FP16 = 2, FP8 = 1)
- Only term that grows with the sweep — scales with `B × L_seq`.

**Overhead**

Activations + framework ≈ **+10–20%**. Inference activations are small. Our B/K/S decode runs
`B×K` query positions per forward (vs `B×1` for AR), inflating *transient activation* but not KV.

### 2. The model — Qwen3-32B (dense AR, B/K/S emulation)

We don't run a real diffusion / multi-token model. We take a **normal autoregressive model that
nano-vLLM already supports** and **impose the (B, K, S) decode pattern** on it:

- AR baseline `(K=1, S=1)`: 1 token/step/request (native).
- Multi-token: present **K query positions per request per forward**, repeat **S forwards** before committing K tokens.

We measure **system cost (throughput/latency), not output quality** — so the emulated compute/memory
pattern is what matters; the generated text can be meaningless. The merged **chunked-prefill** path
provides the multi-position-forward primitive this needs, so we avoid implementing diffusion.

**Config** (`config.json`): `L=64`, `hidden=5120`, 64 attention heads, `H_kv=8` (GQA), `d_head=128`,
`vocab=151936`, native ctx `40960`, BF16 release. **≈ 32.8B params** (dense).

**Weights**

| | INT4 | INT8 | BF16 |
|---|---|---|---|
| **Weights** (`32.8B × b`) | ~16.4 GB | ~32.8 GB | **~65.5 GB** |

BF16 is the released format and the zero-effort path (matches the ~65.5 GB on-disk download).
Quantization is optional — only for extra headroom or if we want to *study* quantization.

**KV cache**

```
KV/token (FP16) = 2 · 64 · 8 · 128 · 2  =  262,144 B  =  256 KB/token
```

| sweep corner (FP16 KV) | tokens (B × L_seq) | KV |
|---|---|---|
| B=1 × 1K   | 1,024      | ~0.27 GB |
| B=64 × 1K  | 65,536     | ~17 GB |
| B=1 × 10K  | 10,240     | ~2.7 GB |
| **B=64 × 10K** (worst) | **655,360** | **~172 GB** |

**Worst-corner total and why TP=4**

```
Total (BF16, B=64 × 10K)  ≈  (65.5 GB weights + 172 GB KV) × 1.15  ≈  273 GB
```

- **273 GB < 384 GB → fits**, but the KV alone (172 GB) and weights (65.5 GB) blow past a single
  96 GB card. **TP=4** pools all 384 GB and shards both weights and KV across the 4 cards:
  per card ≈ `65.5/4 + 172/4 + overhead ≈ 60 GB` < 96 GB (under the default 0.9 utilization budget).
- Using one fixed TP=4 config for the whole sweep also keeps every `(B, K, S, prefix)` point directly comparable.

### 3. Takeaways

- 384 GB makes VRAM a non-issue: **BF16 fits (~273 GB worst corner)** → no quantization required.
- KV is small *per token* thanks to GQA (8 KV heads, 256 KB/token); the worst corner is large only
  because of `B=64 × 10K` (655K tokens).
- The whole sweep runs on stock Qwen3-32B **without implementing diffusion** — the B/K/S pattern is emulated.

### Sources

- KV cache formula: [Brenndoerfer](https://mbrenndoerfer.com/writing/kv-cache-memory-calculation-llm-inference-gpu), [Lyceum Technology](https://lyceum.technology/magazine/kv-cache-memory-calculation-llm/)
- VRAM = weights + KV + overhead: [BentoML LLM Inference Handbook](https://bentoml.com/llm/getting-started/calculating-gpu-memory-for-llms), [Anyscale Docs](https://docs.anyscale.com/llm/batch-inference/resource-allocation/gpu-memory)
- INT4 / INT8 quantization: [VRLA Tech](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/), [Hivenet](https://www.hivenet.com/post/llm-quantization-guide)
- Qwen3-32B: [HF Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)

---

## K-over-S multi-token decode — design (Milestone 2)

Emulate "**commit K tokens every S steps**" on vanilla Qwen3-32B to find when `K>1` beats the
`(1,1)` AR baseline. **Decode-stage system cost only** (throughput/latency), *not* quality — so the
token ids are arbitrary. Same model / TP=4 / random per-request prefix (1K, 10K) / **256 committed**
output tokens as the vanilla sweep.

### Mechanism (block-structured decode)

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

### Decisions (confirmed)

1. **Intra-block mask = causal** (AR-within-block), not bidirectional. Maps directly onto flash-attn's
   `causal=True` varlen path / the merged chunked-prefill primitive; a bidirectional-suffix mask is not
   a simple flash-attn flag.
2. **Fixed placeholder ids** fed to the K positions for all S steps (no feedback of per-step
   predictions). Quality is irrelevant, and this keeps each of the S steps identical compute.

### What changes (engine modification — reuses chunked-prefill)

- **Scheduler** (`nanovllm/engine/scheduler.py`): schedule `K` query positions/seq/step; track each
  block's step counter `s ∈ [1, S]`; **commit** (persist KV + `L += K`) only at `s == S`; finish when
  256 committed.
- **Attention decode branch** (`nanovllm/layers/attention.py`): process the K block like a *causal
  prefill chunk* against cache `[0, L)`; **suppress the KV-cache store for `s < S`, store at `s == S`**.
- **`bench_decode.py`**: add `--k`, `--s`; loop `256/K` blocks × `S` steps; still time only the decode
  window; throughput = committed `256·B / time`; per-token latency. Add columns `k, s, eff (=K/S)`.
- **Sweep**: extend to the **12 valid (K,S) pairs × B × prefix** (96 runs). `(1,1)` is shared with the
  vanilla baseline (`vanilla.csv`).

### Why this is the measurement

Per step does `B×K` query rows × S forwards to land K tokens (**more FLOPs**), but the **KV cache grows
the same as AR** (only `+K` per committed block; scratch positions never persist). Small-batch decode is
memory-bandwidth-bound and underuses the GPU, so packing `B×K` rows per forward can be near-free —
until compute saturates. **Where `K>1` beats `(1,1)` across `(B, K, S, prefix)` is the result.**

### Open / later

- Exact "scratch" KV handling for `s < S` (recompute-in-attention vs temporary cache slots) — pick the
  simplest the flash-attn path allows.
- The S forwards are identical compute (fixed ids), but we run them for real to capture per-step overhead.

---

## K-over-S implementation — stages & mechanism

Implements the decode path designed above, **reusing nano-vLLM's chunked-prefill attention** instead of
writing a new attention kernel.

### Key insight

A K-over-S **block step is just a chunked-prefill step over K query positions**:
- `flash_attn_varlen_func(causal=True, block_table=…)` already does "K causal queries attend to
  `[0, L+K)` paged KV" (this is nano-vLLM's prefix-cache prefill path).
- `store_kvcache` already **skips** cache slots set to `-1`.

So K-over-S needs **no attention change**: feed K placeholder positions `[L, L+K)`; set
`slot_mapping` = real slots on the **last** of the S steps (persist + commit) and `-1` otherwise
(the K positions stay "scratch", recomputed each step). Causal intra-block falls out of `causal=True`.

### Code changes (all reuse existing machinery)

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

### Stages

1. **(done)** demand.md harness — schema, two sweeps, 4-figure plotter, OOM/skip reporting.
2. the engine path above — implemented, **UNVALIDATED until a GPU is available.**
3. **(on GPU once flash-attn builds)** smoke a tiny config (e.g. `K=2,S=1,B=1,prefix=128`),
   fix any bookkeeping, then run `sweep_full.sh`.

### Caveats to validate on GPU

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

---

## Directory map & quick start

Measures **decode throughput & latency** of Qwen3-32B driven in the (B, K, S) decode pattern on
4× RTX PRO 6000 Blackwell (TP=4), UCR HPCC.

| file | what |
|---|---|
| `demand.md` | deliverables spec + consolidated reference (this file) |
| `experiment_plan.md` | the 1-vs-K study, sweep grid, status |
| `HPCC.md` | step-by-step cluster runbook |
| `env.sh` | routes all caches/downloads into `<repo>/.cache/`, loads `.env`, activates `.venv` |
| `setup_env.sh` | one-time: uv venv (py3.12) + torch cu128 + deps + flash-attn |
| `install_flash_attn.sh` | prebuilt-wheel-first, else source compile |
| `build_flash_attn_wheel.sh` | build a reusable flash-attn wheel (ada6000 + blackwell6000) |
| `download_model.sh` | `hf download` into `.cache/models/` |
| `smoke.py` | load Qwen3-32B at TP=4, decode a few tokens |
| `bench_decode.py` | one config: time the decode of 256 tokens for a batch |
| `bench_vark.py` | variable-K (ragged) decode benchmark |
| `vark_schedule.py` | variable-K schedule helper |
| `sweep.sh` | vanilla K=S=1 sweep → `results/vanilla.csv` (resumable) |
| `sweep_full.sh` | full (K,S) grid → `results/sweep.csv` |
| `sweep_ks.py` | (K,S) sweep driver |
| `merge_results.sh` | merge per-array CSV parts into one results CSV |
| `plot_results.py` | tables + 4 demand figures from a results CSV (+ OOM/skip report) |
| `run.slurm` | submit smoke / vanilla / `FULL=1` sweep to `short_gpu` / `gpu:blackwell6000:4` |
| `run_array.slurm` / `run_ks_array.slurm` / `run_vark_array.slurm` | SLURM array variants for the sweeps |
| `results/` | output CSVs + plots (gitignored; delivered via scp) |

### TL;DR

```bash
bash experiments/setup_env.sh           # once: venv + torch + flash-attn
bash experiments/download_model.sh      # once: -> .cache/models/Qwen3-32B
cd experiments
SMOKE=1 sbatch run.slurm                # smoke test (load + decode a few tokens)
sbatch run.slurm                        # the 8-run K=S=1 baseline -> results/vanilla.csv
FULL=1 sbatch run.slurm                 # full (K,S) grid          -> results/sweep.csv
python plot_results.py results/sweep.csv
```
