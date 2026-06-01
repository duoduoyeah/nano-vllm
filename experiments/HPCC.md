# HPCC Runbook — Vanilla Decode Sweep (Qwen3-32B, TP=4, Blackwell)

What the **UCR HPCC side** does to run the `K=S=1` decode sweep. Everything external
(model weights, HF/torch/triton caches) is routed into repo-local **`.cache/`** (gitignored,
disposable); secrets live in **`.env`**. One script, `experiments/env.sh`, wires this up and is
sourced by every other script.

> Harness: `bench_decode.py` (one config), `sweep.sh` (loops it), `run.slurm` (submits it).
> `K=S=1` = plain AR decode (no engine changes yet); multi-token K/S is a later milestone.

## 0. Layout & conventions

- **Caches/weights → `<repo>/.cache/`** (gitignored). `env.sh` exports `HF_HOME`, `HF_HUB_CACHE`,
  `TORCH_HOME`, `TRITON_CACHE_DIR`, `XDG_CACHE_HOME`, `UV_CACHE_DIR` into it, plus `MODELS_DIR`
  (`.cache/models`) where downloaded weights land. `rm -rf .cache` reclaims all of it.
- **Secrets → `<repo>/.env`** (gitignored): `HF_TOKEN=...`. Copy `.env.example` → `.env` and fill in.
  `env.sh` loads it via `set -a; source .env; set +a`. Qwen3-32B is public so the token is optional,
  but it's kept for gated models / rate limits.
- **venv → `<repo>/.venv`** (gitignored), created with `uv`. `env.sh` auto-activates it if present.

## 1. Get the code (SSH)

```bash
ssh -T git@github.com                                       # greets you by username
git clone -b project git@github.com:duoduoyeah/nano-vllm.git
cd nano-vllm
cp .env.example .env        # then edit .env and paste your HF_TOKEN
```
(Repo is a fork of a public repo — HTTPS clone works too if you won't push:
`https://github.com/duoduoyeah/nano-vllm.git`.)

## 2. Provision the environment (one time)

```bash
bash experiments/setup_env.sh
```
This uses **uv** to: create `.venv` (Python 3.12), install **torch 2.9.1 (cu128, Blackwell sm_120)**
and nano-vLLM's deps, then **compile flash-attn 2.8.3 from source**.

> **flash-attn must be compiled, not pip-installed from a wheel.** Rocky 8 ships glibc 2.28, but
> published flash-attn wheels (incl. the ones flash-attn's own `setup.py` auto-downloads from
> Dao-AILab releases) need glibc ≥ 2.32, so they fail to load. `install_flash_attn.sh` sets
> `FLASH_ATTENTION_FORCE_BUILD=TRUE` and `--no-binary` to force a real source build against
> `cuda/12.8` + `gcc/11.5.0` for archs `8.9;12.0` (ada6000 + blackwell6000). The compile is long and
> RAM-hungry — best run as a CPU batch job (`epyc`, many cores), not on a busy login node.

**Reusable wheel (build once, install fast forever).** `bash experiments/build_flash_attn_wheel.sh`
compiles a wheel into `.cache/wheels/` covering Ada6000 (sm_89) + Blackwell (sm_120). `setup_env.sh`
/ `install_flash_attn.sh` then auto-install it (or a copy published as a release asset, via
`FLASH_ATTN_WHEEL_URL=...`) in seconds — skipping the compile on every future clone/teammate.
The wheel is pinned to this Python (cp312) + torch (2.9.1+cu128) + glibc (2.28); a `.meta.txt`
sidecar records exactly that.

## 3. Download Qwen3-32B (→ `.cache/models`)

```bash
bash experiments/download_model.sh            # Qwen/Qwen3-32B by default
```
Lands in `.cache/models/Qwen3-32B` (~65.5 GB, BF16). nano-vLLM needs a local directory
(`Config` asserts `os.path.isdir(model)`); `env.sh` sets `MODEL_PATH` to this path automatically.

## 4. Smoke test (4 GPUs)

```bash
cd experiments
SMOKE=1 sbatch run.slurm                      # batch
# or interactively on a 4-GPU Blackwell node:
srun --partition=short_gpu --qos=short_gpu --gres=gpu:blackwell6000:4 \
     --cpus-per-task=16 --mem=200G --time=0:30:00 --pty bash -l
source experiments/env.sh && TP=4 python experiments/smoke.py
```
Expect `OK: [...]` → weights load + the TP=4 path work end to end.

## 5. Run the sweep

```bash
cd experiments
sbatch run.slurm            # vanilla K=S=1 sweep -> results/vanilla.csv  (demand.md priority 1)
FULL=1 sbatch run.slurm     # full (K,S) sweep    -> results/sweep.csv    (demand.md priority 2)
```
Or interactively on a 4-GPU node:
```bash
source experiments/env.sh
cd experiments && TP=4 OUTPUT_LEN=256 bash sweep.sh        # vanilla
cd experiments && TP=4 OUTPUT_LEN=256 bash sweep_full.sh   # full (K,S)
```
> Full sweep: K=1,S=1 runs the production decode; K>1 uses the K-over-S engine path
> (`kovers_impl.md`) — implemented but **unvalidated until the first GPU run**. Failures are
> recorded as `error:…`; no config is silently dropped.

### GPU queue / partitions (UCR HPCC)

Blackwell6000 (RTX PRO 6000, 96 GB, sm_120) is on **gpu13/gpu14**:

| Partition | QOS | MaxWall | Notes |
|---|---|---|---|
| `short_gpu` | `short_gpu` | **2h** | default here; `--gres=gpu:blackwell6000:4` |
| `raise` | `raise` | 30d | account-gated (RAISE@UCR); for the long sweep |

`run.slurm` defaults to `short_gpu`. For the 30d queue: `sbatch --partition=raise --qos=raise --time=8:00:00 run.slurm`.
Helpers (add to `~/.bashrc`): `black-free` (cards available now), `black-who`, `black-queue`.

## 6. Results (deliverables — `demand.md`)

- `results/vanilla.csv` (K=S=1) and `results/sweep.csv` (full grid), one row per config.
- Columns (demand.md schema, then provenance): `model, tp, K, S, batch, prefix_len, output_len,
  decode_time_s, throughput_tok_s, latency_ms_per_tok, peak_mem_gb, eff_tok_per_step` + `gpu, seed, vocab, status`.
  - `throughput_tok_s = B·T / decode_time`, `latency_ms_per_tok = decode_time / T` (decode-only; prefill untimed).
  - `peak_mem_gb` is rank-0 under TP=4 (TODO: aggregate across ranks).
  - `status` = `ok` / `OOM` / `skipped:…` / `error:…` — **every config gets a row; none is silently dropped.**
- Hardware context to report with the numbers: **TP=4 over PCIe, no NVLink**.
- Figures: `python plot_results.py results/sweep.csv` → markdown tables + 4 PNGs
  (throughput/latency vs batch per K; throughput vs K; equal-K/S wider-K-vs-more-S) + a list of OOM/skipped configs.
- Copy back to laptop (use the `*-login` SSH alias):
  `scp 'hpcc-sli-login:<path>/experiments/results/*.csv' .`

## Sweep covered

`prefix ∈ {1024, 10240} × batch ∈ {1, 4, 16, 64}`, `output_len = 256`, `TP = 4`.
Baseline only (`K=S=1`); multi-token K/S decoding is a later milestone (engine modification).
