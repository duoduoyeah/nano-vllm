# experiments/ — vanilla decode sweep (Milestone 1)

Measures **decode throughput & latency** of Qwen3-32B driven in (B, K=1, S=1) mode on
4× RTX PRO 6000 Blackwell (TP=4), UCR HPCC. The study question (when does multi-token
K>1 decoding beat K=1?) and sweep grid are in `experiment_plan.md`; VRAM math is in
`memory_math.md`; the cluster runbook is `HPCC.md`.

One data point = take a batch of **B** requests, each with its **own random prefix**
(`prefix_len` ids drawn under the model's vocab → no prefix-cache reuse), run prefill
**untimed**, then **time the decode of 256 tokens** for the whole batch.

## Files

| file | what |
|---|---|
| `experiment_plan.md` | the 1-vs-K study, sweep grid, status |
| `demand.md` | deliverables spec — configs, CSV schema, figures, rules |
| `kovers_design.md` | K-over-S (K>1) decode design + confirmed decisions (Milestone 2) |
| `memory_math.md` | VRAM math (weights + KV) for Qwen3-32B |
| `HPCC.md` | step-by-step cluster runbook |
| `env.sh` | routes all caches/downloads into `<repo>/.cache/`, loads `.env`, activates `.venv` |
| `setup_env.sh` | one-time: uv venv (py3.12) + torch cu128 + deps + flash-attn |
| `install_flash_attn.sh` | prebuilt-wheel-first, else source compile |
| `build_flash_attn_wheel.sh` | build a reusable flash-attn wheel (ada6000 + blackwell6000) |
| `download_model.sh` | `hf download` into `.cache/models/` |
| `smoke.py` | load Qwen3-32B at TP=4, decode 16 tokens |
| `bench_decode.py` | one config: time the decode of 256 tokens for a batch |
| `sweep.sh` | vanilla K=S=1 sweep → `results/vanilla.csv` (resumable) |
| `sweep_full.sh` | full (K,S) grid → `results/sweep.csv` (K>1 via the K-over-S engine path; unvalidated until GPU) |
| `run.slurm` | submit smoke / vanilla / `FULL=1` sweep to `short_gpu` / `gpu:blackwell6000:4` |
| `plot_results.py` | tables + 4 demand figures from a results CSV (+ OOM/skip report) |
| `results/` | output CSVs + plots |

## TL;DR

```bash
bash experiments/setup_env.sh           # once: venv + torch + flash-attn
bash experiments/download_model.sh      # once: -> .cache/models/Qwen3-32B
cd experiments
SMOKE=1 sbatch run.slurm                # smoke test (load + decode 16 tokens)
sbatch run.slurm                        # the 8-run K=S=1 baseline -> results/vanilla.csv
FULL=1 sbatch run.slurm                 # full (K,S) grid          -> results/sweep.csv
python plot_results.py results/sweep.csv
```
