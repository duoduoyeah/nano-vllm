# HPCC Runbook — Vanilla Decode Sweep (Qwen3-32B, TP=4)

What the **HPCC side** does to run the K=S=1 decode sweep. Code is written on the
laptop and pushed to GitHub; HPCC pulls and runs. Fill the `<...>` placeholders with
your cluster's values (`sinfo`, `module avail`).

> The decode benchmark is `experiments/bench_decode.py`; `sweep.sh` loops it; `run.slurm`
> submits it. K=S=1 = plain AR decode (no engine changes yet).

## 0. Where things go
- **Weights + HF cache → scratch / bigdata** (large quota), NOT `$HOME`.
  `export SCRATCH=<your scratch or /bigdata path>`
- **Code → anywhere** (home is fine; it's small).

## 1. Get the code (SSH)

Get a GitHub SSH key working on HPCC — two ways:
- **Agent forwarding (recommended — no private key on the shared cluster):** from the
  laptop, `ssh -A <user>@<hpcc>` (key loaded: check `ssh-add -l`).
- **Key on HPCC:** `ssh-keygen -t ed25519 -C "<email>"`, then add `~/.ssh/id_ed25519.pub`
  to GitHub (Settings → SSH keys, or as a repo Deploy key). One-time.

Verify, then clone:
```bash
ssh -T git@github.com                                      # should greet you by username
git clone -b project git@github.com:duoduoyeah/nano-vllm.git
cd nano-vllm                                               # (or: git pull, if already cloned)
```
> The fork mirrors a public repo: if it's **public**, skip SSH and clone over HTTPS
> (`https://github.com/duoduoyeah/nano-vllm.git`, no auth). Use SSH if it's **private**
> or you'll **push** from HPCC.

## 2. Python env (Python 3.10–3.12; NOT 3.13)
```bash
module load <CUDA_MODULE> <PYTHON_MODULE>          # check `module avail cuda`
python -m venv $SCRATCH/envs/nanovllm && source $SCRATCH/envs/nanovllm/bin/activate
pip install --upgrade pip
pip install "torch>=2.4.0"                          # use the cluster-recommended CUDA build
pip install flash-attn --no-build-isolation         # THE gotcha: needs nvcc + torch present; slow
pip install -e .                                    # triton, transformers, xxhash, + package
```
> `flash-attn` is required (decode calls `flash_attn_with_kvcache`). If the source build
> fails, install a **prebuilt wheel** matching your python + torch + CUDA + cxx11abi from
> the flash-attn GitHub releases. Build on a node that has `nvcc` (CUDA toolkit module).

## 3. Download Qwen3-32B (to scratch, as a local dir)
```bash
HF_HUB_CACHE=$SCRATCH/hf huggingface-cli download Qwen/Qwen3-32B \
    --local-dir $SCRATCH/models/Qwen3-32B
export MODEL_PATH=$SCRATCH/models/Qwen3-32B
```
nano-vLLM needs a **local directory** (it asserts `os.path.isdir(model)`), not a repo id.
~64 GB (BF16).

## 4. Smoke test (4 GPUs)
On an interactive 4-GPU node (or via sbatch):
```bash
MODEL_PATH=$SCRATCH/models/Qwen3-32B TP=4 python experiments/smoke.py
```
Expect `OK: [...]` → weights load + TP=4 path work.

## 5. Run the sweep
Edit `experiments/run.slurm` placeholders (`<GPU_PARTITION>`, `--gres`, `<CUDA_MODULE>`,
`<PATH_TO_ENV>`), then:
```bash
cd experiments
sbatch run.slurm
```
Or interactively on a 4-GPU node:
```bash
cd experiments
MODEL_PATH=$SCRATCH/models/Qwen3-32B TP=4 OUTPUT_LEN=256 bash sweep.sh
```

## 6. Results
- Land in `experiments/results/vanilla.csv` — one row per `prefix × batch`.
- Columns: `batch, prefix_len, output_len, decode_time_s, throughput_tok_s, latency_ms_per_tok, peak_mem_gb_rank0`.
- Copy back to laptop to plot:
  `scp <user>@<hpcc>:<path>/experiments/results/vanilla.csv .`

## Sweep covered
`prefix ∈ {1024, 10240} × batch ∈ {1, 4, 16, 64}`, `output_len = 256`, `TP = 4`.
Baseline only (K=S=1); multi-token K/S decoding is a later milestone (engine modification).
