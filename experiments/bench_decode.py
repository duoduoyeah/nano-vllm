"""Decode-only throughput/latency benchmark for nano-vLLM (per experiments/demand.md).

One invocation = one config: (model, tp, K, S, prefix_len, output_len, batch).
Each request gets its OWN random token-id prefix (ids under the model's vocab -> distinct
per request -> no prefix-cache reuse). Prefill runs untimed; only decode is timed.

  K=S=1  -> vanilla AR decode via the production step() loop (cudagraph).
  K>1    -> K-over-S multi-token decode (commit K tokens per S forward passes) via
            LLM.kovers_decode (see kovers_design.md / kovers_impl.md).

Run on a 4-GPU Blackwell node (env.sh sets MODEL_PATH + routes caches to .cache/):
    source experiments/env.sh
    python experiments/bench_decode.py --tp 4 --k 1 --s 1 --prefix-len 1024 \
        --output-len 256 --batch 64 --out results/vanilla.csv

Metrics (demand.md):
    throughput_tok_s   = batch * output_len / decode_time     (system)
    latency_ms_per_tok = 1000 * decode_time / output_len      (per request)
decode_time is the decode phase only (prefill excluded). For K=1 the timer starts at the first
decode step; for K>1 it brackets the K-over-S block loop. Either way it attributes output_len
tokens to that window per demand.md's B*T/decode_time definition.
"""
import argparse
import csv
import json
import os
import time
from random import seed, randint

import torch
from nanovllm import LLM, SamplingParams

# demand.md schema first, then extra provenance/bookkeeping columns.
COLUMNS = ["model", "tp", "K", "S", "batch", "prefix_len", "output_len",
           "decode_time_s", "throughput_tok_s", "latency_ms_per_tok",
           "peak_mem_gb", "eff_tok_per_step",
           "gpu", "seed", "vocab", "status"]


def model_vocab_size(model_dir: str) -> int:
    with open(os.path.join(model_dir, "config.json")) as f:
        return int(json.load(f)["vocab_size"])


def make_prompts(batch: int, prefix_len: int, vocab: int) -> list[list[int]]:
    # distinct random ids per request, under the vocab size -> no shared prefix / cache reuse
    return [[randint(0, vocab - 1) for _ in range(prefix_len)] for _ in range(batch)]


def kv_fits(llm, batch: int, prefix_len: int, output_len: int) -> bool:
    # True iff this run's KV (batch x (prefix+output) tokens) fits the allocated block pool.
    # Lets us record a clean OOM up front (e.g. B=128 x prefix 10240) instead of preempting/hanging.
    bm = llm.scheduler.block_manager
    need = batch * ((prefix_len + output_len + bm.block_size - 1) // bm.block_size)
    have = len(bm.free_block_ids) + len(bm.used_block_ids)
    return need <= have


def add_requests(llm, prompts, output_len):
    # temperature must be > 1e-10 (this build forbids greedy); value is irrelevant to TIMING and
    # ignore_eos fixes the token count, so 1.0 is fine for a decode-cost (not quality) benchmark.
    sp = [SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=output_len)
          for _ in range(len(prompts))]
    for p, s in zip(prompts, sp):
        llm.add_request(p, s)


def run_vanilla_loop(llm, output_len):
    """K=S=1 AR decode via the production step() loop (requests already added).
    Returns (decode_time_s, peak_mem_gb_rank0). Peak is reset at the FIRST decode step so it
    measures the DECODE-phase peak — consistent with kovers_decode (comparable mem column)."""
    # step() -> (outputs, num_tokens); >0 prefill, <0 decode. Timer starts at first decode step.
    t0 = None
    while not llm.is_finished():
        _, num_tokens = llm.step()
        if num_tokens < 0 and t0 is None:
            t0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
    decode_time = (time.perf_counter() - t0) if t0 is not None else float("nan")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return decode_time, peak_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--s", type=int, default=1)
    ap.add_argument("--prefix-len", type=int, required=True)   # 1024 or 10240
    ap.add_argument("--output-len", type=int, default=256)     # T
    ap.add_argument("--batch", type=int, required=True)        # 1, 4, 16, 64
    ap.add_argument("--vocab", type=int, default=None,
                    help="upper bound for random prefix ids; default = model's vocab_size")
    ap.add_argument("--out", default="results/vanilla.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed(args.seed)

    assert args.model and os.path.isdir(args.model), \
        f"--model / MODEL_PATH must be a local weights dir (got {args.model!r}); source env.sh first"
    vocab = args.vocab or model_vocab_size(args.model)
    K, S = args.k, args.s

    row = {
        "model": os.path.basename(os.environ.get("MODEL_PATH", "") or args.model),
        "tp": args.tp, "K": K, "S": S, "batch": args.batch,
        "prefix_len": args.prefix_len, "output_len": args.output_len,
        "decode_time_s": "", "throughput_tok_s": "", "latency_ms_per_tok": "",
        "peak_mem_gb": "", "eff_tok_per_step": round(K / S, 4),
        "gpu": "", "seed": args.seed, "vocab": vocab, "status": "ok",
    }

    try:
        assert K >= 1 and S >= 1 and K / S >= 1, "require K >= 1, S >= 1, K/S >= 1"
        assert args.output_len % K == 0, "output_len must be divisible by K"
        llm = LLM(args.model, enforce_eager=False, tensor_parallel_size=args.tp,
                  max_model_len=args.prefix_len + args.output_len + 16,
                  decode_k=K, max_num_seqs=max(args.batch, 1))   # decode_k=K -> cudagraph seqlen_q=K
        row["gpu"] = torch.cuda.get_device_name(0)
        if not kv_fits(llm, args.batch, args.prefix_len, args.output_len):
            row["status"] = "OOM"
        else:
            add_requests(llm, make_prompts(args.batch, args.prefix_len, vocab), args.output_len)
            if K == 1 and S == 1:
                decode_time, peak_gb = run_vanilla_loop(llm, args.output_len)
            else:
                decode_time, peak_gb = llm.kovers_decode(K, S, args.output_len)
            row["decode_time_s"] = round(decode_time, 4)
            row["throughput_tok_s"] = round(args.batch * args.output_len / decode_time, 2)
            row["latency_ms_per_tok"] = round(1000 * decode_time / args.output_len, 3)
            row["peak_mem_gb"] = round(peak_gb, 2)   # rank-0 only under TP; see TODO
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
    except Exception as e:                       # demand.md: never silently drop a config
        row["status"] = f"error:{type(e).__name__}"

    print(row)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
