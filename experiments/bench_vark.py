"""Ragged variable-K decode benchmark (S=1) for nano-vLLM.

One invocation = one config. Two modes, written to the SAME CSV so they line up:
  --mode ragged   each request commits a VARIABLE K in [k_min,k_max] per step, balanced so every step
                  totals the same query rows (avg_k = output_len/num_steps); paged VARLEN path +
                  single cudagraph (LLM.kovers_decode_vark). The experiment's subject.
  --mode uniform  every request commits avg_k tokens every step (K=avg_k, S=1) via the uniform K path
                  (LLM.kovers_decode). The same-average-work reference: ragged vs uniform isolates the
                  cost of raggedness; both are cudagraph-replayed.

Compare against the existing K=1 AR baseline (vanilla.csv) for the "beats 1-token AR?" question.
Each request gets a distinct random prefix (no cache reuse); prefill untimed; only decode is timed.

    source experiments/env.sh
    python experiments/bench_vark.py --tp 4 --mode ragged --batch 64 --prefix-len 1024 \
        --output-len 256 --num-steps 64 --k-min 1 --k-max 8 --out results/vark.csv
"""
import argparse
import csv
import os
import time

import torch
from nanovllm import LLM
import bench_decode as bd
from vark_schedule import build_balanced_schedule, describe

COLUMNS = ["model", "tp", "mode", "k_min", "k_max", "avg_k", "num_steps", "batch",
           "prefix_len", "output_len", "decode_time_s", "throughput_tok_s",
           "latency_s_per_req", "peak_mem_gb", "cudagraph", "gpu", "seed", "vocab", "status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--mode", choices=["ragged", "uniform"], default="ragged")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--prefix-len", type=int, required=True)     # 1024 or 10240
    ap.add_argument("--output-len", type=int, default=256)       # T (per request)
    ap.add_argument("--num-steps", type=int, default=64)         # fixed steps; avg_k = output_len/num_steps
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--out", default="results/vark.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert args.model and os.path.isdir(args.model), \
        f"--model / MODEL_PATH must be a local weights dir (got {args.model!r}); source env.sh first"
    assert args.output_len % args.num_steps == 0, "output_len must be divisible by num_steps"
    vocab = args.vocab or bd.model_vocab_size(args.model)
    avg_k = args.output_len // args.num_steps
    total_q = args.batch * avg_k                                 # constant query rows per step

    row = {
        "model": os.path.basename(os.environ.get("MODEL_PATH", "") or args.model),
        "tp": args.tp, "mode": args.mode, "k_min": args.k_min, "k_max": args.k_max,
        "avg_k": avg_k, "num_steps": args.num_steps, "batch": args.batch,
        "prefix_len": args.prefix_len, "output_len": args.output_len,
        "decode_time_s": "", "throughput_tok_s": "", "latency_s_per_req": "",
        "peak_mem_gb": "", "cudagraph": "", "gpu": "", "seed": args.seed,
        "vocab": vocab, "status": "ok",
    }

    try:
        if args.mode == "ragged":
            llm = LLM(args.model, enforce_eager=False, tensor_parallel_size=args.tp,
                      max_model_len=args.prefix_len + args.output_len + 16,
                      decode_varlen=True, varlen_total_q=total_q, varlen_k_max=args.k_max,
                      max_num_seqs=args.batch)
            row["cudagraph"] = "yes" if hasattr(llm.model_runner, "varlen_graph") else "eager"
        else:  # uniform reference: K=avg_k, S=1 via the uniform decode graph
            assert args.output_len % avg_k == 0
            llm = LLM(args.model, enforce_eager=False, tensor_parallel_size=args.tp,
                      max_model_len=args.prefix_len + args.output_len + 16,
                      decode_k=avg_k, max_num_seqs=args.batch)
            row["cudagraph"] = "yes"
        row["gpu"] = torch.cuda.get_device_name(0)

        if not bd.kv_fits(llm, args.batch, args.prefix_len, args.output_len):
            row["status"] = "OOM"
        else:
            bd.add_requests(llm, bd.make_prompts(args.batch, args.prefix_len, vocab), args.output_len)
            if args.mode == "ragged":
                M = build_balanced_schedule(args.batch, args.output_len, args.k_min, args.k_max,
                                            args.num_steps, args.seed)
                lo, hi, mean, totals = describe(M)
                print(f"schedule: K in [{lo},{hi}], mean {mean:.3f}, per-step totals {totals}")
                assert totals == [total_q], "schedule not column-balanced -> graph won't apply"
                dt, peak = llm.kovers_decode_vark(M, args.output_len)
            else:
                dt, peak = llm.kovers_decode(avg_k, 1, args.output_len)
            row["decode_time_s"] = round(dt, 4)
            row["throughput_tok_s"] = round(args.batch * args.output_len / dt, 2)
            row["latency_s_per_req"] = round(dt, 4)        # lockstep static batch: 256 tok/req in dt s
            row["peak_mem_gb"] = round(peak, 2)
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
    except Exception as e:                                  # never silently drop a config (demand.md)
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
