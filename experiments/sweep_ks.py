"""Per-(K,S) in-process sweep: load Qwen3-32B ONCE, run all (batch x prefix) configs for one (K,S).

Amortizes the ~1.5 min model reload across a pair's 8 configs (12 loads for the whole grid instead
of 96 fresh processes). Writes one per-config part CSV each (results/parts/K{K}_S{S}_P{P}_B{B}.csv),
skipping any that already exist, so it composes with run_array.slurm / merge_results.sh. The engine
is reset between configs (KV blocks freed, queues cleared) for isolation.

    python experiments/sweep_ks.py --k 2 --s 2 [--output-len 256] [--batches 1,4,16,64] [--prefixes 1024,10240]
"""
import argparse
import csv
import os
from random import seed

import torch
from nanovllm import LLM
import bench_decode as bd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--batches", default="1,4,16,64")
    ap.add_argument("--prefixes", default="1024,10240")
    ap.add_argument("--partsdir", default="results/parts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert args.model and os.path.isdir(args.model), \
        f"--model / MODEL_PATH must be a local weights dir (got {args.model!r}); source env.sh first"
    K, S = args.k, args.s
    batches = [int(x) for x in args.batches.split(",")]
    prefixes = [int(x) for x in args.prefixes.split(",")]
    os.makedirs(args.partsdir, exist_ok=True)

    # Only run configs whose part file is missing; don't even load the model if none remain.
    todo = [(P, B) for P in prefixes for B in batches
            if not os.path.exists(os.path.join(args.partsdir, f"K{K}_S{S}_P{P}_B{B}.csv"))]
    if not todo:
        print(f"all configs for K={K} S={S} already done")
        return

    vocab = bd.model_vocab_size(args.model)
    max_p = max(prefixes)
    # one model load covers every prefix in this pair (KV pool sized to GPU mem regardless).
    # decode_k=K captures cudagraphs for seqlen_q=K (non-eager K>1); max_num_seqs bounds the capture.
    llm = LLM(args.model, enforce_eager=False, tensor_parallel_size=args.tp,
              max_model_len=max_p + args.output_len + 16,
              decode_k=K, max_num_seqs=max(batches))
    gpu = torch.cuda.get_device_name(0)

    for (P, B) in todo:
        out = os.path.join(args.partsdir, f"K{K}_S{S}_P{P}_B{B}.csv")
        seed(args.seed)
        row = {
            "model": os.path.basename(args.model), "tp": args.tp, "K": K, "S": S, "batch": B,
            "prefix_len": P, "output_len": args.output_len, "decode_time_s": "",
            "throughput_tok_s": "", "latency_ms_per_tok": "", "peak_mem_gb": "",
            "eff_tok_per_step": round(K / S, 4), "gpu": gpu, "seed": args.seed,
            "vocab": vocab, "status": "ok",
        }
        try:
            assert args.output_len % K == 0, "output_len must be divisible by K"
            if not bd.kv_fits(llm, B, P, args.output_len):
                row["status"] = "OOM"                     # KV won't fit (e.g. B=128 x prefix 10240)
            else:
                bd.add_requests(llm, bd.make_prompts(B, P, vocab), args.output_len)
                if K == 1 and S == 1:
                    dt, peak = bd.run_vanilla_loop(llm, args.output_len)
                else:
                    dt, peak = llm.kovers_decode(K, S, args.output_len)
                row["decode_time_s"] = round(dt, 4)
                row["throughput_tok_s"] = round(B * args.output_len / dt, 2)
                row["latency_ms_per_tok"] = round(1000 * dt / args.output_len, 3)
                row["peak_mem_gb"] = round(peak, 2)
        except torch.cuda.OutOfMemoryError:
            row["status"] = "OOM"
        except Exception as e:                       # never silently drop a config (demand.md)
            row["status"] = f"error:{type(e).__name__}"
        finally:
            llm.reset()                              # clean state for the next config in this process
        print(row)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=bd.COLUMNS)
            w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()
