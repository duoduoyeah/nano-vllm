"""Decode-only throughput/latency benchmark for nano-vLLM (vanilla AR, K=S=1).

One invocation = one config: (model, tensor_parallel_size, prefix_len, output_len, batch).
Each request gets its OWN random token-id prefix (distinct -> no prefix-cache reuse).
We isolate DECODE: run prefill untimed, then time only the decode steps.

Run on HPCC (4 GPUs), NOT this laptop:
    MODEL_PATH=$SCRATCH/models/Qwen3-32B python bench_decode.py \
        --tp 4 --prefix-len 1024 --output-len 256 --batch 64 --out results/vanilla.csv

This is the K=S=1 baseline harness. Multi-token (K>1, S steps) is a later milestone
that needs an engine modification; this file deliberately uses stock nano-vLLM decode.
"""
import argparse
import csv
import os
import time
from random import seed, randint

import torch
from nanovllm import LLM, SamplingParams


def make_prompts(batch: int, prefix_len: int, vocab: int = 10000) -> list[list[int]]:
    # distinct random ids per request -> no shared prefix, no prefix-cache hit reuse
    return [[randint(0, vocab - 1) for _ in range(prefix_len)] for _ in range(batch)]


def run_one(llm, batch: int, prefix_len: int, output_len: int) -> dict:
    # LLM likely subclasses LLMEngine; fall back gracefully. TODO: confirm and simplify.
    engine = getattr(llm, "llm_engine", getattr(llm, "engine", llm))

    prompts = make_prompts(batch, prefix_len)
    sp = [SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=output_len)
          for _ in range(batch)]
    for p, s in zip(prompts, sp):
        engine.add_request(p, s)

    torch.cuda.reset_peak_memory_stats()
    t0 = None  # timer starts at the FIRST decode step (prefill is untimed)
    # step() -> (outputs, num_tokens); num_tokens > 0 == prefill, < 0 == decode.
    # NOTE: with chunked prefill, prefill spans several steps before decode begins.
    while not engine.is_finished():
        _, num_tokens = engine.step()
        if num_tokens < 0 and t0 is None:
            t0 = time.perf_counter()
    decode_time = time.perf_counter() - t0

    total_decode_tokens = batch * output_len
    return {
        "model": os.path.basename(os.environ.get("MODEL_PATH", "")),
        "tp": getattr(getattr(llm, "config", None), "tensor_parallel_size", None),
        "batch": batch,
        "prefix_len": prefix_len,
        "output_len": output_len,
        "decode_time_s": round(decode_time, 4),
        "throughput_tok_s": round(total_decode_tokens / decode_time, 2),   # system: B*T / t
        "latency_ms_per_tok": round(1000 * decode_time / output_len, 3),   # per-request: t / T
        # TODO: with TP>1 this is rank-0 only; aggregate across ranks for true peak.
        "peak_mem_gb_rank0": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--prefix-len", type=int, required=True)   # 1024 or 10240
    ap.add_argument("--output-len", type=int, default=256)     # T (decode length)
    ap.add_argument("--batch", type=int, required=True)        # 1, 4, 16, 64
    ap.add_argument("--out", default="results/vanilla.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed(args.seed)

    assert args.model and os.path.isdir(args.model), \
        f"--model / MODEL_PATH must be a local weights dir (got {args.model!r})"

    llm = LLM(
        args.model,
        enforce_eager=False,
        tensor_parallel_size=args.tp,
        max_model_len=args.prefix_len + args.output_len + 16,
    )
    row = run_one(llm, args.batch, args.prefix_len, args.output_len)
    print(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
