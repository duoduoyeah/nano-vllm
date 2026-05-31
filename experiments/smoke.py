"""Smoke test: confirm Qwen3-32B loads and decodes on HPCC at TP=4.

Run on HPCC (4 GPUs), NOT this laptop:
    MODEL_PATH=$SCRATCH/models/Qwen3-32B TP=4 python smoke.py

Passes a few raw token ids (no tokenizer needed) and decodes 16 tokens.
If this prints "OK: ..." the model + tensor-parallel path work end to end.
"""
import os

from nanovllm import LLM, SamplingParams


def main():
    path = os.environ.get("MODEL_PATH") or os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    assert os.path.isdir(path), f"MODEL_PATH must be a local weights dir (got {path!r})"
    tp = int(os.environ.get("TP", "4"))

    llm = LLM(path, enforce_eager=True, tensor_parallel_size=tp, max_model_len=2048)
    out = llm.generate(
        [[1, 2, 3, 4, 5]],
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=16),
    )
    print("OK:", out)


if __name__ == "__main__":
    main()
