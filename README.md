# Nano-vLLM — CS213 multi-token decode study

A fork of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (a lightweight
vLLM implementation in ~1,200 lines of Python) used as the base engine for our
CS213 course project on **multi-token (K>1) decoding**.

The course work lives in **[`experiments/`](experiments/README.md)** — start there.

It covers:
- the **1-vs-K decode study** (when does K>1 decoding beat K=1?) and sweep grid
- the **K-over-S** multi-token decode engine path
- VRAM math for Qwen3-32B and the UCR HPCC cluster runbook
- the benchmark harness, sweep scripts, and result plots

See [`experiments/README.md`](experiments/README.md) for the full layout and a TL;DR
to reproduce the sweep.

## Upstream

This is built on nano-vllm. For the original engine, its features, install, and
benchmarks, see the [upstream repository](https://github.com/GeeeekExplorer/nano-vllm).
