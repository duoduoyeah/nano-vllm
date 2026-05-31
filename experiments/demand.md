# Experiment Demands — Decode Benchmark

What we need delivered from HPCC. Run the configs, record the measurements, hand back
the data and figures below. Setup details live in `experiment_plan.md`; how to run on
the cluster is in `HPCC.md`.

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
   (K query positions per forward × S forwards per block); not yet implemented — extend
   the decode branch in `nanovllm/layers/attention.py` + the scheduler.

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
