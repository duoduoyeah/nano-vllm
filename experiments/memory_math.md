# Model Memory Math — VRAM for the Sweep

How much GPU memory the model needs, and whether it fits on
**4× RTX PRO 6000 Blackwell (96 GB each = 384 GB total)**.

## 1. Equations

```
Total VRAM  ≈  Weights  +  KV cache  +  overhead
```

### Weights

```
W = N_total × b
```

- `N_total` = total parameter count. Qwen3-32B is **dense**, so every parameter is resident and active.
- `b` = bytes per parameter:

| precision | b (bytes/param) |
|---|---|
| FP32 | 4 |
| FP16 / BF16 | 2 |
| INT8 | 1 |
| INT4 | 0.5 |

### KV cache

```
KV = 2 · L · H_kv · d_head · L_seq · B · b_kv
```

- `2` = Key + Value
- `L` = layers · `H_kv` = KV heads (GQA: ≪ attention heads) · `d_head` = head dim
- `L_seq` = prefix + generated (≈ prefix here) · `B` = batch · `b_kv` = bytes/elem (FP16 = 2, FP8 = 1)
- Only term that grows with the sweep — scales with `B × L_seq`.

### Overhead

Activations + framework ≈ **+10–20%**. Inference activations are small. Our B/K/S decode runs
`B×K` query positions per forward (vs `B×1` for AR), inflating *transient activation* but not KV.

## 2. The model — Qwen3-32B (dense AR, B/K/S emulation)

We don't run a real diffusion / multi-token model. We take a **normal autoregressive model that
nano-vLLM already supports** and **impose the (B, K, S) decode pattern** on it:

- AR baseline `(K=1, S=1)`: 1 token/step/request (native).
- Multi-token: present **K query positions per request per forward**, repeat **S forwards** before committing K tokens.

We measure **system cost (throughput/latency), not output quality** — so the emulated compute/memory
pattern is what matters; the generated text can be meaningless. The merged **chunked-prefill** path
provides the multi-position-forward primitive this needs, so we avoid implementing diffusion.

**Config** (`config.json`): `L=64`, `hidden=5120`, 64 attention heads, `H_kv=8` (GQA), `d_head=128`,
`vocab=151936`, native ctx `40960`, BF16 release. **≈ 32.8B params** (dense).

### Weights

| | INT4 | INT8 | BF16 |
|---|---|---|---|
| **Weights** (`32.8B × b`) | ~16.4 GB | ~32.8 GB | **~65.5 GB** |

BF16 is the released format and the zero-effort path (matches the ~65.5 GB on-disk download).
Quantization is optional — only for extra headroom or if we want to *study* quantization.

### KV cache

```
KV/token (FP16) = 2 · 64 · 8 · 128 · 2  =  262,144 B  =  256 KB/token
```

| sweep corner (FP16 KV) | tokens (B × L_seq) | KV |
|---|---|---|
| B=1 × 1K   | 1,024      | ~0.27 GB |
| B=64 × 1K  | 65,536     | ~17 GB |
| B=1 × 10K  | 10,240     | ~2.7 GB |
| **B=64 × 10K** (worst) | **655,360** | **~172 GB** |

### Worst-corner total and why TP=4

```
Total (BF16, B=64 × 10K)  ≈  (65.5 GB weights + 172 GB KV) × 1.15  ≈  273 GB
```

- **273 GB < 384 GB → fits**, but the KV alone (172 GB) and weights (65.5 GB) blow past a single
  96 GB card. **TP=4** pools all 384 GB and shards both weights and KV across the 4 cards:
  per card ≈ `65.5/4 + 172/4 + overhead ≈ 60 GB` < 96 GB (under the default 0.9 utilization budget).
- Using one fixed TP=4 config for the whole sweep also keeps every `(B, K, S, prefix)` point directly comparable.

## 3. Takeaways

- 384 GB makes VRAM a non-issue: **BF16 fits (~273 GB worst corner)** → no quantization required.
- KV is small *per token* thanks to GQA (8 KV heads, 256 KB/token); the worst corner is large only
  because of `B=64 × 10K` (655K tokens).
- The whole sweep runs on stock Qwen3-32B **without implementing diffusion** — the B/K/S pattern is emulated.

## Sources

- KV cache formula: [Brenndoerfer](https://mbrenndoerfer.com/writing/kv-cache-memory-calculation-llm-inference-gpu), [Lyceum Technology](https://lyceum.technology/magazine/kv-cache-memory-calculation-llm/)
- VRAM = weights + KV + overhead: [BentoML LLM Inference Handbook](https://bentoml.com/llm/getting-started/calculating-gpu-memory-for-llms), [Anyscale Docs](https://docs.anyscale.com/llm/batch-inference/resource-allocation/gpu-memory)
- INT4 / INT8 quantization: [VRLA Tech](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/), [Hivenet](https://www.hivenet.com/post/llm-quantization-guide)
- Qwen3-32B: [HF Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)
