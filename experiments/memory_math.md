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

- `N_total` = **total** parameter count. For **MoE**, this is *all* experts (they must be resident). Active params reduce compute/FLOPs, **not** weight memory.
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

Activations + framework ≈ **+10–20%**. Inference activations are small; MoE computes only the active experts. Our B/K/S decode runs `B×K` query positions per forward (vs `B×1` for AR), inflating *transient activation* but not KV.

## 2. Models we'll use

### LLaDA2.0-flash (100B) — native masked-diffusion reference

Config: `L=32`, `H_kv=4`, `d_head=128`, 256 experts (8 active), BF16 release · 100B total / 6.1B active.

| | INT4 | INT8 | BF16 |
|---|---|---|---|
| **Weights** (`100B × b`) | 50 GB | 100 GB | 200 GB |

KV/token (FP16) = `2·32·4·128·2` = **64 KB** (upper bound — flash uses sliding-window attention on most layers, so the real value is lower).
Worst sweep corner `B=64 × 10K` ≈ 640K tokens → **~42 GB** KV.

| Worst-corner total (W + KV + 15%) | INT4 | INT8 | BF16 |
|---|---|---|---|
| total | ~106 GB | ~163 GB | ~278 GB |
| cards (96 GB each) | 2 | 2–3 | 3–4 |

→ **Even BF16 fits in 384 GB.** Quantization is *optional* for flash; released weights are BF16, so BF16 is the zero-effort path. INT8 for extra headroom; INT4 unnecessary.

### Vanilla AR model (B/K/S emulation) — practical path

We may not run LLaDA's diffusion decode at all. Instead, take a **normal autoregressive model that nano-vLLM already supports** and **impose the (B, K, S) decode pattern** on it:

- AR baseline `(K=1, S=1)`: 1 token/step/request (native).
- Multi-token: present **K query positions per request per forward**, repeat **S forwards** before committing K tokens.

We measure **system cost (throughput/latency), not output quality** — so the emulated compute/memory pattern is what matters; the generated text can be meaningless. The recently-merged **chunked-prefill** path provides the multi-position-forward primitive this needs, so we avoid implementing diffusion.

Memory uses the same equations; numbers depend on the chosen model.

> **TBD — which AR model.** Must be nano-vLLM-supported (Qwen2/Qwen3 family). Pick the size to match the study (≈100B to mirror flash, or smaller for faster iteration). Fill weights/KV here once chosen.

## 3. Takeaways

- 384 GB makes VRAM a non-issue: **BF16 flash fits (~278 GB)** → no quantization required. INT8/INT4 only if we want headroom or want to *study* quantization.
- KV is small (GQA, 4 KV heads): worst corner only ~42 GB.
- Vanilla-AR emulation runs the whole sweep on a nano-vLLM-supported model **without implementing diffusion** — model choice still open.

## Sources

- KV cache formula: [Brenndoerfer](https://mbrenndoerfer.com/writing/kv-cache-memory-calculation-llm-inference-gpu), [Lyceum Technology](https://lyceum.technology/magazine/kv-cache-memory-calculation-llm/)
- VRAM = weights + KV + overhead: [BentoML LLM Inference Handbook](https://bentoml.com/llm/getting-started/calculating-gpu-memory-for-llms), [Anyscale Docs](https://docs.anyscale.com/llm/batch-inference/resource-allocation/gpu-memory)
- INT4 / INT8 quantization: [VRLA Tech](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/), [Hivenet](https://www.hivenet.com/post/llm-quantization-guide)
- LLaDA 2.0: [arXiv 2512.15745](https://arxiv.org/abs/2512.15745), [HF LLaDA2.0-flash](https://huggingface.co/inclusionAI/LLaDA2.0-flash), [GitHub inclusionAI/LLaDA2.X](https://github.com/inclusionAI/LLaDA2.X)
