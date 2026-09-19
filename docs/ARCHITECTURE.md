# Architecture: why this shape for sub-1B chat

## TL;DR

Deep-thin decoder-only Transformer (MobileLLM-style), GQA, sliding-window +
periodic global attention, SwiGLU, RMSNorm, QK-Norm, RoPE θ=100k, tied
embeddings, 32k English BPE. No MoE (total params are the score), no biases,
no dropout in pretraining.

## The ladder

| model | L | d | Q/KV heads | ffn | params |
|---|---|---|---|---|---|
| 150m | 30 | 576 | 9/3 | 1536 | ~125M |
| 350m | 32 | 960 | 12/4 | 2560 | ~345M |
| 600m | 32 | 1248 | 16/4 | 3328 | ~590M |

Computed by `ModelConfig.num_params` (embeddings counted once — tied).

## Decision log

1. **Deep & thin over shallow & wide.** MobileLLM (Meta, 2024) showed that at
   <1B, depth buys more reasoning/chat quality per parameter than width.
   30+ layers at d≤1280 is the regime. Training is still stable with
   RMSNorm + QK-Norm + z-loss + grad clip 1.0.

2. **Tied embeddings + 32k vocab.** At d=576, a 128k vocab (Qwen-style) would
   burn 74M params on embeddings alone — over half the 150m budget. A 32k
   English-focused BPE costs ~18M tied. English-only scope makes this free.

3. **GQA (3–4× query grouping).** Saves KV params and — more importantly —
   KV-cache at inference, which matters for 8k RAG contexts on one machine.

4. **Sliding window (1024) + every 4th layer global.** Chat + RAG needs long
   context but tiny models can't afford full O(n²) everywhere. Windowed
   layers handle locality; global layers integrate retrieved passages.
   KV-cache stays bounded (~¼ of full attention).

5. **SwiGLU, RMSNorm, RoPE.** Standard modern recipe; no reason to deviate.
   RoPE θ=100k supports 4k natively and 8k with mild YaRN scaling for RAG.

6. **QK-Norm.** Cheap insurance for 80B-token runs at high LR (3e-3 at 150m).
   Prevents attention-logit blowup late in training.

7. **No MoE, no adapters-as-params.** The challenge counts parameters; MoE
   inflates total params. Retrieval is our "sparse capacity" instead.

8. **z-loss (1e-4).** Stabilizes the tied head's logits over very long runs.

## What we deliberately did NOT do

- **Mamba/hybrid SSMs**: promising for inference, but the small-model chat
  recipe (SFT/DPO/distill tooling, judge comparability) is Transformer-native.
  Revisit if attention proves the bottleneck — the data/retrieval stack is
  architecture-agnostic.
- **BitNet / 1-bit training**: inference win, but training risk on a fixed
  time budget. Post-training quantization (FP8/FP4 on Blackwell) instead.
- **Tiny vocab (<16k)**: saves params but degrades English fertility and
  code; 32k is the sweet spot.

## Context schedule

- Pretrain/anneal: 4096 packed.
- SFT/RAG: 8192 with YaRN ×2 (same weights, scaled RoPE) — room for
  retrieved passages + episodic summary + multi-turn history.
