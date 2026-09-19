# Training recipe (DGX Spark × 1, ~weeks)

Target budget: **~80–100B pretrain + ~8B anneal + SFT + DPO/GKD**, starting
with `mllm-150m` to probe the floor, then 350m/600m only as needed.

## Stage 0 — Tokenizer (hours)

`python scripts/train_tokenizer.py` → 32k English BPE over FineWeb-Edu (400k
docs) + Cosmopedia-v2 (150k) + SmolTalk (50k convos). Byte-level BPE with the
`SPECIAL_TOKENS` control plane (`<|user|>`, `<|search|>`, `<|context|>`, …).

## Stage 1 — Pretrain (~80B tokens, ~3 weeks at 150m)

Mix (`data.PRETRAIN_MIX`):

| source | weight | why |
|---|---|---|
| FineWeb-Edu | 0.50 | best open high-signal web for small LMs |
| DCLM-baseline | 0.20 | diversity / robustness |
| Cosmopedia-v2 | 0.15 | synthetic textbooks — the Phi lesson: small models learn best from clean, didactic text |
| SmolLM-Corpus (cosmopedia) | 0.05 | stories + textbooks |
| Open-Web-Math | 0.05 | reasoning traces |
| StarCoderData | 0.05 | light code dose (structure, not mastery) |

Hyperparams (150m): batch ~1M tokens, WSD peak LR 3e-3 → 3e-4, warmup 2k,
AdamW (0.9, 0.95), wd 0.1, clip 1.0, z-loss 1e-4, bf16, torch.compile,
flash SDPA. Pack to 4096 with `<eos>` separators. Larger sizes: lower peak LR
(2.2e-3 / 1.8e-3), same batch.

Throughput math: 6·N·D FLOPs. 150M × 80B = 7.2e19 FLOP ≈ 21 days at ~40 TFLOP/s
sustained. See `DGX_SPARK.md` for tuning to hit that.

## Stage 2 — Anneal (~8B tokens, ~2 days)

Decay LR (cosine to 6e-5) on the edu/conversational-heavy `ANNEAL_MIX`:
Cosmopedia 0.30, FineWeb-Edu 0.30, SmolTalk-pretrain 0.20, math 0.10, code 0.10.
This is where "textbook + chat" style gets baked in. Include RAG-grounded
docs (passage + question + grounded answer) so grounding isn't SFT-only.

## Stage 3 — SFT (~3M conversations, 1–2 days)

Sources: SmolTalk (460k+, highest priority — matches the SmolLM chat recipe),
UltraChat-200k (multi-turn depth), OpenHermes-2.5 (instruction diversity),
plus **50k RAG-tool trajectories** from `scripts/prepare_sft.py`
(question → `<search>` → context → grounded answer) and teacher-distilled
answers (generate with Llama-3.1-8B/70B-Instruct, train as plain SFT =
sequence-level distillation).

LR 2e-5, 2 epochs, 8k context with YaRN. Mask non-assistant spans.
Context extension is free (RoPE scaling, no new params).

## Stage 4 — Preference + on-policy distillation (1–2 days)

- **ORPO** (preferred on one GPU: no reference model) or DPO (β=0.1) on
  UltraFeedback-binarized + on-policy pairs judged by the teacher.
- **GKD**: sample from the student, score/correct with the teacher, distill.
  Critical for tiny models — closes 30–50% of the gap to the teacher on chat.

LR 5e-6, 1 epoch. Watch for length hacking; keep a brevity slice in eval.

## Stage 5 — Deploy

- FP8 (or NVFP4 on Blackwell) post-training quantization for serving.
- Ship with the knowledge index + episodic memory (`MEMORY.md`).

## Run log convention

Each run appends to `runs.md`: config hash, tokens, wall-clock, tok/s,
eval ppl / IFEval-lite / grounding / MT-Bench-lite, and blind-chat notes.
Promote 150m → 350m only if 150m + RAG fails the "would I keep chatting?"
bar after Stage 4.

## Data hygiene

- English-first; drop non-English via FastText filter in a preprocessing pass
  if needed (FineWeb-Edu is already English-heavy).
- Deduplicate SFT against eval prompts (exact + 13-gram).
- Keep a fixed 5k-doc FineWeb-Edu held-out slice for ppl comparability.
