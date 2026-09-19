# mLLM — the smallest LLM that can actually chat

**Goal:** find the parameter floor for a *conversational* English LLM — coherent multi-turn chat + reasonable instruction-following — **well under 1B params**, trained on a **single NVIDIA DGX Spark**, using every legitimate trick available: deep-thin architectures, distillation, high-quality data curricula, and **disk-backed external memory (RAG)** so facts live on SSD instead of in weights.

Not TinyStories. This model has to *engage with the user*: hold a conversation, follow instructions, admit what it doesn't know, and look things up.

## The bet

A tiny dense LM (100–600M) **cannot** memorize the world — and shouldn't try. So:

1. **Weights learn** conversation, reasoning patterns, instruction-following, tool use, and style.
2. **Disk learns** facts: a multi-GB hybrid (dense + BM25) knowledge index + per-user episodic memory on the Spark's 4TB SSD.
3. **Training teaches grounding**: retrieval-augmented pretraining, search-query generation (`<search>…</search>`), and distillation from a large teacher.

## Model ladder (find the floor empirically)

| Model | Layers | Hidden | Heads (Q/KV) | FFN | Vocab | Params ≈ | Target |
|---|---|---|---|---|---|---|---|
| `mllm-150m` | 30 | 576 | 9 / 3 | 1536 | 32k, tied | ~125M | floor probe — can it chat at all? |
| `mllm-350m` | 32 | 960 | 12 / 4 | 2560 | 32k, tied | ~345M | hopeful sweet spot |
| `mllm-600m` | 32 | 1248 | 16 / 4 | 3328 | 32k, tied | ~590M | quality ceiling under 1B |

All: decoder-only Transformer, RMSNorm, SwiGLU, RoPE (θ=100k, 4k→8k), GQA, QK-Norm, sliding-window (1024, every 4th layer global), tied embeddings, no biases. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Param counts **exclude** the disk index (that's the point — storage is ~1000× cheaper than parameters).

## Training recipe (single DGX Spark, ~weeks)

```
Stage 0  tokenizer      32k English BPE                    (hours)
Stage 1  pretrain       ~80B tokens, FineWeb-Edu/DCLM/Cosmopedia mix
Stage 2  anneal         ~8B tokens, edu + instruction + RAG-grounded docs
Stage 3  SFT            ~3M convs, teacher-distilled + search-tool trajectories
Stage 4  preference     DPO/ORPO + on-policy distillation (GKD)
```

Full data mixes, LR schedules, batch sizes, and DGX-Spark throughput notes: [`docs/RECIPE.md`](docs/RECIPE.md), [`docs/DGX_SPARK.md`](docs/DGX_SPARK.md).

## Retrieval-first memory

- **Knowledge index**: Wikipedia + curated web-edu chunks, int8-quantized dense vectors (memmapped numpy) + BM25 sidecar in SQLite. 20M chunks ≈ 8GB on disk.
- **Episodic memory**: per-user turns + rolling summaries, retrieved every turn for continuity.
- **Tool use**: the model emits `<search>query</search>` when it needs facts; the runtime injects `<context>…</context>` and the model answers grounded — or says it doesn't know.

See [`docs/MEMORY.md`](docs/MEMORY.md).

## Repo layout

```
configs/        model + stage configs (150m / 350m / 600m)
src/mllm/       model, tokenizer, data, train, sft, dpo, distill, eval
src/mllm/memory/  disk vector store, hybrid retriever, RAG pipeline
scripts/        tokenizer prep, data prep, memory build, launch scripts
docs/           architecture, recipe, memory, DGX Spark guide
tests/          smoke tests (memory/data run without torch)
```

## Quickstart

```bash
# 1. install (DGX Spark: CUDA wheel; CPU-only also works for tests)
pip install -e ".[train]"        # full training stack
pip install -e .                 # minimal (serve + memory, numpy only)

# 2. run tests
python -m pytest tests/ -x -q

# 3. train the 32k tokenizer
python scripts/train_tokenizer.py --config configs/tokenizer.yaml

# 4. launch pretraining (single DGX Spark, checkpointing + small microbatches)
bash scripts/train.sh configs/150m.yaml pretrain

# 5. build the knowledge index (Wikipedia)
python scripts/build_memory.py --config configs/memory.yaml

# 6. chat with RAG + episodic memory
python -m mllm.serve --checkpoint checkpoints/mllm-150m-sft --memory data/memory --cli
```

## Evaluation

```bash
bash scripts/eval.sh checkpoints/mllm-150m-sft   # ppl + IFEval-lite + grounding + coherence probes
python -m mllm.eval --checkpoint ... --mtbench-lite  # needs a judge (API or local 8B)
```

Success = coherent multi-turn chat judged blind against 1B baselines (Llama-3.2-1B, Qwen2.5-0.5B, SmolLM2), at a fraction of the parameters — with retrieval carrying factuality.

## Status

Scaffolding + working code; first runs target `mllm-150m` to probe the floor, then scale only as needed. See `docs/RECIPE.md` for the run log convention.
