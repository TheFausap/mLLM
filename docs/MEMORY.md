# Retrieval-first memory: facts on disk, fluency in weights

The core trick of this project: a 125M model cannot memorize Wikipedia, so we
stop asking it to. Parameters learn *how to talk, reason, follow instructions,
and use tools*; the SSD learns *facts*.

## Two stores

### 1. Knowledge index (shared, read-mostly)

- **Content**: Wikipedia-en (~1.5M articles) + curated FineWeb-Edu passages,
  chunked to ~1200 chars with 200 overlap → ~10–20M chunks.
- **Dense**: 384-dim embeddings (BAAI/bge-small-en-v1.5, 33M params — the
  embedder is *not* counted in the LM budget, same as a tokenizer),
  int8-quantized, numpy memmap shards (200k rows each). ~8GB for 20M chunks.
- **Sparse**: BM25 (`rank_bm25`) with pure-python TF-IDF fallback; SQLite sidecar.
- **Fusion**: reciprocal-rank fusion of dense + sparse (overfetch 20 → top 4–5).
- **Build**: `python scripts/build_memory.py --config configs/memory.yaml`.

### 2. Episodic memory (per-user, read-write)

- Raw turns (SQLite, capped) + a **rolling summary written by the model itself**
  (`EpisodicMemory.update_summary`), refreshed every few turns.
- Injected as `<|memory|>…<|/memory|>` each turn: summary + last ~6 turns.
- This is what makes multi-turn chat feel continuous despite a tiny model.

## The `<search>` loop

1. User message arrives.
2. Probe pass: model emits `<|search|>query<|/search|>` or `NOSEARCH`.
   (Fallback: always retrieve for substantive `?` questions.)
3. Hybrid retrieval returns top-k chunks → `<|context|>`.
4. Final pass generates the grounded answer — or abstains
   ("I don't know…") when the context lacks the answer.

The model is *trained* for this loop: Stage-2 RAG-grounded docs + Stage-3
`<search>` trajectories (`scripts/prepare_sft.py`), and grounding is an
explicit eval axis (`eval.GROUNDING_CASES`).

## Why this shrinks the floor

- Factuality benchmarks stop measuring memorization (params) and start
  measuring *reading comprehension + abstention* — skills that fit in 100–300M.
- Long-tail knowledge, updates ("cutoff"), and personalization become index
  operations, not retraining.
- Honest accounting: we report LM params (<1B) **and** index size separately.
  Storage is ~1000× cheaper than parameters, and the Spark ships 4TB of it.

## Failure modes & mitigations

| risk | mitigation |
|---|---|
| retrieval miss → hallucination | train abstention; eval it; `always_retrieve` mode for factual domains |
| context stuffing drowns the tiny model | top-k=4, 6k-char budget, dedupe; global layers every 4th |
| stale index | rebuild script is idempotent; version the index dir |
| slow brute-force search at 20M rows | shard streaming keeps RAM flat; add IVF/clustering later if needed |
| embedder mismatch (hash fallback) | hash embedder is tests-only; prod uses bge-small |
