"""Build the disk knowledge index (Wikipedia + FineWeb-Edu chunks).

Usage: python scripts/build_memory.py --config configs/memory.yaml
"""
import argparse
import yaml


def chunks_of(text: str, size: int, overlap: int):
    text = " ".join(text.split())
    i = 0
    while i < len(text):
        yield text[i:i + size]
        i += size - overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/memory.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    from mllm.data import check_streaming_codecs
    check_streaming_codecs()  # fail fast if zstd/lz4 codecs are missing
    import datasets
    from mllm.memory import DiskVectorStore, HybridRetriever
    try:
        from mllm.memory.retriever import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder(cfg.get("embedder", "BAAI/bge-small-en-v1.5"))
    except Exception as e:
        print(f"[memory] WARNING: {e}; using HashEmbedder fallback", flush=True)
        from mllm.memory import HashEmbedder
        emb = HashEmbedder(cfg.get("dim", 384))

    store = DiskVectorStore(cfg["out"], dim=cfg.get("dim", 384),
                            shard_size=cfg.get("shard_size", 200000))
    retr = HybridRetriever(store, emb)
    size, ov = cfg.get("chunk_chars", 1200), cfg.get("overlap_chars", 200)

    total = 0
    for src in cfg["sources"]:
        print(f"[memory] indexing {src['hf']} ...", flush=True)
        ds = datasets.load_dataset(src["hf"], src.get("name"), split="train", streaming=True)
        buf, metas = [], []
        for row in ds:
            if total >= 0 and len(buf) == 0 and total > 0 and total % 100000 == 0:
                print(f"[memory] indexed={total} store={len(store)}", flush=True)
            t = row.get(src["field"], "")
            if not isinstance(t, str) or len(t) < 200:
                continue
            title = row.get("title", src["hf"])
            for ch in chunks_of(t, size, ov):
                buf.append(ch)
                metas.append({"source": src["hf"], "title": str(title)[:120]})
                if len(buf) >= 512:
                    retr.index(buf, metas)
                    total += len(buf)
                    buf, metas = [], []
            if total >= src.get("limit", 10**18) * 4:
                break
        if buf:
            retr.index(buf, metas)
            total += len(buf)
    print(f"[memory] done. chunks={len(store)} out={cfg['out']}", flush=True)


if __name__ == "__main__":
    main()
