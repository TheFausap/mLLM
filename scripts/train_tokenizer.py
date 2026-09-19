"""Train the 32k English BPE tokenizer.

Usage: python scripts/train_tokenizer.py --config configs/tokenizer.yaml
"""
import argparse
import yaml


def iter_texts(cfg):
    from mllm.data import load_first_available
    for src in cfg["sources"]:
        print(f"[tok] streaming {src['hf']} ...", flush=True)
        configs = [src.get("name")] + list(src.get("fallbacks") or [])
        ds, used = load_first_available(src["hf"], configs, "train",
                                        need_field=src["field"])
        print(f"[tok]   config: {used}", flush=True)
        n = 0
        for row in ds:
            if n >= src.get("n", 100000):
                break
            if src["field"] == "messages":
                for m in row.get("messages", []):
                    yield m.get("content", "")[:4000]
                    n += 1
                    if n >= src.get("n", 100000):
                        break
            else:
                t = row.get(src["field"], "")
                if isinstance(t, str) and len(t) > 100:
                    yield t[:8000]
                    n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tokenizer.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    from mllm.data import check_streaming_codecs
    check_streaming_codecs()  # fail fast if zstd/lz4 codecs are missing
    from mllm.tokenizer import train_bpe_tokenizer
    tok = train_bpe_tokenizer(iter_texts(cfg), vocab_size=cfg.get("vocab_size", 32000),
                              save_dir=cfg.get("out", "tokenizer/"))
    print(f"[tok] saved {cfg.get('out')} vocab={len(tok)}", flush=True)
    # sanity
    ids = tok.encode("Hello! The DGX Spark trains tiny models. <|user|> hi")
    print("[tok] sample ids:", ids[:20])
    print("[tok] roundtrip:", tok.decode(ids)[:120])


if __name__ == "__main__":
    main()
