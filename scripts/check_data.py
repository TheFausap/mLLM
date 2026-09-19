"""Pre-flight probe for every HF dataset/config/field mLLM references.

Catches wrong config names (e.g. cosmopedia has no 'v2') in ~1-2 minutes
instead of crashing hours into a run.

Usage: python scripts/check_data.py [--sft] [--memory]
  --sft     also probe SFT builtins (smoltalk, ultrachat)
  --memory  also probe memory.yaml sources (wikipedia is slow to resolve)
"""
import argparse
import sys

sys.path.insert(0, "src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", action="store_true")
    ap.add_argument("--memory", action="store_true")
    args = ap.parse_args()

    import yaml
    from mllm.data import (MixItem, probe_source, source_name,
                           check_streaming_codecs, PRETRAIN_MIX, ANNEAL_MIX)

    check_streaming_codecs()
    failures = []

    def check(m: MixItem, label: str):
        r = probe_source(m)
        name = source_name(m)
        if r["ok"]:
            print(f"[ok]   {label:12s} {name} ({r['chars']} chars, {r['dt']}s)",
                  flush=True)
        else:
            print(f"[FAIL] {label:12s} {name}: {r['error']}", flush=True)
            failures.append((label, name, r["error"]))

    print("== pretrain mix ==", flush=True)
    for m in PRETRAIN_MIX:
        check(m, "pretrain")
    print("== anneal mix ==", flush=True)
    for m in ANNEAL_MIX:
        check(m, "anneal")

    print("== tokenizer sources ==", flush=True)
    tcfg = yaml.safe_load(open("configs/tokenizer.yaml"))
    for src in tcfg["sources"]:
        if src["field"] == "messages":
            from mllm.data import load_first_available
            configs = [src.get("name")] + list(src.get("fallbacks") or [])
            try:
                ds, used = load_first_available(src["hf"], configs, "train",
                                                need_field="messages")
                row = next(iter(ds))
                n = len(row["messages"])
                print(f"[ok]   tokenizer    {src['hf']}:{used}[messages] "
                      f"(sample convo: {n} turns)", flush=True)
            except Exception as e:
                print(f"[FAIL] tokenizer    {src['hf']}: {str(e)[:250]}", flush=True)
                failures.append(("tokenizer", src["hf"], str(e)[:250]))
        else:
            check(MixItem(src["hf"], src.get("name"), src["field"]), "tokenizer")

    if args.sft:
        print("== sft builtins ==", flush=True)
        from mllm.sft import load_hf_sft
        for name in ("smoltalk", "ultrachat"):
            try:
                rows = load_hf_sft(name, limit=2)
                print(f"[ok]   sft          hf:{name} ({len(rows)} rows, "
                      f"{len(rows[0]['messages'])} msgs/row)", flush=True)
            except Exception as e:
                print(f"[FAIL] sft          hf:{name}: {str(e)[:250]}", flush=True)
                failures.append(("sft", name, str(e)[:250]))

    if args.memory:
        print("== memory sources ==", flush=True)
        mcfg = yaml.safe_load(open("configs/memory.yaml"))
        for src in mcfg["sources"]:
            check(MixItem(src["hf"], src.get("name"), src.get("field", "text")),
                  "memory")

    print(flush=True)
    if failures:
        print(f"{len(failures)} FAILURES:", flush=True)
        for label, name, err in failures:
            print(f"  - [{label}] {name}: {err}", flush=True)
        sys.exit(1)
    print("All sources OK.", flush=True)


if __name__ == "__main__":
    main()
