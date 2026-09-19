"""Assemble the SFT jsonl: SmolTalk + UltraChat + RAG-tool trajectories.

RAG trajectories are synthesized from the knowledge index itself:
  (question from chunk) -> <search> -> (retrieved context) -> grounded answer.
Questions/answers are drafted with templates + a teacher model if available
(--teacher generates answers; otherwise template answers from the chunk).

Usage:
    python scripts/prepare_sft.py --memory data/memory/knowledge --out data/sft.jsonl \\
        --rag-n 50000 [--teacher <hf-id>]
"""
import argparse
import json
import random
import re

TEMPLATES = [
    "What does the passage say about {topic}?",
    "Summarize the key facts about {topic}.",
    "Explain {topic} briefly, using the given context.",
    "According to the context, what is {topic}?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rag-n", type=int, default=50000)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from mllm.memory import DiskVectorStore
    store = DiskVectorStore(args.memory)
    n = len(store)
    print(f"[sft-prep] index chunks={n}", flush=True)

    teacher = None
    if args.teacher:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        ttok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
        tmod = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype="auto",
                                                   device_map="auto", trust_remote_code=True)
        teacher = pipeline("text-generation", model=tmod, tokenizer=ttok, max_new_tokens=200)

    import sqlite3
    con = sqlite3.connect(f"{args.memory}/texts.db")
    rows = con.execute("SELECT text, meta FROM chunks ORDER BY RANDOM() LIMIT ?",
                       (args.rag_n,)).fetchall()

    with open(args.out, "w") as f:
        # 1) plain SFT rows come from HF at train time (see sft.py); here we emit RAG rows
        for text, meta in rows:
            meta = json.loads(meta or "{}")
            topic = (meta.get("title") or text[:60]).strip() or "this topic"
            q = rng.choice(TEMPLATES).format(topic=topic)
            if teacher:
                prompt = f"Context: {text[:1500]}\n\nQuestion: {q}\nAnswer in 2-3 sentences:"
                ans = teacher(prompt)[0]["generated_text"][len(prompt):].strip()
            else:
                first = re.split(r"(?<=[.!?])\s+", text.strip())
                ans = " ".join(first[:2])[:600]
            rec = {"messages": [
                {"role": "user", "content": q},
                {"role": "assistant",
                 "content": f"<|search|>{topic}<|/search|>"},
                {"role": "user", "content": f"<|context|>\n{text[:1500]}\n<|/context|>"},
                {"role": "assistant", "content": ans},
            ], "kind": "rag"}
            f.write(json.dumps(rec) + "\n")
    print(f"[sft-prep] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
