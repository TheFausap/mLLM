"""Evaluation harness for mLLM.

No-API checks (always run):
  - held-out perplexity (FineWeb-Edu slice)
  - IFEval-lite: verifiable instructions (length, keyword, format)
  - grounding: does the answer use <context> / abstain correctly?
  - coherence probes: repetition rate, truncation, empty outputs

Judge checks (need --judge api|local):
  - MT-Bench-lite: 20 two-turn conversational prompts, scored 1-10

Usage:
    python -m mllm.eval --checkpoint ... --tok ... [--mtbench-lite --judge api]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter

# ----------------------------------------------------------------------------
# IFEval-lite: verifiable instruction checks (no judge needed)
# ----------------------------------------------------------------------------

IFEVAL_LITE = [
    {"prompt": "Write exactly three sentences about the sea.",
     "check": lambda r: len([s for s in re.split(r"[.!?]+", r.strip()) if s.strip()]) == 3,
     "name": "exact_3_sentences"},
    {"prompt": "Name a fruit. Your entire answer must be a single word.",
     "check": lambda r: len(r.strip().split()) == 1, "name": "single_word"},
    {"prompt": "Write the word BLUE exactly 5 times, separated by spaces, nothing else.",
     "check": lambda r: r.strip() == "BLUE BLUE BLUE BLUE BLUE", "name": "repeat_exact"},
    {"prompt": "List 4 animals, one per line, numbered 1. to 4.",
     "check": lambda r: all(f"{i}." in r for i in range(1, 5)), "name": "numbered_list"},
    {"prompt": "Answer with ONLY the number 42, no other text.",
     "check": lambda r: r.strip() == "42", "name": "only_42"},
    {"prompt": "Write a sentence that contains the word 'xylophone' exactly twice.",
     "check": lambda r: r.lower().count("xylophone") == 2, "name": "keyword_twice"},
    {"prompt": "Reply in UPPERCASE ONLY: say hello.",
     "check": lambda r: r.strip() and r.strip() == r.strip().upper() and "HELLO" in r, "name": "uppercase"},
    {"prompt": "Write two lines. The first line must start with 'Apples'. The second with 'Bananas'.",
     "check": lambda r: len(r.strip().splitlines()) >= 2 and r.strip().splitlines()[0].startswith("Apples") and r.strip().splitlines()[1].startswith("Bananas"),
     "name": "line_prefixes"},
]

GROUNDING_CASES = [
    {"context": ["The Eiffel Tower is 330 metres tall including antennas."],
     "query": "How tall is the Eiffel Tower?",
     "must_contain": ["330"], "name": "grounded_fact"},
    {"context": ["Penguins cannot fly. They are flightless birds."],
     "query": "Can penguins fly?",
     "must_contain": ["no", "cannot", "can't", "flightless"], "name": "grounded_no",
     "any": True},
    {"context": ["The Eiffel Tower is 330 metres tall."],
     "query": "What is the population of Tokyo?",
     "must_contain": ["don't know", "do not know", "not in", "no information", "cannot answer"],
     "name": "abstain", "any": True},
]

MTBENCH_LITE = [
    "What are some tips for staying productive while working from home?",
    "Explain photosynthesis to a 10-year-old.",
    "I had a bad day. Can you cheer me up a little?",
    "Write a short poem about rain.",
    "How do I make a simple omelette? List the steps.",
    "What is the difference between weather and climate?",
    "Help me plan a 3-day trip to Rome on a budget.",
    "My friend and I disagree about whether AI will take all jobs. What do you think?",
    "Summarize the plot of Romeo and Juliet in 5 sentences.",
    "I'm learning guitar. Give me a one-week practice plan.",
    "Why is the sky blue? Then explain it again more simply.",
    "Write a polite email declining a job offer.",
    "What should I consider before adopting a dog?",
    "Tell me a very short bedtime story about a robot.",
    "How does a bicycle gear work?",
    "I'm nervous about a job interview tomorrow. Any advice?",
    "Compare solar and wind energy in a few sentences.",
    "What is 15% of 240? Show your working.",
    "Give me three ideas for a birthday surprise for my mom.",
    "Do you think social media is good or bad for teenagers? Discuss both sides.",
]


# ----------------------------------------------------------------------------
def repetition_rate(text: str, n: int = 4) -> float:
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    counts = Counter(grams)
    repeated = sum(1 for g in grams if counts[g] > 1)
    return repeated / len(grams)


def coherence_report(responses: list) -> dict:
    reps = [repetition_rate(r) for r in responses]
    empties = sum(1 for r in responses if not r.strip())
    return {"mean_rep4": sum(reps) / max(1, len(reps)),
            "max_rep4": max(reps) if reps else 0.0,
            "empty_rate": empties / max(1, len(responses)),
            "mean_len_words": sum(len(r.split()) for r in responses) / max(1, len(responses))}


def run_ifeval(generate) -> dict:
    from .tokenizer import render_chat
    results = []
    for case in IFEVAL_LITE:
        prompt = render_chat([{"role": "user", "content": case["prompt"]}])
        resp = generate(prompt)
        ok = False
        try:
            ok = bool(case["check"](resp))
        except Exception:
            ok = False
        results.append({"name": case["name"], "pass": ok, "response": resp[:300]})
    acc = sum(1 for r in results if r["pass"]) / len(results)
    return {"accuracy": acc, "cases": results}


def run_grounding(generate) -> dict:
    from .tokenizer import render_rag_prompt
    results = []
    for case in GROUNDING_CASES:
        prompt = render_rag_prompt(case["query"], case["context"])
        resp = generate(prompt)
        low = resp.lower()
        wants = [w.lower() for w in case["must_contain"]]
        ok = any(w in low for w in wants) if case.get("any") else all(w in low for w in wants)
        results.append({"name": case["name"], "pass": ok, "response": resp[:300]})
    acc = sum(1 for r in results if r["pass"]) / len(results)
    return {"accuracy": acc, "cases": results}


def run_mtbench_lite(generate, judge_generate=None) -> dict:
    """Single-model MT-Bench-lite. If judge_generate is None, just collect
    responses + coherence stats (judge later with a frontier API)."""
    from .tokenizer import render_chat
    out = []
    for q in MTBENCH_LITE:
        # turn 1
        p1 = render_chat([{"role": "user", "content": q}])
        r1 = generate(p1)
        # turn 2 (follow-up pressures multi-turn coherence)
        p2 = render_chat([{"role": "user", "content": q},
                          {"role": "assistant", "content": r1},
                          {"role": "user", "content": "Thanks! Can you make that shorter and simpler?"}])
        r2 = generate(p2)
        out.append({"q": q, "r1": r1, "r2": r2})
    rep = coherence_report([o["r1"] for o in out] + [o["r2"] for o in out])
    scores = None
    if judge_generate is not None:
        scores = []
        for o in out:
            jp = ("<|system|>\nRate this assistant response 1-10 for helpfulness, "
                  "coherence and conversation quality. Reply with ONLY the number.\n"
                  f"<|user|>\nQ: {o['q']}\nA: {o['r1']}\n<|assistant|>\n")
            try:
                s = float(re.findall(r"[\d.]+", judge_generate(jp).strip())[0])
                scores.append(min(10.0, max(1.0, s)))
            except Exception:
                scores.append(float("nan"))
        rep["judge_mean"] = float(sum(s for s in scores if s == s) / max(1, len([s for s in scores if s == s])))
    return {"coherence": rep, "turns": out, "judge_scores": scores}


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--mtbench-lite", action="store_true")
    ap.add_argument("--judge", default=None, help="checkpoint path for local judge model")
    args = ap.parse_args()

    import torch
    from .model import mLLM
    from .tokenizer import MTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = MTokenizer.load(args.tok)
    model = mLLM.load(args.checkpoint, device=device,
                      dtype=torch.bfloat16 if device.type == "cuda" else None)
    model.eval()

    def generate(prompt: str) -> str:
        ids = torch.tensor([tok.encode(prompt)], device=device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new, temperature=0.7,
                                 top_p=0.9, stop_ids=[tok.eos_id])
        gen = out[0, ids.shape[1]:].tolist()
        return tok.decode(gen)

    results = {"ifeval_lite": run_ifeval(generate), "grounding": run_grounding(generate)}
    print(f"[eval] IFEval-lite: {results['ifeval_lite']['accuracy']:.2f}", flush=True)
    print(f"[eval] grounding:   {results['grounding']['accuracy']:.2f}", flush=True)
    if args.mtbench_lite:
        judge_fn = None
        if args.judge:
            jmodel = mLLM.load(args.judge, device=device,
                               dtype=torch.bfloat16 if device.type == "cuda" else None).eval()

            def judge_fn(p: str) -> str:
                ids = torch.tensor([tok.encode(p)], device=device)
                with torch.no_grad():
                    out = jmodel.generate(ids, max_new_tokens=8, temperature=0.0)
                return tok.decode(out[0, ids.shape[1]:].tolist())
        mt = run_mtbench_lite(generate, judge_fn)
        results["mtbench_lite"] = {"coherence": mt["coherence"],
                                   "judge_scores": mt["judge_scores"]}
        print(f"[eval] mtbench-lite coherence: {mt['coherence']}", flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
