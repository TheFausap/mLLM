"""DPO / ORPO preference tuning for chat quality (tiny-model friendly).

DPO on UltraFeedback-style pairs (chosen/rejected) + optional ORPO (single
forward pass, no reference model — cheaper on one GPU).

Usage:
    python -m mllm.dpo --base checkpoints/mllm-150m-sft --tok tokenizer/ \\
        --data data/prefs.jsonl --out checkpoints/mllm-150m-dpo --algo orpo
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from .model import mLLM
from .tokenizer import MTokenizer
from .data import format_sft


def seq_logps(model, x, y):
    logits = model(x)["logits"].float()
    lp = F.log_softmax(logits[:, :-1], -1)
    tgt = y[:, 1:]
    mask = (tgt != -100)
    got = lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (got * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--data", required=True, help="jsonl: {prompt, chosen, rejected} messages")
    ap.add_argument("--out", required=True)
    ap.add_argument("--algo", default="orpo", choices=["dpo", "orpo"])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = MTokenizer.load(args.tok)
    model = mLLM.load(args.base, device=device,
                      dtype=torch.bfloat16 if device.type == "cuda" else None)
    model.train()
    ref = None
    if args.algo == "dpo":
        ref = mLLM.load(args.base, device=device,
                        dtype=torch.bfloat16 if device.type == "cuda" else None).eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    print(f"[dpo] pairs={len(rows)} algo={args.algo}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def enc(msgs):
        e = format_sft(msgs, tok, args.max_len)
        return (torch.tensor([e["input_ids"]], device=device),
                torch.tensor([e["labels"]], device=device))

    step = 0
    for ep in range(args.epochs):
        for r in rows:
            xc, yc = enc(r["prompt"] + [{"role": "assistant", "content": r["chosen"]}])
            xr, yr = enc(r["prompt"] + [{"role": "assistant", "content": r["rejected"]}])
            opt.zero_grad(set_to_none=True)
            if args.algo == "dpo":
                with torch.no_grad():
                    ref_c, ref_r = seq_logps(ref, xc, yc), seq_logps(ref, xr, yr)
                pi_c, pi_r = seq_logps(model, xc, yc), seq_logps(model, xr, yr)
                logits = args.beta * ((pi_c - ref_c) - (pi_r - ref_r))
                loss = -F.logsigmoid(logits).mean()
            else:  # ORPO: SFT loss + odds-ratio penalty, no ref model
                logits_c = model(xc)["logits"].float()
                sft = F.cross_entropy(logits_c[:, :-1].reshape(-1, logits_c.shape[-1]),
                                      yc[:, 1:].reshape(-1), ignore_index=-100)
                pi_c = seq_logps(model, xc, yc)
                with torch.no_grad():
                    pass
                pi_r = seq_logps(model, xr, yr)
                odds = torch.sigmoid(pi_c - pi_r).clamp_min(1e-8)
                loss = sft + args.beta * (-torch.log(odds)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"[dpo] ep={ep} step={step} loss={loss.item():.3f}", flush=True)
    model.save(args.out)
    print(f"[dpo] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
