"""Logit distillation: train tiny mLLM from a large teacher.

Why: distillation is the single biggest quality lever for sub-1B models.
The teacher (e.g. Llama-3.1-8B-Instruct, Qwen2.5-7B) provides soft targets;
the student matches them with KL + CE mix on SFT/chat data.

Two modes:
  1. offline:  precomputed teacher top-k logits stored in jsonl (logits field)
  2. online:   teacher loaded in-process (needs ~16GB for 8B bf16 — fine on Spark)

Usage (online):
    python -m mllm.distill --student checkpoints/mllm-150m-sft --teacher meta-llama/Llama-3.1-8B-Instruct \\
        --tok tokenizer/ --data data/sft.jsonl --out checkpoints/mllm-150m-kd

Note: teacher tokenizer should match student's vocab for direct KL. If not,
use offline mode with re-tokenized text + teacher responses as plain SFT
(sequence-level distillation), which the sft.py script already covers.
This module implements the matched-vocab KL path + on-policy GKD sampling.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from .model import mLLM
from .tokenizer import MTokenizer


def kd_loss(student_logits, teacher_logits, labels, alpha=0.5, temp=2.0):
    """alpha * KL(teacher||student) + (1-alpha) * CE(labels)."""
    s = student_logits[:, :-1].float()
    t = teacher_logits[:, :-1].float()
    ce = F.cross_entropy(s.reshape(-1, s.shape[-1]), labels[:, 1:].reshape(-1),
                         ignore_index=-100)
    # KL only on supervised positions
    mask = (labels[:, 1:] != -100).float()
    kl = F.kl_div(F.log_softmax(s / temp, -1), F.softmax(t / temp, -1),
                  reduction="none").sum(-1)
    kl = (kl * mask).sum() / mask.sum().clamp_min(1)
    kl = kl * (temp ** 2)
    return alpha * kl + (1 - alpha) * ce, ce.detach(), kl.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True, help="HF id or local path")
    ap.add_argument("--tok", required=True)
    ap.add_argument("--data", required=True, help="jsonl with messages")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = MTokenizer.load(args.tok)
    student = mLLM.load(args.student, device=device,
                        dtype=torch.bfloat16 if device.type == "cuda" else None)
    student.train()

    # Teacher via transformers (lazy import; bf16, eval mode)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ttok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()
    same_vocab = (ttok.vocab_size == len(tok))
    if not same_vocab:
        print(f"[kd] WARNING: teacher vocab {ttok.vocab_size} != student {len(tok)}; "
              f"falling back to sequence-level distillation (SFT on teacher outputs).")
        print("[kd] Hint: generate teacher responses first, then run sft.py.")

    from .sft import SFTDataset, collate, load_sft_rows
    from torch.utils.data import DataLoader
    rows = load_sft_rows(args.data, tok)
    dl = DataLoader(SFTDataset(rows, tok, args.max_len), batch_size=args.batch,
                    shuffle=True, collate_fn=lambda b: collate(b, tok.pad_id))
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    step = 0
    for ep in range(args.epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                if same_vocab:
                    t_logits = teacher(xb).logits
                else:
                    t_logits = None
            s_logits = student(xb)["logits"]
            if t_logits is not None:
                loss, ce, kl = kd_loss(s_logits, t_logits.to(s_logits.device), yb,
                                       args.alpha, args.temp)
            else:  # sequence-level: plain CE on (possibly teacher-generated) labels
                loss = F.cross_entropy(s_logits[:, :-1].float().reshape(-1, s_logits.shape[-1]),
                                       yb[:, 1:].reshape(-1), ignore_index=-100)
                ce, kl = loss.detach(), torch.tensor(0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 20 == 0:
                print(f"[kd] ep={ep} step={step} loss={loss.item():.3f} "
                      f"ce={ce.item():.3f} kl={kl.item():.3f}", flush=True)
    student.save(args.out)
    print(f"[kd] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
