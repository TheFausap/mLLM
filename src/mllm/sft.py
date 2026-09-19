"""SFT (+ RAG-tool-use trajectories) for mLLM.

Data: SmolTalk / UltraChat / OpenHermes + synthetic RAG trajectories.
Loss masking: only assistant spans supervised.

Usage:
    python -m mllm.sft --base checkpoints/mllm-150m-pt --tok tokenizer/ \\
        --out checkpoints/mllm-150m-sft --epochs 2
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .model import mLLM
from .tokenizer import MTokenizer, render_chat
from .data import format_sft


class SFTDataset(Dataset):
    def __init__(self, rows, tok: MTokenizer, max_len: int = 8192):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        msgs = r["messages"]
        enc = format_sft(msgs, self.tok, self.max_len)
        return {"input_ids": enc["input_ids"], "labels": enc["labels"]}


def collate(batch, pad_id: int):
    import torch
    L = max(len(b["input_ids"]) for b in batch)
    x = torch.full((len(batch), L), pad_id, dtype=torch.long)
    y = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        x[i, :n] = torch.tensor(b["input_ids"])
        y[i, :n] = torch.tensor(b["labels"])
    return x, y


def load_sft_rows(spec: str, tok, limit: int = 0) -> list:
    """spec: jsonl path(s) comma-separated, or 'hf:<dataset>' for built-ins."""
    rows = []
    for part in spec.split(","):
        part = part.strip()
        if part.startswith("hf:"):
            rows += load_hf_sft(part[3:], limit)
        else:
            with open(part) as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
                    if limit and len(rows) >= limit:
                        break
    return rows


def _norm_msgs(raw_msgs) -> list:
    """Normalize [{role, content}] — use real roles when present."""
    msgs = []
    for i, m in enumerate(raw_msgs):
        role = m.get("role", "")
        if role not in ("user", "assistant", "system"):
            role = "user" if i % 2 == 0 else "assistant"
        content = m.get("content", "")
        if isinstance(content, str) and content.strip():
            msgs.append({"role": role, "content": content})
    return msgs


def load_hf_sft(name: str, limit: int = 0) -> list:
    from .data import load_first_available
    rows = []
    if name == "smoltalk":
        ds, used = load_first_available(
            "HuggingFaceTB/smoltalk",
            ["everyday-conversations", "smol-magpie-ultra", "long-conversations", "all"],
            "train", need_field="messages")
        print(f"[sft] smoltalk config: {used}", flush=True)
        for r in ds:
            msgs = _norm_msgs(r["messages"])
            if len(msgs) >= 2:
                rows.append({"messages": msgs})
            if limit and len(rows) >= limit:
                break
    elif name == "ultrachat":
        import datasets
        ds = datasets.load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                                   streaming=True)
        for r in ds:
            msgs = _norm_msgs(r["messages"])
            if len(msgs) >= 2:
                rows.append({"messages": msgs})
            if limit and len(rows) >= limit:
                break
    else:
        raise ValueError(f"unknown SFT builtin: {name}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--data", default="hf:smoltalk",
                    help="jsonl paths or hf:smoltalk,hf:ultrachat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = MTokenizer.load(args.tok)
    model = mLLM.load(args.base, device=device,
                      dtype=torch.bfloat16 if device.type == "cuda" else None)
    model.train()
    # extend context for RAG (YaRN-style) — same weights, scaled RoPE
    model.cfg.rope_scaling = "yarn"
    for layer in model.layers:
        layer.attn.rotary = type(layer.attn.rotary)(
            layer.attn.rotary.head_dim, 8192, model.cfg.rope_theta, "yarn", 2.0).to(device)

    rows = load_sft_rows(args.data, tok, args.limit)
    print(f"[sft] rows={len(rows)}", flush=True)
    ds = SFTDataset(rows, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_id))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step = 0
    for ep in range(args.epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
                logits = model(xb)["logits"].float()
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                       yb[:, 1:].reshape(-1), ignore_index=-100)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 20 == 0:
                print(f"[sft] ep={ep} step={step} loss={loss.item():.3f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m.save(args.out)
    print(f"[sft] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
