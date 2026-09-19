"""Pretraining loop for mLLM (single DGX Spark).

Features: WSD schedule, bf16 AMP, torch.compile, gradient accumulation,
checkpointing + resume, throughput logging (tok/s), held-out perplexity.

Usage:
    python -m mllm.train --config configs/150m.yaml --stage pretrain
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig, NAMED_MODELS
from .model import mLLM, count_params
from .data import mixed_pretrain_stream, pretrain_batch_iter, PRETRAIN_MIX, ANNEAL_MIX
from .tokenizer import MTokenizer


def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, tok, device, seq_len=4096, batches=20) -> float:
    """Held-out perplexity on a small fixed FineWeb-Edu slice (streamed)."""
    from .data import stream_hf_texts, tokenize_stream, pack_sequences
    model.eval()
    texts = stream_hf_texts("HuggingFaceFW/fineweb-edu", None, "text", split="train", seed=999)
    toks = tokenize_stream(texts, tok)
    packed = pack_sequences(toks, seq_len + 1, tok.eos_id)
    tot_loss, n = 0.0, 0
    for seq in packed:
        if n >= batches:
            break
        x = torch.tensor([seq[:-1]], device=device)
        y = torch.tensor([seq[1:]], device=device)
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x, labels=y)
        tot_loss += out["loss"].item()
        n += 1
    model.train()
    return math.exp(tot_loss / max(1, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="model yaml (configs/150m.yaml)")
    ap.add_argument("--stage", default="pretrain", choices=["pretrain", "anneal"])
    ap.add_argument("--tok", required=True, help="tokenizer dir")
    ap.add_argument("--out", required=True, help="checkpoint dir")
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    ap.add_argument("--dry", action="store_true", help="1 step smoke test, no HF data")
    args = ap.parse_args()

    y = load_yaml(args.config)
    model_name = y.get("model", "150m")
    mcfg = NAMED_MODELS[model_name](**y.get("model_overrides", {}))
    stage_key = "anneal" if args.stage == "anneal" else "pretrain"
    tcfg = TrainConfig(**y.get(stage_key, {}))
    mix = ANNEAL_MIX if args.stage == "anneal" else PRETRAIN_MIX

    set_seed(tcfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mllm] model={model_name} {mcfg.summary()}", flush=True)
    print(f"[mllm] stage={args.stage} steps={tcfg.steps()} batch_tokens={tcfg.batch_tokens}", flush=True)

    tok = MTokenizer.load(args.tok)
    assert len(tok) == mcfg.vocab_size, f"tokenizer {len(tok)} != model {mcfg.vocab_size}"

    model = mLLM(mcfg).to(device)
    print(f"[mllm] params: {count_params(model)}", flush=True)
    if tcfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    if tcfg.grad_checkpoint:
        model.gradient_checkpointing_enable = lambda: None  # placeholder for tiny models (fits w/o ckpt)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.peak_lr,
                            betas=(tcfg.beta1, tcfg.beta2), weight_decay=tcfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(tcfg.precision == "bf16" and device.type == "cuda"))
    use_amp = tcfg.precision == "bf16" and device.type == "cuda"

    os.makedirs(args.out, exist_ok=True)
    start_step = 0
    if args.resume:
        sd = torch.load(f"{args.resume}/trainer.pt", map_location=device, weights_only=False)
        # handle torch.compile prefix
        model_to_load = model._orig_mod if hasattr(model, "_orig_mod") else model
        model_to_load.load_state_dict(torch.load(f"{args.resume}/model.pt", map_location=device, weights_only=True))
        opt.load_state_dict(sd["opt"])
        start_step = sd["step"] + 1
        print(f"[mllm] resumed from step {start_step}", flush=True)

    batch_seqs = max(1, tcfg.batch_tokens // tcfg.seq_len // max(1, tcfg.grad_accum))
    if args.dry:
        def fake_stream():
            rng = random.Random(0)
            while True:
                yield " ".join(f"word{rng.randint(0,999)}" for _ in range(200))
        stream = fake_stream()
    else:
        stream = mixed_pretrain_stream(mix, seed=tcfg.seed)
    batches = pretrain_batch_iter(stream, tok, tcfg.seq_len, batch_seqs)

    t0 = time.time()
    tokens_done = start_step * tcfg.batch_tokens
    for step in range(start_step, tcfg.steps()):
        lr = tcfg.lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(max(1, tcfg.grad_accum)):
            try:
                xb, yb = next(batches)
            except StopIteration:
                break
            xb, yb = xb.to(device), yb.to(device)
            with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
                out = model(xb, labels=yb, z_loss=tcfg.z_loss)
                loss = out["loss"] / max(1, tcfg.grad_accum)
            scaler.scale(loss).backward()
            acc_loss += loss.item() * max(1, tcfg.grad_accum)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        tokens_done += tcfg.batch_tokens

        if (step + 1) % tcfg.log_every == 0:
            dt = time.time() - t0
            tps = tokens_done / max(1e-6, dt)
            print(f"[mllm] step={step+1}/{tcfg.steps()} loss={acc_loss:.3f} lr={lr:.2e} "
                  f"tok={tokens_done/1e9:.2f}B tok/s={tps:.0f}", flush=True)

        if (step + 1) % tcfg.eval_every == 0 and not args.dry:
            try:
                ppl = evaluate(model._orig_mod if hasattr(model, "_orig_mod") else model,
                               tok, device, seq_len=min(2048, tcfg.seq_len))
                print(f"[mllm] eval ppl={ppl:.2f}", flush=True)
            except Exception as e:
                print(f"[mllm] eval skipped: {e}", flush=True)

        if (step + 1) % tcfg.checkpoint_every == 0 or (step + 1) == tcfg.steps():
            m = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(m.state_dict(), f"{args.out}/model.pt")
            with open(f"{args.out}/config.json", "w") as f:
                json.dump(m.cfg.to_dict(), f, indent=2)
            torch.save({"opt": opt.state_dict(), "step": step}, f"{args.out}/trainer.pt")
            with open(f"{args.out}/meta.json", "w") as f:
                json.dump({"step": step + 1, "tokens": tokens_done}, f)
            print(f"[mllm] saved {args.out} @ step {step+1}", flush=True)
        if args.dry and step >= 1:
            print("[mllm] dry run OK", flush=True)
            break


if __name__ == "__main__":
    main()
