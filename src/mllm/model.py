"""mLLM decoder-only Transformer (Llama-style, deep-thin, GQA+SWA, tied embeddings).

Requires torch (training/inference). Config lives in config.py (torch-free).
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


# ----------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 100000.0,
                 scaling: Optional[str] = None, factor: float = 2.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        if scaling == "yarn":  # simple YaRN-style: scale frequencies for 4k->8k
            inv_freq = inv_freq / factor
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_cos = None
        self._cached_sin = None
        self._cached_len = 0

    def _build(self, seq_len: int, device, dtype):
        if self._cached_len >= seq_len and self._cached_cos is not None \
                and self._cached_cos.device == device:
            return self._cached_cos[:seq_len], self._cached_sin[:seq_len]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        self._cached_cos, self._cached_sin, self._cached_len = cos, sin, seq_len
        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor, start: int = 0):
        # q,k: (B, H, T, D)
        T = q.shape[2]
        cos, sin = self._build(start + T, q.device, q.dtype)
        cos = cos[start:start + T].unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
        sin = sin[start:start + T].unsqueeze(0).unsqueeze(0)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        # GPT-NeoX style rotation on halves
        cos1, cos2 = cos.chunk(2, dim=-1)
        sin1, sin2 = sin.chunk(2, dim=-1)
        q_rot = torch.cat([q1 * cos1 - q2 * sin1, q2 * cos2 + q1 * sin2], dim=-1)
        k_rot = torch.cat([k1 * cos1 - k2 * sin1, k2 * cos2 + k1 * sin2], dim=-1)
        return q_rot, k_rot


# ----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        d, n_h, n_kv = cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads
        self.head_dim = d // n_h
        self.n_rep = n_h // n_kv
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, self.head_dim * n_kv, bias=False)
        self.v_proj = nn.Linear(d, self.head_dim * n_kv, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps) if cfg.use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps) if cfg.use_qk_norm else nn.Identity()
        self.rotary = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta,
                                      cfg.rope_scaling, cfg.rope_factor)
        self.dropout = cfg.dropout
        # sliding window except every Nth layer (global)
        if cfg.sliding_window and (layer_idx % cfg.global_attention_every != 0):
            self.window = cfg.sliding_window
        else:
            self.window = None

    def forward(self, x, cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        B, T, _ = x.shape
        n_h, n_kv, hd = self.cfg.num_heads, self.cfg.num_kv_heads, self.head_dim
        q = self.q_proj(x).view(B, T, n_h, hd).transpose(1, 2)       # (B,H,T,D)
        k = self.k_proj(x).view(B, T, n_kv, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, n_kv, hd).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        start = cache[0].shape[2] if cache is not None else 0
        q, k = self.rotary(q, k, start=start)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v) if use_cache else None
        # repeat KV heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        # Sliding window: during incremental decode (T==1) we can trim the KV
        # cache; during training/prefill (T>1) trimming would break the causal
        # alignment of is_causal, so use an explicit band mask instead.
        attn_mask = None
        is_causal = True
        if self.window is not None and k.shape[2] > self.window:
            if T == 1 and cache is not None:
                k = k[:, :, -self.window:, :]
                v = v[:, :, -self.window:, :]
                if new_cache is not None:
                    new_cache = (new_cache[0][:, :, -self.window:, :],
                                 new_cache[1][:, :, -self.window:, :])
            else:
                L, S = T, k.shape[2]
                qi = torch.arange(L, device=x.device)[:, None]
                kj = torch.arange(S, device=x.device)[None, :]
                # keys are left-aligned with queries here (no cache in this path)
                attn_mask = (kj <= qi) & (qi - kj < self.window)
                is_causal = False
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_cache


# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.hidden_size, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cache=None, use_cache=False):
        a, new_cache = self.attn(self.attn_norm(x), cache, use_cache)
        x = x + a
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_cache


# ----------------------------------------------------------------------------
class mLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if cfg.tie_embeddings:
            self.lm_head = None  # use tok_emb.T
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, labels=None, cache=None, use_cache=False,
                z_loss: float = 0.0):
        # input_ids: (B,T)
        x = self.tok_emb(input_ids)
        new_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            c = cache[i] if cache is not None else None
            x, nc = layer(x, c, use_cache)
            if use_cache:
                new_caches.append(nc)
        h = self.final_norm(x)
        logits = h @ self.tok_emb.weight.T if self.lm_head is None else self.lm_head(h)
        out = {"logits": logits}
        if use_cache:
            out["cache"] = new_caches
        if labels is not None:
            # shift for next-token prediction
            shift = logits[:, :-1].contiguous().float()
            tgt = labels[:, 1:].contiguous() if labels.shape == input_ids.shape else labels
            # labels passed as input_ids -> shift inside; or pre-shifted (B,T-1)
            if tgt.shape[1] == input_ids.shape[1]:
                tgt = tgt[:, 1:].contiguous()
            loss = F.cross_entropy(shift.view(-1, shift.shape[-1]), tgt.view(-1),
                                   ignore_index=-100)
            if z_loss > 0:
                lse = torch.logsumexp(shift, dim=-1)
                loss = loss + z_loss * (lse ** 2).mean()
            out["loss"] = loss
        return out

    # -- generation ---------------------------------------------------------
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9,
                 stop_ids: Optional[List[int]] = None,
                 repetition_penalty: float = 1.0) -> torch.Tensor:
        self.eval()
        ids = input_ids
        cache = None
        for _ in range(max_new_tokens):
            out = self(ids[:, -self.cfg.max_seq_len:] if cache is None else ids[:, -1:],
                       cache=cache, use_cache=True)
            cache = out["cache"]
            logits = out["logits"][:, -1, :].float()
            if repetition_penalty != 1.0:
                for b in range(ids.shape[0]):
                    for tok in ids[b].tolist():
                        if logits[b, tok] > 0:
                            logits[b, tok] /= repetition_penalty
                        else:
                            logits[b, tok] *= repetition_penalty
            if temperature and temperature > 0:
                logits = logits / temperature
                # nucleus
                if top_p < 1.0:
                    sp, si = torch.sort(logits, descending=True)
                    cum = torch.cumsum(F.softmax(sp, -1), -1)
                    cutoff = (cum > top_p).float().argmax(-1)
                    for b in range(ids.shape[0]):
                        sp[b, cutoff[b] + 1:] = float("-inf")
                    logits = torch.full_like(logits, float("-inf")).scatter(1, si, sp)
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if stop_ids and nxt.item() in stop_ids:
                break
        return ids

    # -- persistence --------------------------------------------------------
    def save(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        import json
        with open(f"{path}/config.json", "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)
        try:
            from safetensors.torch import save_file
            save_file(self.state_dict(), f"{path}/model.safetensors")
        except ImportError:
            torch.save(self.state_dict(), f"{path}/model.pt")

    @classmethod
    def load(cls, path: str, device="cpu", dtype=None):
        import json, os
        with open(f"{path}/config.json") as f:
            cfg = ModelConfig(**{k: v for k, v in json.load(f).items()
                                 if k in ModelConfig.__dataclass_fields__})
        model = cls(cfg)
        try:
            from safetensors.torch import load_file
            sd = load_file(f"{path}/model.safetensors", device=device)
        except (ImportError, FileNotFoundError, OSError):
            sd = torch.load(f"{path}/model.pt", map_location=device, weights_only=True)
        model.load_state_dict(sd)
        if dtype is not None:
            model = model.to(dtype)
        return model.to(device)


def count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = sum(p.numel() for n, p in model.named_parameters() if "tok_emb" in n)
    return {"total": total, "trainable": trainable, "embeddings": emb,
            "non_embedding": total - emb}
