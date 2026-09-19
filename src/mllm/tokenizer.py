"""Tokenizer: 32k English BPE (HuggingFace `tokenizers` when available).

Falls back to a deterministic whitespace/byte tokenizer for offline tests.
Special tokens double as the chat + RAG control plane.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Dict, Optional

SPECIAL_TOKENS = [
    "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
    "<|user|>", "<|assistant|>", "<|system|>",
    "<|context|>", "<|/context|>",
    "<|search|>", "<|/search|>",
    "<|memory|>", "<|/memory|>",
]


class MTokenizer:
    def __init__(self, backend, vocab_size: int, stoi: Dict[str, int], itos: List[str]):
        self.backend = backend  # huggingface Tokenizer or None
        self.vocab_size = vocab_size
        self.stoi = stoi
        self.itos = itos
        for t in SPECIAL_TOKENS:
            assert t in stoi, f"missing special token {t}"
        self.pad_id = stoi["<|pad|>"]
        self.bos_id = stoi["<|bos|>"]
        self.eos_id = stoi["<|eos|>"]
        self.unk_id = stoi["<|unk|>"]

    # -- encode/decode ------------------------------------------------------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if self.backend is not None:
            ids = self.backend.encode(text).ids
        else:
            ids = _fallback_encode(text, self.stoi)
        out = []
        if add_bos:
            out.append(self.bos_id)
        out += ids
        if add_eos:
            out.append(self.eos_id)
        return out

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if self.backend is not None:
            return self.backend.decode([i for i in ids if i < self.vocab_size],
                                       skip_special_tokens=skip_special)
        toks = []
        for i in ids:
            t = self.itos[i] if 0 <= i < len(self.itos) else "<|unk|>"
            if skip_special and t in SPECIAL_TOKENS:
                continue
            toks.append(t)
        return _fallback_decode(toks)

    def __len__(self):
        return self.vocab_size

    # -- persistence ---------------------------------------------------------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        if self.backend is not None:
            self.backend.save(f"{path}/tokenizer.json")
        with open(f"{path}/vocab.json", "w") as f:
            json.dump({"vocab_size": self.vocab_size, "itos": self.itos}, f)

    @classmethod
    def load(cls, path: str) -> "MTokenizer":
        with open(f"{path}/vocab.json") as f:
            meta = json.load(f)
        itos, vocab_size = meta["itos"], meta["vocab_size"]
        stoi = {t: i for i, t in enumerate(itos)}
        backend = None
        tj = f"{path}/tokenizer.json"
        if os.path.exists(tj):
            try:
                from tokenizers import Tokenizer as HFTokenizer
                backend = HFTokenizer.from_file(tj)
            except ImportError:
                backend = None
        return cls(backend, vocab_size, stoi, itos)


# ----------------------------------------------------------------------------
def train_bpe_tokenizer(texts, vocab_size: int = 32000, save_dir: Optional[str] = None) -> MTokenizer:
    """Train a byte-level BPE on an iterable of texts."""
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer

    tok = HFTokenizer(BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                         show_progress=True)
    tok.train_from_iterator(texts, trainer=trainer)
    raw = tok.get_vocab()
    itos = [None] * vocab_size
    for piece, idx in raw.items():
        if idx < vocab_size:
            itos[idx] = piece
    for i in range(vocab_size):
        if itos[i] is None:
            itos[i] = f"<|extra_{i}|>"
    stoi = {t: i for i, t in enumerate(itos)}
    mtok = MTokenizer(tok, vocab_size, stoi, itos)
    if save_dir:
        mtok.save(save_dir)
    return mtok


def build_fallback_tokenizer(vocab_size: int = 32000) -> MTokenizer:
    """Deterministic offline tokenizer for tests (no training data needed)."""
    itos = list(SPECIAL_TOKENS)
    # byte-level base (256) then hashed word pieces
    for i in range(256):
        itos.append(f"<|b{i}|>")
    i = len(itos)
    n = 0
    while len(itos) < vocab_size:
        itos.append(f"w{n}")
        n += 1
    stoi = {t: i for i, t in enumerate(itos)}
    return MTokenizer(None, vocab_size, stoi, itos)


_WORD_RE = re.compile(r"\S+|\s+")


def _fallback_encode(text: str, stoi: Dict[str, int]) -> List[int]:
    ids = []
    for m in _WORD_RE.finditer(text):
        piece = m.group(0)
        if piece in stoi:
            ids.append(stoi[piece])
        elif piece.strip() == "":
            ids.append(stoi.get(" ", stoi["<|unk|>"]))
        else:
            # hash word pieces deterministically into w* space
            h = abs(hash(piece)) % (len(stoi) - len(SPECIAL_TOKENS) - 256)
            ids.append(len(SPECIAL_TOKENS) + 256 + h)
    return ids


def _fallback_decode(toks: List[str]) -> str:
    out = []
    for t in toks:
        if t.startswith("<|b") and t.endswith("|>"):
            continue
        if t.startswith("w") and t[1:].isdigit():
            out.append(f"[{t}]")
        else:
            out.append(t)
    return "".join(out)


# ----------------------------------------------------------------------------
# Chat template (shared by SFT, DPO, RAG, serve)
# ----------------------------------------------------------------------------

def render_chat(messages: List[Dict[str, str]], add_assistant_header: bool = True) -> str:
    """messages: [{role: system|user|assistant, content: str}]."""
    parts = []
    for m in messages:
        role = m["role"]
        tag = {"system": "<|system|>", "user": "<|user|>",
               "assistant": "<|assistant|>"}[role]
        parts.append(f"{tag}\n{m['content'].strip()}\n")
    if add_assistant_header and (not messages or messages[-1]["role"] != "assistant"):
        parts.append("<|assistant|>\n")
    return "".join(parts)


def render_rag_prompt(query: str, contexts: List[str], memories: Optional[List[str]] = None,
                      system: Optional[str] = None) -> str:
    sys = system or ("You are mLLM, a small but sharp assistant. Answer using the "
                     "provided context when it is relevant. If the context does not "
                     "contain the answer, say you don't know rather than inventing facts.")
    blocks = [f"<|system|>\n{sys}\n"]
    if memories:
        blocks.append("<|memory|>\n" + "\n---\n".join(memories) + "\n<|/memory|>\n")
    if contexts:
        blocks.append("<|context|>\n" + "\n---\n".join(contexts) + "\n<|/context|>\n")
    blocks.append(f"<|user|>\n{query.strip()}\n<|assistant|>\n")
    return "".join(blocks)


SEARCH_RE = re.compile(r"<\|search\|>(.*?)<\|/search\|>", re.DOTALL)


def extract_search_queries(text: str) -> List[str]:
    return [q.strip() for q in SEARCH_RE.findall(text) if q.strip()]
