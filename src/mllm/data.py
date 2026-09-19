"""Data: streaming pretrain mixes, packing, SFT formatting, RAG augmentation.

Torch-free formatting utilities + optional torch Dataset wrappers (lazy import)
so unit tests run without the training stack.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator, List, Dict, Optional, Sequence

from .tokenizer import MTokenizer, render_chat


# ----------------------------------------------------------------------------
# Pretraining mixes (Stage 1/2). Weights must sum to 1. All English-first.
# ----------------------------------------------------------------------------

@dataclass
class MixItem:
    hf_path: str          # HF dataset path
    hf_name: Optional[str] = None
    text_field: str = "text"
    weight: float = 0.0
    note: str = ""


PRETRAIN_MIX: List[MixItem] = [
    MixItem("HuggingFaceFW/fineweb-edu", None, "text", 0.50, "high-quality educational web"),
    MixItem("mlfoundations/dclm-baseline-1.0", None, "text", 0.20, "filtered web diversity"),
    MixItem("HuggingFaceTB/cosmopedia", "web_samples_v2", "text", 0.15, "synthetic textbooks (Phi-style)"),
    MixItem("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text", 0.05, "stories+textbooks"),
    MixItem("open-web-math/open-web-math", None, "text", 0.05, "math reasoning"),
    MixItem("bigcode/starcoderdata", "python", "content", 0.05, "code, light dose"),
]

ANNEAL_MIX: List[MixItem] = [
    MixItem("HuggingFaceTB/cosmopedia", "web_samples_v2", "text", 0.30, "textbook quality"),
    MixItem("HuggingFaceFW/fineweb-edu", None, "text", 0.30, "edu web"),
    MixItem("HuggingFaceTB/smollm-corpus", "smoltalk", "text", 0.20, "conversational pretrain"),
    MixItem("open-web-math/open-web-math", None, "text", 0.10, "math"),
    MixItem("bigcode/starcoderdata", "python", "content", 0.10, "code"),
]

SFT_SOURCES: List[str] = [
    "HuggingFaceTB/smoltalk",        # 460k+ high-quality convos, SmolLM recipe
    "HuggingFaceH4/ultrachat_200k",  # multi-turn depth
    "teknium/OpenHermes-2.5",        # instruction diversity
    "HuggingFaceTB/ultrafeedback_binarized",  # DPO stage
]


def sample_mix(rng: random.Random, mix: Sequence[MixItem]) -> MixItem:
    r = rng.random()
    acc = 0.0
    for m in mix:
        acc += m.weight
        if r <= acc:
            return m
    return mix[-1]


# ----------------------------------------------------------------------------
# Packing: concatenate tokenized docs into fixed-length rows with <eos> separators
# ----------------------------------------------------------------------------

def pack_sequences(seqs: Sequence[List[int]], seq_len: int, eos_id: int) -> Iterator[List[int]]:
    buf: List[int] = []
    for s in seqs:
        buf += s + [eos_id]
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            buf = buf[seq_len:]


def tokenize_stream(texts: Iterator[str], tok: MTokenizer) -> Iterator[List[int]]:
    for t in texts:
        if t and t.strip():
            yield tok.encode(t[:20000])  # guard pathological docs


# ----------------------------------------------------------------------------
# SFT formatting with loss masking (mask everything before each assistant turn)
# ----------------------------------------------------------------------------

def format_sft(messages: List[Dict[str, str]], tok: MTokenizer,
               max_len: int = 8192) -> Dict[str, List[int]]:
    """Returns input_ids + labels (labels=-100 on non-assistant spans)."""
    input_ids: List[int] = []
    labels: List[int] = []
    assistant_tag = tok.encode("<|assistant|>\n")
    for i, m in enumerate(messages):
        chunk = f"<|{m['role']}|>\n{m['content'].strip()}\n"
        ids = tok.encode(chunk)
        input_ids += ids
        if m["role"] == "assistant":
            labels += ids
        else:
            labels += [-100] * len(ids)
    if messages and messages[-1]["role"] != "assistant":
        input_ids += tok.encode("<|assistant|>\n")
        labels += [-100] * len(tok.encode("<|assistant|>\n"))
    input_ids = [tok.bos_id] + input_ids[:max_len - 2] + [tok.eos_id]
    labels = [-100] + labels[:max_len - 2] + [tok.eos_id]
    return {"input_ids": input_ids, "labels": labels}


def format_rag_sft(query: str, contexts: List[str], answer: str,
                   tok: MTokenizer, search_query: Optional[str] = None,
                   max_len: int = 8192) -> Dict[str, List[int]]:
    """RAG-tool-use trajectory: optionally emit <search> then grounded answer."""
    from .tokenizer import render_rag_prompt
    if search_query:
        # model learns: question -> search call -> (context injected) -> answer
        pre = (f"<|user|>\n{query.strip()}\n<|assistant|>\n"
               f"<|search|>{search_query.strip()}<|/search|>\n")
        post = render_rag_prompt(query, contexts)
        post += answer.strip() + "\n"
        full = pre + post.split("<|assistant|>\n", 1)[1] if "<|assistant|>" in post else pre + answer
        input_ids = [tok.bos_id] + tok.encode(full)[:max_len - 2] + [tok.eos_id]
        # mask user/context spans: only supervise search call + final answer
        labels = list(input_ids)
        # simple approach: mask everything up to first <|assistant|>
        prefix = tok.encode(pre.split("<|assistant|>\n")[0] + "<|assistant|>\n")
        n_mask = min(len(labels), 1 + len(prefix))
        for j in range(n_mask):
            labels[j] = -100
        labels[-1] = tok.eos_id
        return {"input_ids": input_ids, "labels": labels}
    prompt = render_rag_prompt(query, contexts)
    return format_sft(
        [{"role": "user", "content": query},
         {"role": "assistant", "content": answer}], tok, max_len)


# ----------------------------------------------------------------------------
# HF streaming loaders (import datasets lazily; used by train.py on the Spark)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Preflight: HF streaming needs compression codecs (fail fast, clear message)
# ----------------------------------------------------------------------------

def check_streaming_codecs() -> None:
    """Verify zstd/lz4 codecs exist before opening streaming datasets.

    Without these, `datasets` streaming dies deep inside fsspec with
    `ValueError: Compression type zstd not supported`. Raise a helpful
    error instead.
    """
    missing = []
    try:
        import zstandard  # noqa: F401
    except ImportError:
        missing.append("zstandard")
    try:
        import lz4.frame  # noqa: F401
    except ImportError:
        missing.append("lz4")
    if missing:
        raise RuntimeError(
            "Missing streaming codec(s): " + ", ".join(missing) + ". "
            "mLLM datasets (FineWeb-Edu, DCLM, ...) are zstd/lz4 compressed. "
            "Install with: pip install " + " ".join(missing) +
            '  (or the full stack: pip install -e ".[train]")'
        )


def stream_hf_texts(hf_path: str, hf_name: Optional[str], text_field: str,
                    split: str = "train", seed: int = 0):
    check_streaming_codecs()
    import datasets
    ds = datasets.load_dataset(hf_path, hf_name, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for row in ds:
        t = row.get(text_field, "")
        if isinstance(t, str) and len(t) > 50:
            yield t


def mixed_pretrain_stream(mix: Sequence[MixItem], seed: int = 0) -> Iterator[str]:
    rng = random.Random(seed)
    iters = {m.hf_path + (m.hf_name or ""): stream_hf_texts(m.hf_path, m.hf_name, m.text_field, seed=rng.randint(0, 10**9))
             for m in mix}
    while True:
        m = sample_mix(rng, mix)
        try:
            yield next(iters[m.hf_path + (m.hf_name or "")])
        except StopIteration:
            continue


# ----------------------------------------------------------------------------
# Pre-flight probing: fail fast on wrong dataset/config/field names
# ----------------------------------------------------------------------------

def source_name(m: MixItem) -> str:
    return f"{m.hf_path}:{m.hf_name or 'default'}[{m.text_field}]"


def probe_source(m: MixItem, n_rows: int = 3) -> Dict:
    """Try loading a streaming source and reading n_rows. Never raises."""
    import time
    t0 = time.time()
    try:
        it = stream_hf_texts(m.hf_path, m.hf_name, m.text_field, seed=0)
        rows = [next(it) for _ in range(n_rows)]
        return {"ok": True, "rows": len(rows),
                "chars": sum(len(r) for r in rows),
                "dt": round(time.time() - t0, 1)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                "dt": round(time.time() - t0, 1)}


def load_first_available(path: str, configs: Sequence[Optional[str]],
                         split: str = "train", need_field: Optional[str] = None):
    """Try dataset configs in order (streaming); return (dataset, config_used).

    Probes one row to force resolution now, since streaming is lazy.
    """
    import datasets
    errors = []
    for cfg in configs:
        try:
            ds = datasets.load_dataset(path, cfg, split=split, streaming=True)
            first = next(iter(ds))
            if need_field and need_field not in first:
                raise ValueError(f"config {cfg!r} has no {need_field!r} field")
            ds = datasets.load_dataset(path, cfg, split=split, streaming=True)
            return ds, cfg
        except Exception as e:
            errors.append(f"{cfg}: {type(e).__name__}: {str(e)[:150]}")
    raise RuntimeError(f"no working config for {path} {list(configs)}: {errors}")


def validate_mix(mix: Sequence[MixItem], label: str = "mix") -> None:
    """Probe every source; raise with a clear report if any fail."""
    bad = []
    for m in mix:
        r = probe_source(m)
        name = source_name(m)
        if r["ok"]:
            print(f"[data] {label} OK   {name} "
                  f"({r['chars']} chars in {r['dt']}s)", flush=True)
        else:
            print(f"[data] {label} FAIL {name}: {r['error']}", flush=True)
            bad.append(name)
    if bad:
        raise RuntimeError(
            f"{len(bad)} source(s) in {label} failed to load: {bad}. "
            "Run `python scripts/check_data.py` for a full report.")


# ----------------------------------------------------------------------------
# Torch datasets (lazy import)
# ----------------------------------------------------------------------------

def pretrain_batch_iter(text_stream, tok: MTokenizer, seq_len: int, batch_seqs: int):
    """Yield (input_ids, labels) numpy/torch batches of packed sequences."""
    import numpy as np
    try:
        import torch
        use_torch = True
    except ImportError:
        use_torch = False
    toks = tokenize_stream(text_stream, tok)
    packed = pack_sequences(toks, seq_len + 1, tok.eos_id)
    batch = []
    for seq in packed:
        batch.append(seq)
        if len(batch) == batch_seqs:
            arr = np.array(batch, dtype=np.int64)
            x, y = arr[:, :-1], arr[:, 1:]
            if use_torch:
                yield torch.from_numpy(x), torch.from_numpy(y)
            else:
                yield x, y
            batch = []
