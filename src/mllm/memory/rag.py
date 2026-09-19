"""RAGPipeline: <search> -> retrieve -> <context> -> generate loop.

Also owns episodic (per-user) memory: raw turns + rolling summary in SQLite.
Torch-free except for the injected `generate_fn(prompt) -> text` callable,
so the same pipeline serves the real model, a baseline, or a stub in tests.
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Callable, List, Dict, Optional

from ..tokenizer import render_rag_prompt, extract_search_queries
from .retriever import HybridRetriever


class EpisodicMemory:
    """Per-user conversation memory: recent turns + rolling summary."""

    def __init__(self, db_path: str):
        import os
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.con = sqlite3.connect(db_path)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS turns "
            "(user_id TEXT, ts INTEGER, role TEXT, content TEXT)")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS summaries (user_id TEXT PRIMARY KEY, text TEXT)")
        self.con.commit()

    def append(self, user_id: str, role: str, content: str, ts: int = 0):
        self.con.execute("INSERT INTO turns VALUES (?,?,?,?)",
                         (user_id, ts, role, content[:4000]))
        self.con.commit()

    def recent(self, user_id: str, n: int = 10) -> List[Dict[str, str]]:
        rows = self.con.execute(
            "SELECT role, content FROM turns WHERE user_id=? ORDER BY rowid DESC LIMIT ?",
            (user_id, n)).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def get_summary(self, user_id: str) -> str:
        row = self.con.execute("SELECT text FROM summaries WHERE user_id=?",
                               (user_id,)).fetchone()
        return row[0] if row else ""

    def set_summary(self, user_id: str, text: str):
        self.con.execute("INSERT OR REPLACE INTO summaries VALUES (?,?)",
                         (user_id, text[:2000]))
        self.con.commit()

    def update_summary(self, user_id: str, generate_fn: Callable[[str], str]):
        """Rolling summary written by the model itself (keeps memory fresh)."""
        recent = self.recent(user_id, 12)
        if not recent:
            return
        convo = "\n".join(f"{t['role']}: {t['content'][:500]}" for t in recent)
        prompt = ("<|system|>\nSummarize the durable facts about the user and "
                  "conversation so far in under 120 words. Be specific.\n"
                  f"<|user|>\n{convo}\n<|assistant|>\n")
        try:
            self.set_summary(user_id, generate_fn(prompt).strip())
        except Exception:
            pass


class RAGPipeline:
    def __init__(self, knowledge: Optional[HybridRetriever],
                 episodic: Optional[EpisodicMemory] = None,
                 top_k: int = 4, max_context_chars: int = 6000,
                 always_retrieve: bool = False):
        self.knowledge = knowledge
        self.episodic = episodic
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.always_retrieve = always_retrieve

    def _gather_contexts(self, queries: List[str]) -> List[str]:
        ctxs: List[str] = []
        if not self.knowledge:
            return ctxs
        for q in queries[:2]:  # cap retrieval rounds
            for d in self.knowledge.search(q, k=self.top_k):
                ctxs.append(d["text"][:2000])
        # dedupe + budget
        seen, out, budget = set(), [], self.max_context_chars
        for c in ctxs:
            key = c[:120]
            if key in seen:
                continue
            seen.add(key)
            out.append(c[:budget])
            budget -= len(c)
            if budget <= 0:
                break
        return out

    def chat_turn(self, user_id: str, message: str,
                  generate_fn: Callable[[str], str]) -> Dict:
        """One grounded turn. Returns {response, queries, contexts}."""
        memories: List[str] = []
        history: List[Dict[str, str]] = []
        if self.episodic:
            s = self.episodic.get_summary(user_id)
            if s:
                memories.append(f"Conversation summary: {s}")
            history = self.episodic.recent(user_id, 6)

        # Pass 1: let the model decide if it needs search (short probe).
        queries: List[str] = []
        if self.always_retrieve:
            queries = [message]
        else:
            probe_prompt = render_rag_prompt(
                message, [], memories or None,
                system=("You are mLLM. If you need external facts to answer, "
                        "output ONLY: <|search|>query<|/search|>. Otherwise output: NOSEARCH"))
            probe = generate_fn(probe_prompt).strip()
            queries = extract_search_queries(probe)
            if not queries and "NOSEARCH" not in probe and len(message.split()) > 4:
                # fallback: retrieve on substantive questions anyway
                if message.strip().endswith("?"):
                    queries = [message]

        contexts = self._gather_contexts(queries)
        hist_txt = ""
        if history:
            hist_txt = "\n".join(f"{t['role']}: {t['content'][:400]}" for t in history)
            memories.append(f"Recent history:\n{hist_txt}")
        prompt = render_rag_prompt(message, contexts, memories or None)
        response = generate_fn(prompt).strip()

        if self.episodic:
            import time
            ts = int(time.time())
            self.episodic.append(user_id, "user", message, ts)
            self.episodic.append(user_id, "assistant", response, ts)

        return {"response": response, "queries": queries, "contexts": contexts}
