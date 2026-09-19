"""Serving: CLI chat + tiny HTTP server with RAG + episodic memory.

CLI:
    python -m mllm.serve --checkpoint ... --tok ... --memory data/memory --cli

HTTP:
    python -m mllm.serve --checkpoint ... --tok ... --memory data/memory --port 8000
    POST /chat {"user_id": "u1", "message": "hello"}
"""
from __future__ import annotations

import argparse
import json
import os


def build_generate_fn(checkpoint: str, tok_dir: str, max_new: int = 256,
                      temperature: float = 0.7):
    import torch
    from .model import mLLM
    from .tokenizer import MTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = MTokenizer.load(tok_dir)
    model = mLLM.load(checkpoint, device=device,
                      dtype=torch.bfloat16 if device.type == "cuda" else None)
    model.eval()

    def generate(prompt: str) -> str:
        ids = torch.tensor([tok.encode(prompt[-12000:])], device=device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, temperature=temperature,
                                 top_p=0.9, stop_ids=[tok.eos_id],
                                 repetition_penalty=1.05)
        return tok.decode(out[0, ids.shape[1]:].tolist())

    return generate


def build_pipeline(memory_dir: str, **kw):
    from .memory import DiskVectorStore, HybridRetriever, HashEmbedder, RAGPipeline
    from .memory.rag import EpisodicMemory
    knowledge = None
    kpath = os.path.join(memory_dir, "knowledge")
    if os.path.exists(os.path.join(kpath, "texts.db")):
        store = DiskVectorStore(kpath)
        try:
            from .memory.retriever import SentenceTransformerEmbedder
            emb = SentenceTransformerEmbedder()
            assert emb.dim == store.dim
        except Exception as e:
            print(f"[serve] dense embedder fallback (hash): {e}", flush=True)
            emb = HashEmbedder(store.dim)
        knowledge = HybridRetriever(store, emb)
        print(f"[serve] knowledge index: {len(store)} chunks", flush=True)
    else:
        print("[serve] no knowledge index found — running without RAG", flush=True)
    episodic = EpisodicMemory(os.path.join(memory_dir, "episodic.db"))
    return RAGPipeline(knowledge, episodic, **kw)


def cli_loop(pipeline, generate_fn, user_id="local"):
    print("mLLM chat (type /quit to exit, /new to reset memory view)\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg in ("/quit", "/exit"):
            break
        if not msg:
            continue
        res = pipeline.chat_turn(user_id, msg, generate_fn)
        if res["queries"]:
            print(f"  [search: {res['queries']}]")
        print(f"mllm> {res['response']}\n")
        # refresh rolling summary every few turns (cheap, async-ish)
        pipeline.episodic.update_summary(user_id, lambda p: generate_fn(p)[:600])


def http_server(pipeline, generate_fn, port: int):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/chat":
                self.send_response(404)
                self.end_headers()
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            res = pipeline.chat_turn(body.get("user_id", "anon"),
                                     body.get("message", ""), generate_fn)
            data = json.dumps(res).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    print(f"[serve] http://0.0.0.0:{port}/chat", flush=True)
    HTTPServer(("0.0.0.0", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--memory", default="data/memory")
    ap.add_argument("--cli", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--always-retrieve", action="store_true")
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    generate_fn = build_generate_fn(args.checkpoint, args.tok, args.max_new,
                                    args.temperature)
    pipeline = build_pipeline(args.memory, top_k=args.top_k,
                              always_retrieve=args.always_retrieve)
    if args.cli:
        cli_loop(pipeline, generate_fn)
    else:
        http_server(pipeline, generate_fn, args.port)


if __name__ == "__main__":
    main()
