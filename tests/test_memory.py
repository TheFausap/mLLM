import numpy as np
from mllm.memory import DiskVectorStore, HybridRetriever, HashEmbedder, RAGPipeline
from mllm.memory.rag import EpisodicMemory


def _retriever(tmp_path):
    store = DiskVectorStore(str(tmp_path / "idx"), dim=64, shard_size=100)
    emb = HashEmbedder(64)
    retr = HybridRetriever(store, emb)
    texts = [
        "The Eiffel Tower is 330 metres tall including antennas. It stands in Paris.",
        "Penguins cannot fly. They are flightless birds living in the southern hemisphere.",
        "To make an omelette, beat eggs, heat butter in a pan, and fold gently.",
    ]
    retr.index(texts, [{"title": t} for t in ["eiffel", "penguin", "omelette"]])
    return retr


def test_hybrid_retrieval(tmp_path):
    retr = _retriever(tmp_path)
    assert len(retr.store) == 3
    hits = retr.search("how tall is the Eiffel Tower in Paris?", k=2)
    assert hits and "330" in hits[0]["text"]
    hits = retr.search("can penguins fly?", k=2)
    assert hits and "flightless" in hits[0]["text"]


def test_dense_int8_roundtrip(tmp_path):
    store = DiskVectorStore(str(tmp_path / "v"), dim=16, shard_size=10)
    rng = np.random.RandomState(0)
    vecs = rng.randn(5, 16).astype(np.float32)
    store.add(["doc %d" % i for i in range(5)], vecs)
    hits = store.dense_search(vecs[3], k=1)
    assert hits[0][0] == 3  # nearest neighbor is itself


def test_rag_pipeline_with_stub(tmp_path):
    retr = _retriever(tmp_path)
    epi = EpisodicMemory(str(tmp_path / "epi.db"))

    def fake_generate(prompt: str) -> str:
        if "NOSEARCH" in prompt or "output ONLY" in prompt:
            return "<|search|>eiffel tower height<|/search|>"
        if "330" in prompt:
            return "The Eiffel Tower is 330 metres tall."
        return "I don't know."

    pipe = RAGPipeline(retr, epi, top_k=2)
    res = pipe.chat_turn("u1", "How tall is the Eiffel Tower?", fake_generate)
    assert "330" in res["response"], res
    assert res["queries"]
    # episodic memory recorded the turn
    assert len(epi.recent("u1")) == 2
    epi.set_summary("u1", "User asked about Paris.")
    assert "Paris" in epi.get_summary("u1")
