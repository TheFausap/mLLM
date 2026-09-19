import pytest

torch = pytest.importorskip("torch")

from mllm.config import ModelConfig
from mllm.model import mLLM, count_params


def tiny_cfg(**kw):
    d = dict(vocab_size=256, hidden_size=64, num_layers=4, num_heads=4,
             num_kv_heads=2, ffn_dim=128, max_seq_len=128, sliding_window=32,
             global_attention_every=2)
    d.update(kw)
    return ModelConfig(**d)


def test_forward_and_loss():
    cfg = tiny_cfg()
    model = mLLM(cfg)
    x = torch.randint(0, 256, (2, 32))
    out = model(x, labels=x)
    assert out["logits"].shape == (2, 32, 256)
    assert out["loss"].item() > 0


def test_tied_embeddings():
    model = mLLM(tiny_cfg())
    assert model.lm_head is None
    n = count_params(model)["total"]
    assert n == tiny_cfg().num_params


def test_generate_greedy():
    cfg = tiny_cfg()
    model = mLLM(cfg).eval()
    x = torch.randint(0, 256, (1, 16))
    out = model.generate(x, max_new_tokens=8, temperature=0.0)
    assert out.shape == (1, 24)


def test_save_load_roundtrip(tmp_path):
    cfg = tiny_cfg()
    model = mLLM(cfg).eval()
    model.save(str(tmp_path))
    m2 = mLLM.load(str(tmp_path)).eval()
    x = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        a = model(x)["logits"]
        b = m2(x)["logits"]
    assert torch.allclose(a, b)


def test_150m_param_count_matches_config():
    from mllm.config import mllm_150m
    assert mllm_150m().num_params < 170_000_000
