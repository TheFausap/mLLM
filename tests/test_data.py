from mllm.tokenizer import build_fallback_tokenizer
from mllm.data import pack_sequences, format_sft, sample_mix, PRETRAIN_MIX
import random


def test_pack_sequences():
    out = list(pack_sequences([[1, 2], [3, 4, 5], [6]], seq_len=4, eos_id=0))
    assert out[0] == [1, 2, 0, 3]
    assert all(len(s) == 4 for s in out)


def test_sft_masking():
    tok = build_fallback_tokenizer(1024)
    enc = format_sft([{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "hello there"}],
                     tok, max_len=128)
    assert len(enc["input_ids"]) == len(enc["labels"])
    assert enc["labels"][0] == -100  # bos masked
    assert any(l != -100 for l in enc["labels"])  # assistant supervised
    assert enc["input_ids"][-1] == tok.eos_id


def test_mix_weights_sum_to_one():
    assert abs(sum(m.weight for m in PRETRAIN_MIX) - 1.0) < 1e-9
    rng = random.Random(0)
    got = {sample_mix(rng, PRETRAIN_MIX).hf_path for _ in range(200)}
    assert len(got) >= 3  # actually samples across sources
