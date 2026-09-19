from mllm.tokenizer import build_fallback_tokenizer
from mllm.data import (pack_sequences, format_sft, sample_mix, PRETRAIN_MIX,
                       check_streaming_codecs)
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


def test_check_streaming_codecs_message():
    # Passes whether or not codecs are installed: either no raise, or a
    # RuntimeError that tells the user exactly what to pip install.
    try:
        check_streaming_codecs()
    except RuntimeError as e:
        assert "pip install" in str(e)
        assert "zstandard" in str(e) or "lz4" in str(e)


def test_mix_config_names_known_good():
    # Regression test: cosmopedia has no 'v2' config (it's web_samples_v2, ...),
    # starcoderdata needs an explicit language subset. See scripts/check_data.py
    # for the live probe that runs on the Spark.
    from mllm.data import ANNEAL_MIX
    for m in list(PRETRAIN_MIX) + list(ANNEAL_MIX):
        if m.hf_path == "HuggingFaceTB/cosmopedia":
            assert m.hf_name in ("web_samples_v1", "web_samples_v2", "stories",
                                 "stanford", "wikihow", "openstax",
                                 "auto_math_text", "kunst-stories"), m
        if m.hf_path == "bigcode/starcoderdata":
            assert m.hf_name is not None and m.text_field == "content", m
        if m.hf_path == "HuggingFaceTB/smollm-corpus":
            assert m.hf_name in ("cosmopedia-v2", "fineweb-edu-dedup",
                                 "python-edu", "smoltalk"), m


def test_mix_weights_sum_to_one():
    assert abs(sum(m.weight for m in PRETRAIN_MIX) - 1.0) < 1e-9
    rng = random.Random(0)
    got = {sample_mix(rng, PRETRAIN_MIX).hf_path for _ in range(200)}
    assert len(got) >= 3  # actually samples across sources
