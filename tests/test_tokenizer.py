from mllm.tokenizer import (build_fallback_tokenizer, render_chat,
                              render_rag_prompt, extract_search_queries)


def test_special_tokens_and_chat_template():
    tok = build_fallback_tokenizer(1024)
    assert tok.pad_id == 0 and tok.bos_id == 1 and tok.eos_id == 2
    p = render_chat([{"role": "user", "content": "hello"}])
    assert "<|user|>" in p and p.endswith("<|assistant|>\n")
    ids = tok.encode(p)
    assert len(ids) > 0 and all(0 <= i < 1024 for i in ids)


def test_rag_prompt_and_search_extract():
    p = render_rag_prompt("How tall?", ["Tower is 330m tall."])
    assert "<|context|>" in p and "330m" in p
    assert extract_search_queries("x <|search|>eiffel height<|/search|> y") == ["eiffel height"]
    assert extract_search_queries("no search here") == []


def test_save_load_roundtrip(tmp_path):
    tok = build_fallback_tokenizer(512)
    tok.save(str(tmp_path))
    from mllm.tokenizer import MTokenizer
    t2 = MTokenizer.load(str(tmp_path))
    assert len(t2) == 512 and t2.eos_id == tok.eos_id
