from mllm.eval import repetition_rate, coherence_report, run_ifeval, run_grounding


def test_repetition_rate():
    assert repetition_rate("the cat sat on the mat") < 0.2
    assert repetition_rate("hello world " * 30) > 0.8


def test_ifeval_with_oracle():
    def oracle(prompt: str) -> str:
        if "single word" in prompt:
            return "Apple"
        if "number 42" in prompt:
            return "42"
        if "BLUE" in prompt:
            return "BLUE BLUE BLUE BLUE BLUE"
        if "UPPERCASE" in prompt:
            return "HELLO"
        if "xylophone" in prompt:
            return "I love my xylophone and your xylophone."
        if "three sentences" in prompt:
            return "One. Two. Three."
        if "4 animals" in prompt:
            return "1. cat\n2. dog\n3. fox\n4. owl"
        if "Apples" in prompt:
            return "Apples are red.\nBananas are yellow."
        return "?"
    res = run_ifeval(oracle)
    assert res["accuracy"] == 1.0, res


def test_grounding_with_oracle():
    def oracle(prompt: str) -> str:
        if "Tokyo" in prompt:
            return "I don't know — the context has no information about Tokyo."
        if "Penguins" in prompt or "penguins" in prompt:
            return "No, penguins cannot fly; they are flightless birds."
        return "The Eiffel Tower is 330 metres tall."
    res = run_grounding(oracle)
    assert res["accuracy"] == 1.0, res


def test_coherence_report():
    rep = coherence_report(["hello world", ""])
    assert rep["empty_rate"] == 0.5
