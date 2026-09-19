from mllm.config import mllm_150m, mllm_350m, mllm_600m, TrainConfig


def test_param_counts_under_1b():
    for fn in (mllm_150m, mllm_350m, mllm_600m):
        cfg = fn()
        n = cfg.num_params
        print(fn.__name__, cfg.summary())
        assert n < 1_000_000_000, f"{fn.__name__} = {n} params, must be < 1B"


def test_ladder_ordering():
    a, b, c = mllm_150m().num_params, mllm_350m().num_params, mllm_600m().num_params
    assert a < b < c


def test_expected_sizes():
    assert 100_000_000 < mllm_150m().num_params < 170_000_000
    assert 280_000_000 < mllm_350m().num_params < 420_000_000
    assert 480_000_000 < mllm_600m().num_params < 700_000_000


def test_wsd_schedule():
    t = TrainConfig(tokens_total=10**9, batch_tokens=10**6, warmup_steps=100,
                    peak_lr=1e-3, min_lr=1e-4, decay_frac=0.2)
    assert t.steps() == 1000
    assert t.lr_at(0) < t.lr_at(99)
    assert abs(t.lr_at(100) - 1e-3) < 1e-9
    assert t.lr_at(999) < t.lr_at(800)
