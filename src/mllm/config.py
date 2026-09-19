"""Model + training configs. Pure python (no torch) so tests run anywhere."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import math


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 576
    num_layers: int = 30
    num_heads: int = 9
    num_kv_heads: int = 3
    ffn_dim: int = 1536
    max_seq_len: int = 4096
    rope_theta: float = 100000.0
    rope_scaling: Optional[str] = None  # None | "yarn"
    rope_factor: float = 2.0            # for 4k -> 8k extension in SFT/RAG
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    sliding_window: Optional[int] = 1024
    global_attention_every: int = 4     # every Nth layer attends globally
    use_qk_norm: bool = True
    dropout: float = 0.0

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, "hidden must divide by heads"
        assert self.num_heads % self.num_kv_heads == 0, "GQA groups must divide evenly"
        self.head_dim = self.hidden_size // self.num_heads

    @property
    def num_params(self) -> int:
        """Approximate parameter count (excludes disk memory by design)."""
        d = self.hidden_size
        kv_dim = self.head_dim * self.num_kv_heads
        per_layer = (
            d * d +            # q
            d * kv_dim +       # k
            d * kv_dim +       # v
            d * d +            # o
            3 * d * self.ffn_dim +  # SwiGLU gate/up/down
            2 * d +            # 2x RMSNorm weight
            (2 * self.head_dim if self.use_qk_norm else 0)
        )
        total = per_layer * self.num_layers
        total += self.vocab_size * d  # embeddings (tied -> counted once)
        if not self.tie_embeddings:
            total += self.vocab_size * d  # lm head
        total += d  # final norm
        return int(total)

    def summary(self) -> str:
        n = self.num_params
        return (
            f"layers={self.num_layers} hidden={self.hidden_size} "
            f"heads={self.num_heads}/{self.num_kv_heads} ffn={self.ffn_dim} "
            f"vocab={self.vocab_size} tied={self.tie_embeddings} "
            f"swa={self.sliding_window} global_every={self.global_attention_every} "
            f"params={n/1e6:.1f}M"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("head_dim", None) if "head_dim" in d else None
        return d


# ----------------------------------------------------------------------------
# The ladder: deep & thin (MobileLLM-style) wins at small scale.
# ----------------------------------------------------------------------------

def mllm_150m(**kw) -> ModelConfig:
    cfg = ModelConfig(
        hidden_size=576, num_layers=30, num_heads=9, num_kv_heads=3,
        ffn_dim=1536, **kw,
    )
    return cfg


def mllm_350m(**kw) -> ModelConfig:
    cfg = ModelConfig(
        hidden_size=960, num_layers=32, num_heads=12, num_kv_heads=4,
        ffn_dim=2560, **kw,
    )
    return cfg


def mllm_600m(**kw) -> ModelConfig:
    cfg = ModelConfig(
        hidden_size=1248, num_layers=32, num_heads=16, num_kv_heads=4,
        ffn_dim=3328, **kw,
    )
    return cfg


NAMED_MODELS = {
    "150m": mllm_150m,
    "350m": mllm_350m,
    "600m": mllm_600m,
}


@dataclass
class TrainConfig:
    # data
    tokens_total: int = 80_000_000_000
    seq_len: int = 4096
    batch_tokens: int = 1_048_576  # ~1M tokens/step (~256 seqs @4k)
    # optimization (WSD schedule)
    peak_lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 2000
    decay_frac: float = 0.10      # last 10% of steps decay to min_lr
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    z_loss: float = 1e-4
    # system
    precision: str = "bf16"       # bf16 mixed on DGX Spark
    grad_accum: int = 1
    checkpoint_every: int = 2000
    eval_every: int = 500
    log_every: int = 20
    seed: int = 1337
    compile: bool = True
    grad_checkpoint: bool = False

    def steps(self) -> int:
        return max(1, self.tokens_total // self.batch_tokens)

    def lr_at(self, step: int) -> float:
        """Warmup-Stable-Decay schedule."""
        total = self.steps()
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / max(1, self.warmup_steps)
        decay_start = int(total * (1 - self.decay_frac))
        if step < decay_start:
            return self.peak_lr
        frac = (step - decay_start) / max(1, total - decay_start)
        # cosine decay to min_lr
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * frac))
