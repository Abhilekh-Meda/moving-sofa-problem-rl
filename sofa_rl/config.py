from dataclasses import dataclass

@dataclass(frozen=True)
class PPOConfig:
    lam: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 10
    minibatch_size: int = 64
    entropy_coef: float = 0.0