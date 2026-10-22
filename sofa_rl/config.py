"""Training configuration, one frozen dataclass per component.

Components take plain constructor arguments; train.py unpacks these configs
into them. TrainConfig nests all of them and is what gets saved per run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class EnvConfig:
    n_steps: int = 200
    max_corner_speed: float = 2.0


@dataclass(frozen=True)
class NetworkConfig:
    hidden_dim: int = 64
    log_std_init: float = -3.0


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 1.0
    lam: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 10
    minibatch_size: int = 64
    entropy_coef: float = 0.0
    steps_per_batch: int = 4096


@dataclass(frozen=True)
class OptimConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    lr_schedule: str = "linear"  # "linear" (decay to 0) or "constant"


@dataclass(frozen=True)
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    total_steps: int = 1_000_000
    seed: int = 0
    run_name: Optional[str] = None
