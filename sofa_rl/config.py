"""Training configuration, one frozen dataclass per component.

Components take plain constructor arguments; train.py unpacks these configs
into them. TrainConfig nests all of them and is what gets saved per run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EnvConfig:
    n_steps: int = 200
    max_corner_speed: float = 2.0
    substeps: int = 1  # hallway cuts per decision step
    reward_area: str = "sampled"  # "sampled" or "certified" (lower bound for the continuous motion)
    cert_dirs: int = 5


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
    # warm start: a checkpoint path (relative to train.RUNS_DIR, or absolute) whose
    # actor and critic weights initialise this run; optimisers start fresh
    init_checkpoint: Optional[str] = None
    # with a warm start, keep the checkpoint's learned log_std, or reset it to network.log_std_init
    init_keep_log_std: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        """Inverse of dataclasses.asdict, e.g. for a config.json or a config sent to a remote worker."""
        return cls(
            env=EnvConfig(**d["env"]),
            network=NetworkConfig(**d["network"]),
            ppo=PPOConfig(**d["ppo"]),
            optim=OptimConfig(**d["optim"]),
            **{k: v for k, v in d.items() if k not in ("env", "network", "ppo", "optim")},
        )
