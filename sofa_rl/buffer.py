"""Rollout storage for on-policy algorithms (PPO).
"""
from __future__ import annotations

import torch


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device | str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.device = torch.device(device)

        self.obs = torch.zeros((buffer_size, obs_dim), device=self.device)
        self.actions = torch.zeros((buffer_size, action_dim), device=self.device)
        self.rewards = torch.zeros(buffer_size, device=self.device)
        self.values = torch.zeros(buffer_size, device=self.device)
        self.log_probs = torch.zeros(buffer_size, device=self.device)
        self.dones = torch.zeros(buffer_size, device=self.device)

        self.pos = 0

    @property
    def full(self) -> bool:
        return self.pos >= self.buffer_size

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool,
    ) -> None:
        if self.full:
            raise RuntimeError("RolloutBuffer is full; reset() before adding more")
        self.obs[self.pos] = obs            # single: (obs_dim,), batch: (buffer_size, obs_dim)
        self.actions[self.pos] = action     # single: (action_dim,), batch: (buffer_size, action_dim)
        self.rewards[self.pos] = reward     # single: (), batch: (buffer_size,)
        self.values[self.pos] = value       # single: (), batch: (buffer_size,)
        self.log_probs[self.pos] = log_prob # single: (), batch: (buffer_size,)
        self.dones[self.pos] = float(done)  # single: (), batch: (buffer_size,)
        self.pos += 1

    def reset(self) -> None:
        self.pos = 0
