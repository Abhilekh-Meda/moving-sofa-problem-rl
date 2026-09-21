"""Critic network
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ValueCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.output

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) # (B, 1)