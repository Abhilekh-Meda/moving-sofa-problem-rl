"""Actor network
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim 

        nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the Gaussian mean for each obs in the batch: shape (batch, action_dim)."""
        raise NotImplementedError