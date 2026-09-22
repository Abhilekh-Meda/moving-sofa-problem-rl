"""Actor network
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.net =  nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.log_std = nn.Parameter(torch.ones(action_dim) * -3.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mu = self.net(obs) # (B, A)
        std = self.log_std.exp().expand_as(mu) # (B, A)
        return Normal(mu, std)
