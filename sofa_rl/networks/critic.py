"""Critic network
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ValueCritic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.obs_dim = obs_dim

        # TODO: define layers (e.g. an MLP obs_dim -> ... -> 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the value estimate for each obs in the batch: shape (batch,)."""
        raise NotImplementedError
