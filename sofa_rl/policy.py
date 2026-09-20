"""Policy interface for PPO on SofaEnv.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class ActorPolicy(nn.Module, ABC):
    """Action-selection wrapper around an actor network.

    obs and action tensors are always batched, shape (batch, obs_dim) and
    (batch, action_dim), matching SofaEnv's Box observation/action spaces.
    """

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor

    @abstractmethod
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action for rollout collection.

        Returns (action, log_prob), each shape (batch, ...).
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate given actions under the current policy, for the PPO update.

        Returns (log_prob, entropy), each shape (batch,).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Return an action for evaluation/inference (no log_prob needed)."""
        raise NotImplementedError
