"""Concrete ActorPolicy for a diagonal-Gaussian actor (e.g. GaussianActor).

Assumes self.actor(obs) returns a torch.distributions.Normal-like object
supporting .sample(), .log_prob(), .entropy(), and .mean.
"""
from __future__ import annotations

from typing import Tuple

import torch

from ..policy import ActorPolicy


class GaussianActorPolicy(ActorPolicy):
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        actions = dist.sample()  # (B, A)
        log_prob = dist.log_prob(actions).sum(-1)  # (B,)
        return actions, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        log_prob = dist.log_prob(actions).sum(-1)  # (B,)
        entropy = dist.entropy().sum(-1)  # (B,)
        return log_prob, entropy

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self.actor(obs)
        if deterministic:
            return dist.mean
        return dist.sample()
