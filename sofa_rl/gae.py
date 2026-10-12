"""Generalized Advantage Estimation over a flat buffer of complete episodes."""
from __future__ import annotations

from typing import Tuple

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and critic return targets.

    rewards, values, dones: tensors of shape (N,) in collection order.
    dones[t] = 1.0 if step t ended its episode (terminated or truncated).
    The buffer must end on an episode boundary, so the last step has no
    bootstrap value.
    Returns (advantages, returns), each of shape (N,).
    """
    N = len(rewards)
    advantages = torch.zeros(N)
    gae = 0

    for t in reversed(range(N)):
        next_value = values[t + 1] if t < N - 1 else 0.0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, normalized_advantages, returns
