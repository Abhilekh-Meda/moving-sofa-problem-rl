"""Rollout storage for on-policy algorithms (PPO).

Pure storage: appends to plain lists as steps come in, sized dynamically
(no buffer_size known upfront -- collection runs to complete episodes, see
PPOAgent.collect_rollout). No sampling, advantage/return computation, or
minibatch logic
"""
from __future__ import annotations

import torch


class RolloutBuffer:
    def __init__(self, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.reset()

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(float(done))

    def reset(self) -> None:
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
