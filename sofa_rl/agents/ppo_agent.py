"""PPO agent"""

from __future__ import annotations

from typing import Dict, List

import gymnasium as gym
import torch

from ..buffer import RolloutBuffer
from ..networks.critic import ValueCritic
from ..policy import ActorPolicy


class PPOAgent:
    def __init__(
        self,
        env: gym.Env,
        policy: ActorPolicy,
        critic: ValueCritic,
        buffer: RolloutBuffer,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        steps_per_batch: int,
    ):
        self.env = env
        self.policy = policy
        self.critic = critic
        self.buffer = buffer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.steps_per_batch = steps_per_batch

    def rollout_episode(self) -> Dict[str, float]:
        """Run one full episode with self.policy, adding each step to
        self.buffer.

        Starts with self.env.reset(). For each step: self.policy.act(obs)
        for (action, log_prob), self.critic(obs) for value, step self.env,
        self.buffer.add(...). Runs until the episode terminates or truncates.
        Returns episode stats: {"return", "length", "success"}.
        """
        obs, _ = self.env.reset()
        done = False
        episode_return = 0.0
        steps = 0

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0) # (1, obs_dim)

            with torch.no_grad():
                action, log_prob = self.policy.act(obs_tensor) # (1, action_dim), (1,)
                value = self.critic(obs_tensor) # (1, 1)
                action_np = action.squeeze(0).cpu().numpy() # np.array (action_dim,)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            self.buffer.add(obs_tensor.squeeze(0), action.squeeze(0), reward, value.squeeze(), log_prob.squeeze(), done)
            obs = next_obs
            episode_return += reward
            steps += 1

        return {"return": episode_return, "length": steps, "success": float(info["success"])}

    def collect_rollout(self) -> List[Dict[str, float]]:
        """Call rollout_episode repeatedly (each starting from a fresh
        self.env.reset()) until self.buffer has at least self.steps_per_batch
        steps. No episode is cut off mid-way: the last rollout_episode()
        call is always allowed to finish, even if it pushes the buffer past
        steps_per_batch.
        """
        self.buffer.reset()
        episodes = []

        while not len(self.buffer) >= self.steps_per_batch:
            episodes.append(self.rollout_episode())
        
        return episodes

    def update(self) -> Dict[str, float]:
        """Run PPO's minibatch epochs over self.buffer, then self.buffer.reset().

        Compute advantages/returns (e.g. GAE) from self.buffer's rewards,
        values, and dones; run K epochs of minibatches through
        self.policy.evaluate_actions(...) and self.critic(...) for the
        clipped surrogate loss, value loss, and entropy bonus; step both
        optimizers. Return a dict of stats (e.g. losses) for logging.
        """
        raise NotImplementedError
