"""PPO agent"""

from __future__ import annotations

from typing import Dict, List

import gymnasium as gym
import torch

from ..buffer import RolloutBuffer
from ..gae import compute_gae
from ..networks.critic import ValueCritic
from ..policy import ActorPolicy
from ..config import PPOConfig


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
        config: PPOConfig
    ):
        self.env = env
        self.policy = policy
        self.critic = critic
        self.buffer = buffer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.steps_per_batch = steps_per_batch
        self.cfg = config

    def _rollout_episode(self) -> Dict[str, float]:
        """Run one full episode with self.policy, adding each step to
        self.buffer.

        Starts with self.env.reset(). For each step: self.policy.act(obs)
        for (action, log_prob), self.critic(obs) for value, step self.env,
        self.buffer.add(...). Runs until the episode terminates or truncates.
        Returns episode stats: {"return", "length", "final_area"}.
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
            self.buffer.add(obs_tensor.squeeze(0), action.squeeze(0), reward, value.squeeze(), log_prob.squeeze(), truncated, terminated)
            obs = next_obs
            episode_return += reward
            steps += 1

        return {
            "return": episode_return,
            "length": steps,
            "final_area": float(info["area"]),
        }

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
            episodes.append(self._rollout_episode())
        
        return episodes

    def update(self) -> Dict[str, float]:
        """Run PPO's minibatch epochs over self.buffer, then self.buffer.reset().

        Compute advantages/returns (e.g. GAE) from self.buffer's rewards,
        values, and dones; run K epochs of minibatches through
        self.policy.evaluate_actions(...) and self.critic(...) for the
        clipped surrogate loss, value loss, and entropy bonus; step both
        optimizers. Return a dict of stats (e.g. losses) for logging.
        """
        obs, actions, rewards, values, log_probs, terminateds, truncateds = self.buffer.get()
        obs_t = torch.stack(obs) # (B, obs_dim)
        actions_t = torch.stack(actions) # (B, action_dim)
        rewards_t = torch.tensor(rewards) # (B,)
        values_t = torch.stack(values) # (B,)
        log_probs_t = torch.stack(log_probs) # (B,)
        dones_t = torch.logical_or(torch.tensor(terminateds), torch.tensor(truncateds)).float()

        advantages, normalized_advantages, returns = compute_gae(rewards_t, values_t, dones_t, gamma=self.cfg.gamma, lam=self.cfg.lam) # all are (B,)

        explained_variance = 1.0 - (returns - values_t).var() / (returns.var() + 1e-8)
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}

        B = len(obs_t)
        for epoch in range(self.cfg.epochs):
            perm = torch.randperm(B)
            for start in range(0, B, self.cfg.minibatch_size):
                idx = perm[start : start+self.cfg.minibatch_size]

                # slicing minibatch
                obs_mb = obs_t[idx]
                actions_mb = actions_t[idx]
                old_log_probs_mb = log_probs_t[idx]
                advantages_mb = advantages[idx]
                normalized_advantages_mb = normalized_advantages[idx]
                returns_mb = returns[idx]

                # forward passes and losses
                new_log_probs, entropy = self.policy.evaluate_actions(obs_mb, actions_mb)
                ratio = (new_log_probs - old_log_probs_mb).exp()
                clipped_ratio = torch.clip(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                surrogate_objective = torch.min(ratio * normalized_advantages_mb, clipped_ratio * normalized_advantages_mb)
                entropy_bonus = self.cfg.entropy_coef * torch.mean(entropy)
                policy_loss = -surrogate_objective.mean() - entropy_bonus

                value = self.critic(obs_mb).squeeze(-1)
                value_loss = ((value - returns_mb) ** 2).mean()

                # logging stats
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.mean().item())
                stats["approx_kl"].append((old_log_probs_mb - new_log_probs).mean().item())
                stats["clip_fraction"].append(((ratio - 1).abs() > self.cfg.clip_eps).float().mean().item())

                # gradient steps
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()
            
        self.buffer.reset()

        result = {k: sum(v) / len(v) for k, v in stats.items()}
        result["explained_variance"] = explained_variance.item()
        result["policy_std"] = self.policy.actor.log_std.exp().mean().item()
        return result


def summarize_episodes(episodes: List[Dict[str, float]]) -> Dict[str, float]:
    """Average per-episode stats from collect_rollout() into one dict."""
    n = len(episodes)
    summary = {f"mean_{k}": sum(e[k] for e in episodes) / n for k in episodes[0]}
    summary["num_episodes"] = n
    summary["num_steps"] = sum(e["length"] for e in episodes)
    return summary
    