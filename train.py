"""Train a PPO agent on SofaEnv to maximise the area of the sofa carved out by
the rotation path.

Writes config.json and metrics.jsonl to runs/<run_name>/.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from sofa_env import SofaEnv
from sofa_rl import GaussianActor, GaussianActorPolicy, PPOAgent, RolloutBuffer, ValueCritic
from sofa_rl.agents.ppo_agent import summarize_episodes
from sofa_rl.config import TrainConfig

RUNS_DIR = Path("runs")


def lr_lambda(schedule: str, n_updates: int):
    if schedule == "linear":
        return lambda step: 1.0 - step / n_updates
    if schedule == "constant":
        return lambda step: 1.0
    raise ValueError(f"unknown lr_schedule {schedule!r}")


def main(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    run_name = cfg.run_name or f"n{cfg.env.n_steps}_seed{cfg.seed}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    env = SofaEnv(**asdict(cfg.env))
    env.reset(seed=cfg.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = GaussianActor(obs_dim, action_dim, **asdict(cfg.network)) # network
    policy = GaussianActorPolicy(actor) # policy
    critic = ValueCritic(obs_dim, hidden_dim=cfg.network.hidden_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.optim.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.optim.critic_lr)

    n_updates = cfg.total_steps // cfg.ppo.steps_per_batch
    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg.optim.lr_schedule, n_updates))
        for opt in (actor_optimizer, critic_optimizer)
    ]

    agent = PPOAgent(
        env=env,
        policy=policy,
        critic=critic,
        buffer=RolloutBuffer(),
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        config=cfg.ppo,
    )

    total_steps = 0
    start = time.time()
    with open(run_dir / "metrics.jsonl", "w") as metrics_file:
        for update in range(n_updates):
            episodes = agent.collect_rollout()
            update_stats = agent.update()
            actor_lr = actor_optimizer.param_groups[0]["lr"]

            for scheduler in schedulers:
                scheduler.step()

            episode_stats = summarize_episodes(episodes)
            total_steps += episode_stats["num_steps"]
            row = {
                "update": update,
                "total_steps": total_steps,
                "actor_lr": actor_lr,
                "max_final_area": max(e["final_area"] for e in episodes),
                **episode_stats,
                **update_stats,
            }
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()

            print(
                f"update {update + 1}/{n_updates} | steps {total_steps} "
                f"| area mean {row['mean_final_area']:.4f} max {row['max_final_area']:.4f} "
                f"| kl {row['approx_kl']:.4f} | ev {row['explained_variance']:.2f} "
                f"| std {row['policy_std']:.3f} | {time.time() - start:.0f}s"
            )


if __name__ == "__main__":
    main(TrainConfig())
