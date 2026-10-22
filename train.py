"""Train a PPO agent on SofaEnv to maximise the area of the sofa carved out by
the rotation path.

Writes config.json and metrics.jsonl to runs/<RUN_NAME>/.
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
from sofa_rl.config import PPOConfig

# env
N_STEPS = 200
MAX_CORNER_SPEED = 2.0

# optimisation
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
STEPS_PER_BATCH = 4096
TOTAL_STEPS = 1_000_000
PPO_CONFIG = PPOConfig()

# run
SEED = 0
RUN_NAME = f"n{N_STEPS}_seed{SEED}"
RUN_DIR = Path("runs") / RUN_NAME


def linear_decay(n_updates: int):
    return lambda step: 1.0 - step / n_updates


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    env = SofaEnv(n_steps=N_STEPS, max_corner_speed=MAX_CORNER_SPEED)
    env.reset(seed=SEED)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = GaussianActor(obs_dim, action_dim) # network
    policy = GaussianActorPolicy(actor) # policy
    critic = ValueCritic(obs_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    n_updates = TOTAL_STEPS // STEPS_PER_BATCH
    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(opt, linear_decay(n_updates))
        for opt in (actor_optimizer, critic_optimizer)
    ]

    agent = PPOAgent(
        env=env,
        policy=policy,
        critic=critic,
        buffer=RolloutBuffer(),
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        steps_per_batch=STEPS_PER_BATCH,
        config=PPO_CONFIG,
    )

    config = {
        "n_steps": N_STEPS,
        "max_corner_speed": MAX_CORNER_SPEED,
        "actor_lr": ACTOR_LR,
        "critic_lr": CRITIC_LR,
        "steps_per_batch": STEPS_PER_BATCH,
        "total_steps": TOTAL_STEPS,
        "seed": SEED,
        "ppo": asdict(PPO_CONFIG),
    }
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))

    total_steps = 0
    start = time.time()
    with open(RUN_DIR / "metrics.jsonl", "w") as metrics_file:
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
    main()
