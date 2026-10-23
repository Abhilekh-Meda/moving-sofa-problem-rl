"""Train a PPO agent on SofaEnv to maximise the area of the sofa carved out by
the rotation path.

Writes config.json, metrics.jsonl, last.pt / best.pt (by certified area), and
paths/update_XXXX.npz (the deterministic rotation path after each update) to
runs/<run_name>/.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from sofa_env import SofaEnv, replay
from sofa_rl import ActorPolicy, GaussianActor, GaussianActorPolicy, PPOAgent, RolloutBuffer, ValueCritic
from sofa_rl.agents.ppo_agent import summarize_episodes
from sofa_rl.config import TrainConfig

RUNS_DIR = Path("runs")
# half-plane directions for the certified area logged every update
CERT_DIRS = 5


def lr_lambda(schedule: str, n_updates: int):
    if schedule == "linear":
        return lambda step: 1.0 - step / n_updates
    if schedule == "constant":
        return lambda step: 1.0
    raise ValueError(f"unknown lr_schedule {schedule!r}")


def certified_area(corners: np.ndarray, cfg: TrainConfig) -> float:
    """Lower bound on the true area of the continuous motion through `corners`
    (straight-line corner motion between decisions; see sofa_env/certified.py).

    The area at training N overstates the true area by about 1.2 / N, more than the
    differences between good runs, so runs are ranked and checkpointed by this instead.
    """
    return replay(
        corners, max_corner_speed=cfg.env.max_corner_speed, reward_area="certified", cert_dirs=CERT_DIRS
    ).area


def path_certified_area(corners: np.ndarray, env_area: float, cfg: TrainConfig) -> float:
    """Certified area of an evaluated path. With reward_area="certified" the env's own area
    is already a valid lower bound (tighter with substeps or more directions), so use it."""
    if len(corners) < cfg.env.n_steps + 1:  # the sofa reached area 0 and the episode ended early
        return 0.0
    if cfg.env.reward_area == "certified":
        return env_area
    return certified_area(corners, cfg)


def evaluate(policy: ActorPolicy, env: SofaEnv) -> tuple[np.ndarray, float]:
    """Run one episode with the policy's mean action.

    Returns (corners, area): the rotation path c_0, ..., c_N of shape (N + 1, 2)
    and the final sofa area at the env's n_steps.
    """
    obs, _ = env.reset()
    corners = [env.corner.copy()]
    done = False
    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0) # (1, obs_dim)
        with torch.no_grad():
            action = policy.predict(obs_tensor, deterministic=True).squeeze(0).cpu().numpy() # (action_dim,)
        obs, _, terminated, truncated, info = env.step(action)
        corners.append(info["corner"])
        done = terminated or truncated
    return np.array(corners), info["area"]


def save_checkpoint(
    path: Path,
    actor: GaussianActor,
    critic: ValueCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    schedulers: list,
    update: int,
    total_steps: int,
    eval_area: float,
    cert_area: float,
    cfg: TrainConfig,
) -> None:
    """Everything needed to inspect the policy or resume training after `update`."""
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "schedulers": [s.state_dict() for s in schedulers],
            "update": update,
            "total_steps": total_steps,
            "eval_area": eval_area,
            "cert_area": cert_area,
            "config": asdict(cfg),
        },
        path,
    )


def main(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    run_name = cfg.run_name or f"n{cfg.env.n_steps}_seed{cfg.seed}"
    run_dir = RUNS_DIR / run_name
    paths_dir = run_dir / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    env = SofaEnv(**asdict(cfg.env))
    env.reset(seed=cfg.seed)
    eval_env = SofaEnv(**asdict(cfg.env))
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = GaussianActor(obs_dim, action_dim, **asdict(cfg.network)) # network
    policy = GaussianActorPolicy(actor) # policy
    critic = ValueCritic(obs_dim, hidden_dim=cfg.network.hidden_dim)
    if cfg.init_checkpoint is not None:
        checkpoint = torch.load(RUNS_DIR / cfg.init_checkpoint, weights_only=False)
        actor.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["critic"])
        if not cfg.init_keep_log_std:
            with torch.no_grad():
                actor.log_std.fill_(cfg.network.log_std_init)
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
    best_cert_area = -np.inf
    start = time.time()
    with open(run_dir / "metrics.jsonl", "w") as metrics_file:
        for update in range(n_updates):
            episodes = agent.collect_rollout()
            update_stats = agent.update()
            actor_lr = actor_optimizer.param_groups[0]["lr"]

            for scheduler in schedulers:
                scheduler.step()

            corners, eval_area = evaluate(policy, eval_env)
            cert_area = path_certified_area(corners, eval_area, cfg)

            # saved for (sofa_env.replay) without the network: for plots and animation
            np.savez(
                paths_dir / f"update_{update:04d}.npz",
                corners=corners,
                area=eval_area,
                cert_area=cert_area,
                polygon=np.array(eval_env.shape.exterior.coords, dtype=np.float64).reshape(-1, 2),
                n_steps=cfg.env.n_steps,
                update=update,
            )

            episode_stats = summarize_episodes(episodes)
            total_steps += episode_stats["num_steps"]
            row = {
                "update": update,
                "total_steps": total_steps,
                "actor_lr": actor_lr,
                "max_final_area": max(e["final_area"] for e in episodes),
                "eval_area": eval_area,
                "cert_area": cert_area,
                **episode_stats,
                **update_stats,
            }
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()

            checkpoint_args = (
                actor, critic, actor_optimizer, critic_optimizer, schedulers,
                update, total_steps, eval_area, cert_area, cfg,
            )
            save_checkpoint(run_dir / "last.pt", *checkpoint_args)
            if cert_area > best_cert_area:
                best_cert_area = cert_area
                save_checkpoint(run_dir / "best.pt", *checkpoint_args)

            print(
                f"update {update + 1}/{n_updates} | steps {total_steps} "
                f"| area mean {row['mean_final_area']:.4f} max {row['max_final_area']:.4f} "
                f"eval {eval_area:.4f} certified {cert_area:.4f} "
                f"| kl {row['approx_kl']:.4f} | ev {row['explained_variance']:.2f} "
                f"| std {row['policy_std']:.3f} | {time.time() - start:.0f}s"
            )


if __name__ == "__main__":
    main(TrainConfig())
