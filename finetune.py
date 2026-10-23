"""Stage 2: fine-tune the stage-1 base policy on the certified-area reward.

The base policy (train.py, runs/base/) is trained on the area at N = 200 sampled
hallway angles, which overstates the true area of the continuous motion by about
0.003. Here the same policy keeps training with reward_area="certified" and 4
substeps per step: the reward is a lower bound on the true area, so the agent
optimises the area it can guarantee. best.pt is chosen by that certified area.

    python finetune.py                                   # from runs/base/best.pt -> runs/finetune/
    python finetune.py --init base2/best.pt --run-name finetune2
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from sofa_rl.config import EnvConfig, OptimConfig, PPOConfig
from train import BASE_CONFIG, run_new

FINETUNE_CONFIG = replace(
    BASE_CONFIG,
    env=EnvConfig(n_steps=200, max_corner_speed=2.0, substeps=4, reward_area="certified", cert_dirs=5),
    ppo=replace(BASE_CONFIG.ppo, entropy_coef=0.0),
    optim=OptimConfig(actor_lr=1e-4, critic_lr=1e-3, lr_schedule="linear"),
    total_steps=500_000,
    run_name="finetune",
    init_checkpoint="base/best.pt",
    init_keep_log_std=True,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default=FINETUNE_CONFIG.init_checkpoint,
                        help="stage-1 checkpoint to start from, relative to runs/")
    parser.add_argument("--run-name", default=FINETUNE_CONFIG.run_name, help="output folder under runs/")
    args = parser.parse_args()
    run_new(replace(FINETUNE_CONFIG, init_checkpoint=args.init, run_name=args.run_name))
