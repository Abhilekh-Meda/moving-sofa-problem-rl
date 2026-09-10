"""Smoke test: a fixed square shape, random per-step actions moving it.

The square here is just a placeholder -- SofaEnv doesn't care how a shape
was produced. This exercises the reset/step/render loop with an untrained
(random) policy, standing in for wherever the real RL agent's actions
will come from later.
"""
import os

import numpy as np

from sofa_env import SofaEnv, polygon_from_points, record_rollout, save_gif


def main():
    width = 1.0
    side = width * 0.8
    square = np.array(
        [
            [-side / 2, -side / 2],
            [side / 2, -side / 2],
            [side / 2, side / 2],
            [-side / 2, side / 2],
        ]
    )
    shape = polygon_from_points(square)

    env = SofaEnv(
        shape=shape,
        corridor_width=width,
        collision_mode="reject",
        max_steps=300,
    )

    frames = record_rollout(env, policy=None, seed=0)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "random_rollout.gif")
    save_gif(frames, out_path)
    print(f"Saved rollout ({len(frames)} frames) to {out_path}")


if __name__ == "__main__":
    main()
