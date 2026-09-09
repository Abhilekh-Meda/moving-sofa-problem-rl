"""Smoke test: play back an arbitrary (untrained) shape + Bezier motion pair.

Both the square and the motion's control points here are hand-picked
placeholders -- SofaEnv doesn't care how either was produced. There's no
expectation this particular pair is valid; this just exercises the
construct-shape / construct-motion / play-rollout pipeline end to end.
"""
import os

import numpy as np

from sofa_env import SofaEnv, BezierMotion, polygon_from_points, record_rollout, save_gif


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

    # x, y: start deep in the horizontal arm, sweep toward the corner.
    # t = tan(theta/2): 0 -> 1 takes theta from 0 -> pi/2.
    motion = BezierMotion(
        x_points=[-5.0, -2.0, 0.5],
        y_points=[0.5, 0.5, -2.0],
        t_points=[0.0, 0.0, 1.0],
    )

    env = SofaEnv(
        shape=shape,
        motion=motion,
        corridor_width=width,
        collision_mode="flag",
        n_steps=200,
    )

    frames = record_rollout(env, seed=0)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "random_rollout.gif")
    save_gif(frames, out_path)
    print(f"Saved rollout ({len(frames)} frames) to {out_path}")


if __name__ == "__main__":
    main()
