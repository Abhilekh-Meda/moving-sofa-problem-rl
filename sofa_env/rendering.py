"""Matplotlib rendering for SofaEnv: single frames and full rollouts."""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np


def _corridor_polys(corridor):
    if corridor.geom_type == "Polygon":
        return [corridor]
    return list(corridor.geoms)


def render_frame(env, save_path: Optional[str] = None):
    """Draw the env's current pose inside the corridor.

    Returns an (H, W, 3) uint8 array when `env.render_mode == "rgb_array"`.
    Otherwise draws to a persistent window ("human" mode) and returns None.
    """
    import matplotlib.pyplot as plt

    if env.render_mode == "rgb_array":
        fig, ax = plt.subplots(figsize=(5, 5))
        own_fig = True
    else:
        if env._fig is None:
            env._fig, env._ax = plt.subplots(figsize=(5, 5))
            plt.ion()
            plt.show()
        fig, ax = env._fig, env._ax
        own_fig = False

    ax.clear()
    for poly in _corridor_polys(env.corridor):
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color="#dbe4ee", edgecolor="#33475b", linewidth=1.0, zorder=0)

    sofa = env._posed_polygon()
    colliding = not env._is_contained(sofa)
    sx, sy = sofa.exterior.xy
    ax.fill(sx, sy, color="#e05a5a" if colliding else "#4c8bf5", alpha=0.85, zorder=2)

    L, w = env.arm_length, env.corridor_width
    ax.set_xlim(-L * 0.55, w * 1.5)
    ax.set_ylim(-L * 0.55, w * 1.5)
    ax.set_aspect("equal")
    x, y, theta = env.pose
    geo_dist = env._geodesic_shape_dist()
    ax.set_title(
        f"x={x:.2f} y={y:.2f} theta={theta:.2f} rad geo={geo_dist:.2f}"
        + ("  COLLIDING" if colliding else "")
    )

    if save_path:
        fig.savefig(save_path)

    if env.render_mode == "rgb_array":
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[..., :3].copy()
        plt.close(fig)
        return frame

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)
    return None


def record_rollout(
    env,
    policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    reset_options: Optional[dict] = None,
) -> List[np.ndarray]:
    """Run one episode and collect rgb_array frames for later playback/saving.

    `policy(obs) -> action`; defaults to random actions from env.action_space.
    """
    prev_mode = env.render_mode
    env.render_mode = "rgb_array"
    try:
        frames = []
        obs, info = env.reset(seed=seed, options=reset_options)
        frames.append(env.render())

        done = False
        steps = 0
        limit = max_steps or env.max_steps
        while not done and steps < limit:
            action = policy(obs) if policy is not None else env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            done = terminated or truncated
            steps += 1
        return frames
    finally:
        env.render_mode = prev_mode


def save_gif(frames: List[np.ndarray], path: str, fps: int = 20) -> None:
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, fps=fps)
