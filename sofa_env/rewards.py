"""Default reward for SofaEnv. Pass a different callable as `reward_fn=` to
override -- the env makes no assumptions about what "progress" means, so
any shaping (distance-to-goal, theta-based, etc.) belongs here, not in env.py.

Any reward_fn must accept these keyword arguments and return a float:
    prev_pose, new_pose, collided, success
(prev_pose/new_pose are (x, y, theta) arrays; this default ignores them --
it's intentionally unshaped, a placeholder for your own reward logic.)
"""
from __future__ import annotations

import numpy as np


def default_reward_fn(
    prev_pose: np.ndarray,
    new_pose: np.ndarray,
    collided: bool,
    success: bool,
    step_penalty: float = 0.01,
    collision_penalty: float = 1.0,
    success_bonus: float = 10.0,
) -> float:
    reward = -step_penalty
    if collided:
        reward -= collision_penalty
    if success:
        reward += success_bonus
    return reward
