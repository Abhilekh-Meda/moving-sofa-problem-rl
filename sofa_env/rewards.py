"""Default reward for SofaEnv. Pass a different callable as `reward_fn=` to override.

Any reward_fn must accept these keyword arguments and return a float:
    prev_progress, new_progress, collided, success
"""
from __future__ import annotations


def default_reward_fn(
    prev_progress: float,
    new_progress: float,
    collided: bool,
    success: bool,
    progress_weight: float = 1.0,
    collision_penalty: float = 1.0,
    success_bonus: float = 5.0,
    step_penalty: float = 0.0,
) -> float:
    reward = progress_weight * (new_progress - prev_progress)
    reward -= step_penalty
    if collided:
        reward -= collision_penalty
    if success:
        reward += success_bonus
    return reward
