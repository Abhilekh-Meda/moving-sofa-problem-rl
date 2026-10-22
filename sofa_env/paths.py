"""Reproduce a sofa from a saved rotation path.

A rotation path is the sequence of corner positions c_0, ..., c_N, an array of
shape (N + 1, 2) with c_0 = (0, 0). theta_k = k * (pi/2) / N is fixed by the
step index and SofaEnv is deterministic, so the path alone determines every
intermediate shape S_k and the final sofa.
"""
from __future__ import annotations

import numpy as np

from .env import SofaEnv

ACTION_TOL = 1e-9


def replay(corners: np.ndarray, max_corner_speed: float = 2.0, **env_kwargs) -> SofaEnv:
    """Step a fresh SofaEnv through `corners` and return it after the last step.

    N is len(corners) - 1. Each step's action is (c_k - c_{k-1}) / max_step,
    because self.corner is updated by self.corner = self.corner + action * self.max_step in SofaEnv.step,
    so to replay a given corner path, actions must be scaled by 1 / max_step.
    which must lie in [-1, 1]; a path that needs larger actions raises instead
    of being silently clipped. Extra keyword arguments go to SofaEnv.
    """
    corners = np.asarray(corners, dtype=np.float64)
    if corners.ndim != 2 or corners.shape[1] != 2 or len(corners) < 2:
        raise ValueError(f"corners must have shape (N + 1, 2) with N >= 1, got {corners.shape}")
    if not np.allclose(corners[0], 0.0):
        raise ValueError(f"corners[0] must be (0, 0), got {corners[0]}")

    env = SofaEnv(n_steps=len(corners) - 1, max_corner_speed=max_corner_speed, **env_kwargs)
    env.reset()

    actions = np.diff(corners, axis=0) / env.max_step  # (N, 2)
    worst = np.abs(actions).max()
    if worst > 1.0 + ACTION_TOL:
        raise ValueError(
            f"path needs a per-step action of {worst:.4f} > 1; "
            f"increase max_corner_speed (currently {max_corner_speed})"
        )

    for action in actions:
        env.step(np.clip(action, -1.0, 1.0))
    return env


def resweep(corners: np.ndarray, factor: int, max_corner_speed: float = 2.0, **env_kwargs) -> SofaEnv:
    """Replay `corners` at `factor` times the angular resolution.

    The path is linearly interpolated in theta from N + 1 to N * factor + 1
    points, so each coarse step is split into `factor` equal fine steps. The
    corner's speed per radian is unchanged, so the actions stay in [-1, 1].
    Sampling the hallway at more angles removes material that survives between
    the coarse angles; use this area, not the training-N area, to score a path.
    """
    corners = np.asarray(corners, dtype=np.float64)
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    n_steps = len(corners) - 1
    coarse = np.arange(n_steps + 1)                                 # step index k
    fine = np.arange(n_steps * factor + 1) / factor                 # fractional k
    fine_corners = np.stack(
        [np.interp(fine, coarse, corners[:, 0]), np.interp(fine, coarse, corners[:, 1])],
        axis=1,
    )  # (N * factor + 1, 2)
    return replay(fine_corners, max_corner_speed=max_corner_speed, **env_kwargs)
