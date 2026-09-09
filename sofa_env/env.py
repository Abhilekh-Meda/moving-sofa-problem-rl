"""Gymnasium-style playback environment for a candidate (shape, motion) pair.

The env does not choose a motion -- it's handed a fully-specified
`BezierMotion` up front. Stepping just advances the shared curve parameter
`s` from 0 to 1 and reports whether the sofa stayed inside the corridor at
that pose. This keeps the reset/step/render loop for watching rollouts
without pretending there's a decision being made at each step.

Shape and motion are constructed by separate modules (`shape.py`,
`motion.py`) before reaching the env -- the env only ever sees a finished
Polygon and a finished BezierMotion, so swapping either representation
later never touches this file.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon

from .corridor import build_l_corridor
from .motion import BezierMotion
from .shape import polygon_from_points
from .rewards import default_reward_fn

RewardFn = Callable[..., float]


class SofaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        shape: Union[Polygon, np.ndarray],
        motion: BezierMotion,
        corridor_width: float = 1.0,
        arm_length: float = 10.0,
        n_steps: int = 200,
        collision_mode: str = "terminate",
        reward_fn: Optional[RewardFn] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.base_shape = (
            shape if isinstance(shape, Polygon) else polygon_from_points(shape)
        )
        if not self.base_shape.is_valid or self.base_shape.area <= 0:
            raise ValueError("shape must be a valid polygon with positive area")

        if collision_mode not in ("terminate", "flag"):
            raise ValueError("collision_mode must be 'terminate' or 'flag'")

        self.motion = motion
        self.corridor_width = corridor_width
        self.arm_length = arm_length
        self.corridor = build_l_corridor(corridor_width, arm_length)
        self.n_steps = n_steps
        self.collision_mode = collision_mode
        self.reward_fn = reward_fn or default_reward_fn
        self.render_mode = render_mode

        # Observation: [x, y, theta, s]
        obs_high = np.array([np.inf, np.inf, np.pi, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high)
        # Nothing chooses actions here -- the motion is already fully specified.
        # Kept only so code expecting a gymnasium action_space doesn't choke;
        # step() ignores whatever is passed in.
        self.action_space = spaces.Discrete(1)

        self.s = 0.0
        self.step_count = 0
        self.pose = np.zeros(3, dtype=np.float64)
        self._fig = None
        self._ax = None

    def _posed_polygon(self, pose: Optional[np.ndarray] = None) -> Polygon:
        pose = self.pose if pose is None else pose
        x, y, theta = pose
        poly = rotate(self.base_shape, theta, origin=(0, 0), use_radians=True)
        poly = translate(poly, xoff=x, yoff=y)
        return poly

    # TODO: only checks the sampled poses, not the continuous sweep between
    # them -- swap in the conservative-advancement / Bernstein-sign-test
    # certifier once it exists, for anything meant to back a real claim.
    def _is_contained(self, poly: Polygon) -> bool:
        return self.corridor.buffer(1e-9).contains(poly)

    def _get_obs(self) -> np.ndarray:
        x, y, theta = self.pose
        return np.array([x, y, theta, self.s], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = 0.0
        self.step_count = 0
        self.pose = self.motion.pose(self.s)[0]

        obs = self._get_obs()
        info = {"colliding": not self._is_contained(self._posed_polygon())}
        return obs, info

    def step(self, action=None):
        prev_progress = self.s
        self.step_count += 1
        self.s = min(1.0, self.step_count / self.n_steps)
        self.pose = self.motion.pose(self.s)[0]

        poly = self._posed_polygon()
        collided = not self._is_contained(poly)
        success = self.s >= 1.0 and not collided

        terminated = success or (collided and self.collision_mode == "terminate")
        truncated = self.step_count >= self.n_steps

        reward = self.reward_fn(
            prev_progress=prev_progress,
            new_progress=self.s,
            collided=collided,
            success=success,
        )

        obs = self._get_obs()
        info = {"colliding": collided, "progress": self.s, "success": success}
        return obs, reward, terminated, truncated, info

    def render(self, save_path: Optional[str] = None):
        from .rendering import render_frame

        return render_frame(self, save_path=save_path)

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None


gym.register(id="Sofa-v0", entry_point="sofa_env.env:SofaEnv")
