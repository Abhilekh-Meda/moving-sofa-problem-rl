"""Gymnasium-style environment: an RL agent moves a fixed polygon through
the corridor via per-step rigid-motion actions.

The shape is given up front (any valid polygon); the motion is not -- the
agent supplies a (dx, dy, dtheta) delta each step. The only goal the env
defines is geometric: get the shape from deep in the horizontal arm to deep
in the vertical arm without ever leaving the corridor. There is no target
rotation angle, no assumption that theta increases monotonically, and no
clipping of state -- the agent is free to rotate however much, however
often, in whatever direction it wants; what makes a path good or bad is
entirely up to the (swappable) reward function, not the env.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box

from .corridor import build_l_corridor
from .shape import polygon_from_points
from .rewards import default_reward_fn

RewardFn = Callable[..., float]


class SofaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        shape: Union[Polygon, np.ndarray],
        corridor_width: float = 1.0,
        arm_length: float = 10.0,
        goal_depth: Optional[float] = None,
        max_steps: int = 300,
        max_dxy: float = 0.05,
        max_dtheta: float = 0.05,
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

        if collision_mode not in ("terminate", "reject"):
            raise ValueError("collision_mode must be 'terminate' or 'reject'")

        self.corridor_width = corridor_width
        self.arm_length = arm_length
        self.corridor = build_l_corridor(corridor_width, arm_length)
        # Goal: fully inside a region deep in the vertical arm, mirroring
        # where the episode starts (deep in the horizontal arm). Purely
        # positional -- says nothing about what theta should be to get there.
        self.goal_depth = goal_depth if goal_depth is not None else arm_length / 2
        self.goal_region = box(0.0, -arm_length, corridor_width, -self.goal_depth)
        self.max_steps = max_steps
        self.max_dxy = max_dxy
        self.max_dtheta = max_dtheta
        self.collision_mode = collision_mode
        self.reward_fn = reward_fn or default_reward_fn
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=np.array([-max_dxy, -max_dxy, -max_dtheta], dtype=np.float32),
            high=np.array([max_dxy, max_dxy, max_dtheta], dtype=np.float32),
        )
        # Raw pose, unclipped and unbounded -- theta can wind past +/-pi or
        # go negative; nothing here tells the agent what "forward" means.
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
        )

        self.pose = np.zeros(3, dtype=np.float64)  # x, y, theta
        self.step_count = 0
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

    def _reached_goal(self, poly: Polygon) -> bool:
        return self.goal_region.contains(poly)

    def _get_obs(self) -> np.ndarray:
        return self.pose.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        start_x = options.get("start_x", -self.arm_length / 2)
        start_y = options.get("start_y", self.corridor_width / 2)
        self.pose = np.array([start_x, start_y, 0.0], dtype=np.float64)
        self.step_count = 0

        obs = self._get_obs()
        info = {"colliding": not self._is_contained(self._posed_polygon())}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).clip(
            self.action_space.low, self.action_space.high
        )
        prev_pose = self.pose.copy()

        new_pose = prev_pose + action
        new_poly = self._posed_polygon(new_pose)
        collided = not self._is_contained(new_poly)

        if collided and self.collision_mode == "reject":
            new_pose = prev_pose
            new_poly = self._posed_polygon(new_pose)

        self.pose = new_pose
        self.step_count += 1

        success = (not collided) and self._reached_goal(new_poly)

        terminated = success or (collided and self.collision_mode == "terminate")
        truncated = self.step_count >= self.max_steps

        reward = self.reward_fn(
            prev_pose=prev_pose,
            new_pose=self.pose,
            collided=collided,
            success=success,
        )

        obs = self._get_obs()
        info = {"colliding": collided, "success": success}
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
