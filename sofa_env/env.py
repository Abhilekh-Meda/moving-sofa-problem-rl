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

from . import geodesic
from .corridor import build_l_corridor
from .shape import polygon_from_points

RewardFn = Callable[..., float]


class SofaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        shape: Union[Polygon, np.ndarray],
        corridor_width: float = 1.0,
        arm_length: Optional[float] = None,
        goal_depth: Optional[float] = None,
        max_steps: int = 1000,
        max_dxy: float = 0.05,
        max_dtheta: float = 0.05,
        collision_mode: str = "terminate",
        reward_fn: Optional[RewardFn] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if reward_fn is None:
            raise ValueError(
                "reward_fn is required -- SofaEnv has no default reward. "
                "Pass a callable accepting prev_dist, new_dist, prev_pose, "
                "new_pose, collided, success."
            )
        base_shape = (
            shape if isinstance(shape, Polygon) else polygon_from_points(shape)
        )
        if not base_shape.is_valid or base_shape.area <= 0:
            raise ValueError("shape must be a valid polygon with positive area")

        if collision_mode not in ("terminate", "reject"):
            raise ValueError("collision_mode must be 'terminate' or 'reject'")

        # Recentre so the shape's own origin is its centroid: _posed_polygon
        # rotates about (0, 0) in local coordinates
        centroid = base_shape.centroid
        self.base_shape = translate(base_shape, xoff=-centroid.x, yoff=-centroid.y)

        self.corridor_width = corridor_width
        # Default: a width x width corner plus one width-sized run in each arm
        self.arm_length = arm_length if arm_length is not None else 2 * corridor_width
        self.corridor = build_l_corridor(corridor_width, self.arm_length)
        # Goal: fully inside a region deep in the vertical arm, mirroring
        # where the episode starts (deep in the horizontal arm)
        self.goal_depth = goal_depth if goal_depth is not None else self.arm_length / 2
        if not (0 < self.goal_depth < self.arm_length):
            raise ValueError("goal_depth must satisfy 0 < goal_depth < arm_length")
        self.goal_region = box(0.0, -self.arm_length, corridor_width, -self.goal_depth)
        self.max_steps = max_steps
        self.max_dxy = max_dxy
        self.max_dtheta = max_dtheta
        self.collision_mode = collision_mode
        self.reward_fn: RewardFn = reward_fn
        self.render_mode = render_mode

        # normalizer: distance from the default start, not from wherever
        # a given episode begins, so geo_dist_norm means the same thing in
        # every episode regardless of reset options.
        default_start = np.array([-self.arm_length / 2, corridor_width / 2, 0.0])
        self._geo_dist_start = max(self._geodesic_dist(default_start), 1e-6)

        self.action_space = spaces.Box(
            low=np.array([-max_dxy, -max_dxy, -max_dtheta], dtype=np.float32),
            high=np.array([max_dxy, max_dxy, max_dtheta], dtype=np.float32),
        )
        # [x_norm, y_norm, sin(theta), cos(theta), time_left_norm, geo_dist_norm]
        # theta itself stays raw/unbounded internally (periodic, unclipped --
        # see module docstring); sin/cos give the agent a periodic-safe view.
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 2.0, 1.0, 1.0, 1.0, np.inf], dtype=np.float32),
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

    def _geodesic_dist(self, pose: Optional[np.ndarray] = None) -> float:
        pose = self.pose if pose is None else pose
        x, y, _ = pose
        return float(geodesic.corridor_geodesic(np.array([[x, y]]), self.goal_depth)[0])

    def _get_obs(self) -> np.ndarray:
        x, y, theta = self.pose
        time_left_norm = 1.0 - self.step_count / self.max_steps
        geo_dist_norm = self._geodesic_dist() / self._geo_dist_start
        return np.array(
            [
                x / self.arm_length,
                y / self.arm_length,
                np.sin(theta),
                np.cos(theta),
                time_left_norm,
                geo_dist_norm,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        start_x = options.get("start_x", -self.arm_length / 2)
        start_y = options.get("start_y", self.corridor_width / 2)
        start_theta = options.get("start_theta", 0.0)
        self.pose = np.array([start_x, start_y, start_theta], dtype=np.float64)
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
            prev_dist=self._geodesic_dist(prev_pose),
            new_dist=self._geodesic_dist(self.pose),
            prev_pose=prev_pose,
            new_pose=self.pose,
            collided=collided,
            success=success,
        )

        obs = self._get_obs()
        info = {
            "colliding": collided,
            "success": success,
            "geodesic_dist": self._geodesic_dist(),
        }
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
