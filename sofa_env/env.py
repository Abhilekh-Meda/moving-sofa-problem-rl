"""Gymnasium-style environment for the moving-sofa problem in the rotation-path
formulation.

Everything happens in the sofa's own frame: the sofa stays still and the
hallway moves around it. The episode starts from a block of material, the
strip 0 <= y <= 1 that survives the slide-in. Each step the hallway turns by a
fixed angle and the agent moves the hallway's inner corner; whatever part of
the shape lies outside the hallway is cut away. After the last step the shape
is cut once more by the exit strip (the slide-out). What remains is the sofa.

    theta_k = k * (pi/2) / N
    L_k     = c_k + R(theta_k) L
    S_k     = largest connected piece of (S_{k-1} ∩ L_k)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from shapely import affinity
from shapely.geometry import Polygon, box

from .certified import certified_cut
from .corridor import build_l_corridor


class SofaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_steps: int = 200,
        max_corner_speed: float = 2.0,
        block_length: float = 5.0,
        arm_length: float = 10.0,
        substeps: int = 1,
        reward_area: str = "sampled",
        cert_dirs: int = 5,
    ):
        """
        n_steps: number of rotation steps N; theta advances by (pi/2) / N per step.
        max_corner_speed: bound on the corner's movement per radian of rotation;
            the per-step bound on each of dx, dy is max_corner_speed * dtheta.
        block_length: the starting block is x in [-block_length, 1], y in [0, 1].
        arm_length: length of each hallway arm, a finite stand-in for infinite
            arms; must exceed the shape's reach from the corner.
        substeps: cuts per step. Each action moves the corner in a straight line
            while theta advances, and the hallway cuts the shape at `substeps`
            evenly spaced points of that motion, so the area (and reward) is
            measured at N * substeps angles while the agent still makes N decisions.
        reward_area: "sampled" cuts with the hallway at the sampled positions only, which
            overestimates the true area of the continuous motion; "certified" keeps only
            points proven to stay in the hallway for the whole motion (see certified.py),
            a lower bound on it. The reward and info["area"] use this area; with
            "certified", info["sampled_area"] also reports the sampled one.
        cert_dirs: half-plane directions for the certified inner-corner cut.
        """
        super().__init__()
        self.n_steps = n_steps
        self.dtheta = (np.pi / 2) / n_steps
        self.max_step = max_corner_speed * self.dtheta # where dtheta is just dt (time step)
        self.block_length = block_length
        self.arm_length = arm_length
        self.substeps = substeps
        if reward_area not in ("sampled", "certified"):
            raise ValueError(f"reward_area must be 'sampled' or 'certified', got {reward_area!r}")
        self.reward_area = reward_area
        self.cert_dirs = cert_dirs

        self.corridor = build_l_corridor(width=1.0, arm_length=arm_length)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # [k / N, c_x, c_y]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)

        self.k = 0
        self.theta = 0.0
        self.corner = np.zeros(2)
        self.shape = Polygon()
        self.area = 0.0
        self.initial_area = 0.0

    def _moving_hallway(self, theta: float, corner: np.ndarray) -> Polygon:
        """The hallway in the sofa frame: self.corridor rotated counterclockwise
        by theta about its inner corner, then translated so the corner is at
        `corner`."""
        rotated = affinity.rotate(self.corridor, theta, origin=(0.0, 0.0), use_radians=True)
        return affinity.translate(rotated, corner[0], corner[1])

    def _exit_strip(self, corner: np.ndarray) -> Polygon:
        """The vertical arm's infinite strip in the sofa frame at theta = pi/2:
        the band corner_y <= y <= corner_y + 1."""
        far = 2.0 * self.arm_length
        return box(-far, corner[1], far, corner[1] + 1.0)

    @staticmethod
    def _largest_piece(shape) -> Polygon:
        """Largest connected component of a shapely geometry, or an empty
        Polygon if there is none."""
        pieces = [g for g in getattr(shape, "geoms", [shape]) if isinstance(g, Polygon) and not g.is_empty]
        if not pieces:
            return Polygon()
        return max(pieces, key=lambda g: g.area)

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self.k / self.n_steps,
                self.corner[0],
                self.corner[1],
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.k = 0
        self.theta = 0.0
        self.corner = np.zeros(2)
        # slide-in: whatever survives the horizontal arm lies in the strip 0 <= y <= 1;
        # the cut by the hallway at theta = 0 caps it at x = 1; the far end is arbitrary
        self.shape = box(-self.block_length, 0.0, 1.0, 1.0) # starting shape
        self.sampled_shape = self.shape
        self.area = self.shape.area
        self.initial_area = self.area
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        start_corner, start_theta = self.corner, self.theta
        self.corner = self.corner + action * self.max_step # translation
        self.k += 1
        self.theta = self.k * self.dtheta # rotation

        poses = [
            (start_theta + (j / self.substeps) * (self.theta - start_theta),
             start_corner + (j / self.substeps) * (self.corner - start_corner))
            for j in range(self.substeps + 1)
        ]
        # intersect the step's hallway positions with each other first (small polygons),
        # then cut the shape once
        hallway = self._moving_hallway(*poses[-1])
        for theta, corner in poses[1:-1]:
            hallway = hallway.intersection(self._moving_hallway(theta, corner))
        last = self.k == self.n_steps

        sampled = self.sampled_shape.intersection(hallway)
        if last:
            sampled = sampled.intersection(self._exit_strip(self.corner))
        self.sampled_shape = self._largest_piece(sampled)

        if self.reward_area == "sampled":
            self.shape = self.sampled_shape
        else:
            shape = self.shape
            for (theta0, c0), (theta1, c1) in zip(poses[:-1], poses[1:]):
                shape = self._largest_piece(certified_cut(shape, theta0, c0, theta1, c1, self.cert_dirs))
            if last:
                shape = shape.intersection(self._exit_strip(self.corner))
            self.shape = self._largest_piece(shape)

        new_area = self.shape.area
        reward = (new_area - self.area) / self.initial_area
        self.area = new_area

        terminated = last or self.area == 0.0
        truncated = False
        info = {"area": self.area, "theta": self.theta, "corner": self.corner.copy()}
        if self.reward_area == "certified":
            info["sampled_area"] = self.sampled_shape.area
        return self._get_obs(), reward, terminated, truncated, info


gym.register(id="Sofa-v0", entry_point="sofa_env.env:SofaEnv")
