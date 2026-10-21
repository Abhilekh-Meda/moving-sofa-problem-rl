"""Exact geodesic (shortest in-corridor path) distance to the goal for the L-corridor.

A shortest path inside a polygon is straight except where it bends around a
reflex (inward-pointing) vertex. The L-corridor built by
`corridor.build_l_corridor` has exactly one reflex vertex, the inner corner at
(0, 0), and the goal region is `y <= -goal_depth` in the vertical arm, so the
distance has a closed form:

    x >= 0 (vertical arm and corner square):  max(0, y + goal_depth)
                                              -- straight down to the goal
    x <  0 (horizontal arm):                  hypot(x, y) + goal_depth
                                              -- straight to the corner, then down

The two branches agree along the opening x = 0, y >= 0, so the distance is
continuous everywhere in the corridor. If the corridor geometry ever changes,
this must change with it.
"""
from __future__ import annotations

import numpy as np


def corridor_geodesic(pts: np.ndarray, goal_depth: float) -> np.ndarray:
    """Distance to the goal region for points of shape (P, 2) -> (P,)."""
    x, y = pts[:, 0], pts[:, 1]
    return np.where(
        x >= 0.0,
        np.maximum(0.0, y + goal_depth),
        np.hypot(x, y) + goal_depth,
    )
