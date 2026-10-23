"""Certified cut: the part of a shape that provably stays in the hallway for the whole
continuous motion between two poses, not only at the sampled angles.

Motion over one step, t in [0, 1]: theta(t) = theta0 + t dtheta, c(t) = c0 + t dc (the
straight-line corner motion SofaEnv and replay() use). In the hallway's own frame a sofa
point p is at q(t) = R(-theta(t)) (p - c(t)), and the hallway is {q_x <= 1, q_y <= 1}
minus the inner block {q_x < 0, q_y < 0}.

Outer walls. For a fixed unit vector u, f(t) = (R(theta(t)) u) . (p - c(t)) has
|f''| <= dtheta^2 |p - c| + 2 dtheta |dc|. A C^2 function that is <= m at both ends of
[0, 1] exceeds m in between by at most max|f''| / 8. So with
    delta = (dtheta^2 rho + 2 dtheta |dc|) / 8,   rho >= max over the shape of |p - c0| + |dc|,
keeping only points with q_x <= 1 - delta and q_y <= 1 - delta at both poses keeps them
inside both outer walls for every t.

Inner block. q is outside {q_x < 0, q_y < 0} whenever u . q >= 0 for some u in the first
quadrant. For each of n_dirs fixed u, a point with u . q >= delta at both poses (same
second-derivative bound) has u . q(t) >= 0 for all t; the union over u of those
two-pose half-plane intersections is kept.

Every point kept therefore stays in the hallway for all t, so the area left after
cutting every step (and the exit strip) is a lower bound on the true area of the
motion. More directions or splitting a step into substeps tightens it.
"""
from __future__ import annotations

from typing import List

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

_BIG = 30.0


def _place(geom, theta: float, corner: np.ndarray):
    """Hallway-frame geometry -> sofa frame: rotate by theta about the corner, move the corner to `corner`."""
    return affinity.translate(affinity.rotate(geom, theta, origin=(0, 0), use_radians=True), corner[0], corner[1])


def _directions(n_dirs: int) -> List[float]:
    return list(np.linspace(0.0, np.pi / 2, n_dirs))


def certified_cut(shape: Polygon, theta0: float, c0: np.ndarray, theta1: float, c1: np.ndarray,
                  n_dirs: int = 5):
    """Points of `shape` that stay in the hallway for the straight-line motion
    (theta0, c0) -> (theta1, c1). Returns a shapely geometry (possibly several pieces)."""
    if shape.is_empty:
        return shape
    dtheta = abs(theta1 - theta0)
    dc = float(np.linalg.norm(np.asarray(c1) - np.asarray(c0)))
    hull = np.asarray(shape.convex_hull.exterior.coords) if shape.area > 0 else np.asarray(c0)[None]
    rho = float(np.max(np.linalg.norm(hull - np.asarray(c0), axis=1))) + dc
    delta = (dtheta ** 2 * rho + 2 * dtheta * dc) / 8.0

    outer = box(-_BIG, -_BIG, 1.0 - delta, 1.0 - delta)
    shape = shape.intersection(_place(outer, theta0, c0)).intersection(_place(outer, theta1, c1))

    safe = []
    for angle in _directions(n_dirs):
        half_plane = affinity.rotate(box(delta, -_BIG, _BIG, _BIG), angle, origin=(0, 0), use_radians=True)
        safe.append(_place(half_plane, theta0, c0).intersection(_place(half_plane, theta1, c1)))
    return shape.intersection(unary_union(safe))
