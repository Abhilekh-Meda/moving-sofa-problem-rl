"""Bernstein/Bezier primitives used by the motion representation (motion.py).

Shape doesn't need these: a polygon's edges are exact by definition (the
straight line between two vertices *is* the boundary, not an approximation
of one), so raw ordered points go straight to a Polygon in shape.py with no
curve-fitting step. Motion is different -- there's a genuine continuous
path to pin down between sampled poses, which is why it needs an actual
parameterized curve instead.
"""
from __future__ import annotations

from math import comb

import numpy as np


def bernstein_weights(n: int, s: np.ndarray) -> np.ndarray:
    """Degree-n Bernstein basis values at each point in `s`.

    Returns an (len(s), n+1) matrix; row j gives the n+1 basis weights at s[j].
    """
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    i = np.arange(n + 1)
    coeffs = np.array([comb(n, k) for k in i], dtype=np.float64)
    return coeffs * (s[:, None] ** i) * ((1.0 - s[:, None]) ** (n - i))


def bezier_eval(control_points: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Evaluate a 1D Bezier curve with the given control points at parameter(s) s."""
    control_points = np.asarray(control_points, dtype=np.float64)
    n = len(control_points) - 1
    weights = bernstein_weights(n, s)
    return weights @ control_points

# for conservative-advancement, not used by neural-net or environment
def bezier_derivative_points(control_points: np.ndarray) -> np.ndarray:
    """Control points of a Bezier curve's derivative (one degree lower)."""
    control_points = np.asarray(control_points, dtype=np.float64)
    n = len(control_points) - 1
    if n == 0:
        return np.zeros(1)
    return n * (control_points[1:] - control_points[:-1])
