"""Bezier-parameterized rigid motion for the sofa: x(s), y(s), theta(s).

Rotation is stored as t = tan(theta/2) rather than theta directly, so that
a future exact crossing-certifier (Weierstrass substitution: cos/sin become
rational functions of t) can reuse these same control points with no
conversion step in between training and certification.
"""
from __future__ import annotations

import numpy as np

from .curves import bezier_derivative_points, bezier_eval


class BezierMotion:
    def __init__(self, x_points, y_points, t_points):
        self.x_points = np.asarray(x_points, dtype=np.float64)
        self.y_points = np.asarray(y_points, dtype=np.float64)
        self.t_points = np.asarray(t_points, dtype=np.float64)

    def pose(self, s) -> np.ndarray:
        """Pose(s) = [x, y, theta] at each parameter value in s. Shape (len(s), 3)."""
        s = np.atleast_1d(np.asarray(s, dtype=np.float64))
        x = bezier_eval(self.x_points, s)
        y = bezier_eval(self.y_points, s)
        t = bezier_eval(self.t_points, s)
        theta = 2.0 * np.arctan(t)
        return np.stack([x, y, theta], axis=-1)
        
    # for conservative-advancement, not used by neural-net or environment
    def velocity(self, s) -> np.ndarray:
        """[dx/ds, dy/ds, dt/ds] at each s. Shape (len(s), 3).

        Note the third component is dt/ds (t = tan(theta/2)), not dtheta/ds --
        convert with dtheta/ds = 2/(1+t^2) * dt/ds if the angular rate is needed.
        """
        s = np.atleast_1d(np.asarray(s, dtype=np.float64))
        dx = bezier_eval(bezier_derivative_points(self.x_points), s)
        dy = bezier_eval(bezier_derivative_points(self.y_points), s)
        dt = bezier_eval(bezier_derivative_points(self.t_points), s)
        return np.stack([dx, dy, dt], axis=-1)
