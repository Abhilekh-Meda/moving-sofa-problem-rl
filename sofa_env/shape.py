"""Turn raw ordered (x, y) points into the polygon SofaEnv consumes.

Points are assumed to already be in traversal order around the boundary
(e.g. sampled at a fixed parameter grid upstream) -- this only closes the
loop and validates it, it does not infer an ordering from an unordered set.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def polygon_from_points(points) -> Polygon:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an (N, 2) array")
    if len(points) < 3:
        raise ValueError("need at least 3 points to form a polygon")

    polygon = Polygon(points)
    if not polygon.is_valid:
        raise ValueError("points do not form a valid (non-self-intersecting) polygon")
    return polygon
