"""Geometry for the classic L-shaped moving-sofa corridor."""
from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import unary_union


def build_l_corridor(width: float = 1.0, arm_length: float = 10.0) -> Polygon:
    """Build an L-shaped hallway of the given width.

    One arm runs in -x, the other in -y, joined by a `width` x `width`
    corner square at the origin:

        horizontal arm: x in [-arm_length, width], y in [0, width]
        vertical arm:   x in [0, width],            y in [-arm_length, width]
    """
    horizontal = Polygon(
        [
            (-arm_length, 0.0),
            (width, 0.0),
            (width, width),
            (-arm_length, width),
        ]
    )
    vertical = Polygon(
        [
            (0.0, -arm_length),
            (width, -arm_length),
            (width, width),
            (0.0, width),
        ]
    )
    return unary_union([horizontal, vertical])
