"""Grid-based geodesic (corner-aware) distance-to-goal field for the L-corridor.

Straight-line distance cuts through the wall for points in different arms;
this instead floods a fine grid outward from the goal cells through only the
in-corridor cells, via multi-source Dijkstra, so the resulting distance
respects the corner. The corridor and goal are fixed for an env's lifetime,
so the field is built once and queried cheaply (O(1) per lookup) every step.

The in-corridor / in-goal tests replicate the two-rectangle union that
`corridor.build_l_corridor` builds -- if that geometry ever changes, this
must change with it.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GeoField:
    minx: float
    miny: float
    cell_size: float
    nx: int
    ny: int
    dist: np.ndarray  # (nx, ny), world-unit geodesic distance to goal; inf where unreachable/outside


def _in_corridor(xs: np.ndarray, ys: np.ndarray, width: float, arm_length: float) -> np.ndarray:
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    horizontal = (X >= -arm_length) & (X <= width) & (Y >= 0.0) & (Y <= width)
    vertical = (X >= 0.0) & (X <= width) & (Y >= -arm_length) & (Y <= width)
    return horizontal | vertical


def _in_goal(xs: np.ndarray, ys: np.ndarray, width: float, arm_length: float, goal_depth: float) -> np.ndarray:
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return (X >= 0.0) & (X <= width) & (Y >= -arm_length) & (Y <= -goal_depth)


def build_geodesic_field(
    width: float,
    arm_length: float,
    goal_depth: float,
    cell_size: float,
) -> GeoField:
    minx, maxx = -arm_length, width
    miny, maxy = -arm_length, width

    nx = max(int(math.ceil((maxx - minx) / cell_size)) + 1, 2)
    ny = max(int(math.ceil((maxy - miny) / cell_size)) + 1, 2)

    xs = minx + (np.arange(nx) + 0.5) * cell_size
    ys = miny + (np.arange(ny) + 0.5) * cell_size

    inside = _in_corridor(xs, ys, width, arm_length)
    goal = _in_goal(xs, ys, width, arm_length, goal_depth) & inside

    dist = np.full((nx, ny), np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []
    for i, j in zip(*np.nonzero(goal)):
        dist[i, j] = 0.0
        heapq.heappush(heap, (0.0, int(i), int(j)))

    diag = cell_size * math.sqrt(2.0)
    neighbors = [
        (-1, 0, cell_size), (1, 0, cell_size),
        (0, -1, cell_size), (0, 1, cell_size),
        (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag),
    ]

    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        for di, dj, cost in neighbors:
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny and inside[ni, nj]:
                nd = d + cost
                if nd < dist[ni, nj]:
                    dist[ni, nj] = nd
                    heapq.heappush(heap, (nd, ni, nj))

    return GeoField(minx=minx, miny=miny, cell_size=cell_size, nx=nx, ny=ny, dist=dist)


def query(field: GeoField, x: float, y: float) -> float:
    i = int(round((x - field.minx) / field.cell_size - 0.5))
    j = int(round((y - field.miny) / field.cell_size - 0.5))
    i = min(max(i, 0), field.nx - 1)
    j = min(max(j, 0), field.ny - 1)
    return float(field.dist[i, j])
