"""Rotation paths of the known sofas, in SofaEnv's convention L_t = x(t) + R_t L.

hammersley_path: x(t) = r (cos 2t - 1, sin 2t) with r = 2/pi; area pi/2 + 2/pi ~ 2.2074.
gerver_path: Gerver's sofa, area ~ 2.21953, from Romik (2016), "Differential
    equations and exact solutions in the moving sofa problem", eq. (25),
    (SOL1)-(SOL5) and Table 1.

Both return the corners c_0, ..., c_N at theta_k = k (pi/2) / N, shape (N + 1, 2),
so they can be passed to replay() or compared with a policy's saved path.
"""
from __future__ import annotations

import numpy as np

HAMMERSLEY_AREA = np.pi / 2 + 2 / np.pi
GERVER_AREA = 2.21953166

# Romik (2016), Table 1
_PHI, _THETA = 0.039177364790083641, 0.681301509382724894
_KAPPA = {
    1: (-0.210322422072688751, 0.25),
    2: (-0.919179292771593322, 0.472406619750805465),
    3: (-0.613763229430251668, 0.889626479003221860),
    4: (-0.308347166088910014, 0.472406619750805465),
    5: (-1.017204036787814585, 0.25),
}
_A1, _A2 = 1.210322422072688751, -0.25
_B1, _B2 = -0.527624598026784624, 0.920258385160637622
_C1, _C2 = 0.626045522848465867, -0.944750803946430751
_D1, _D2 = 1.313022761424232933, -0.525382670414554437
_E1, _E2 = 1.210322422072688751, 0.25


def _thetas(n_steps: int) -> np.ndarray:
    return np.arange(n_steps + 1) * (np.pi / 2) / n_steps


def hammersley_path(n_steps: int) -> np.ndarray:
    t = _thetas(n_steps)
    r = 2 / np.pi
    return np.stack([r * (np.cos(2 * t) - 1), r * np.sin(2 * t)], axis=1)


def _gerver_corner(t: float) -> np.ndarray:
    if t < _PHI:
        v, k = (_A1 * np.cos(t) + _A2 * np.sin(t) - 1, -_A2 * np.cos(t) + _A1 * np.sin(t) - 0.5), 1
    elif t < _THETA:
        v, k = (-0.25 * t * t + _B1 * t + _B2, 0.5 * t - _B1 - 1), 2
    elif t <= np.pi / 2 - _THETA:
        v, k = (_C1 - t, _C2 + t), 3
    elif t <= np.pi / 2 - _PHI:
        v, k = (-0.5 * t + _D1 - 1, -0.25 * t * t + _D1 * t + _D2), 4
    else:
        v, k = (_E1 * np.cos(t) + _E2 * np.sin(t) - 0.5, -_E2 * np.cos(t) + _E1 * np.sin(t) - 1), 5
    c, s = np.cos(t), np.sin(t)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]]) + np.array(_KAPPA[k])


def gerver_path(n_steps: int) -> np.ndarray:
    return np.array([_gerver_corner(t) for t in _thetas(n_steps)])
