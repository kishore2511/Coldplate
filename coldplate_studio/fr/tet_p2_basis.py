"""Quadratic (P2) shape functions on linear tetrahedra (subparametric Taylor–Hood velocity)."""

from __future__ import annotations

import numpy as np

from coldplate_studio.fr.tet_p1 import _tet_gradients

# Local DOF order: vertices 0..3, edges (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
TET_P2_EDGE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def tet_quadrature_4pt() -> tuple[np.ndarray, np.ndarray]:
    """Four-point rule on reference tet (barycentric coordinates, equal weights summing to 1)."""
    a = 0.58541020
    b = 0.13819660
    pts = np.array(
        [
            [a, b, b, b],
            [b, a, b, b],
            [b, b, a, b],
            [b, b, b, a],
        ],
        dtype=float,
    )
    w = np.full(4, 1.0 / 24.0)
    return pts, w


def p2_shape_values(L: np.ndarray) -> np.ndarray:
    """L: (4,) barycentric; returns (10,) P2 basis."""
    L0, L1, L2, L3 = L
    return np.array(
        [
            L0 * (2.0 * L0 - 1.0),
            L1 * (2.0 * L1 - 1.0),
            L2 * (2.0 * L2 - 1.0),
            L3 * (2.0 * L3 - 1.0),
            4.0 * L0 * L1,
            4.0 * L0 * L2,
            4.0 * L0 * L3,
            4.0 * L1 * L2,
            4.0 * L1 * L3,
            4.0 * L2 * L3,
        ],
        dtype=float,
    )


def p2_shape_gradients(L: np.ndarray, vertex_grads: np.ndarray) -> np.ndarray:
    """
    vertex_grads: (4,3) rows are ∇L_i on the linear tet.
    Returns (10, 3) physical gradients of P2 basis at barycentric point L.
    """
    L0, L1, L2, L3 = L
    dNdL = np.zeros((10, 4))
    dNdL[0, 0] = 4.0 * L0 - 1.0
    dNdL[1, 1] = 4.0 * L1 - 1.0
    dNdL[2, 2] = 4.0 * L2 - 1.0
    dNdL[3, 3] = 4.0 * L3 - 1.0
    dNdL[4, 0] = 4.0 * L1
    dNdL[4, 1] = 4.0 * L0
    dNdL[5, 0] = 4.0 * L2
    dNdL[5, 2] = 4.0 * L0
    dNdL[6, 0] = 4.0 * L3
    dNdL[6, 3] = 4.0 * L0
    dNdL[7, 1] = 4.0 * L2
    dNdL[7, 2] = 4.0 * L1
    dNdL[8, 1] = 4.0 * L3
    dNdL[8, 3] = 4.0 * L1
    dNdL[9, 2] = 4.0 * L3
    dNdL[9, 3] = 4.0 * L2
    return dNdL @ vertex_grads


def p1_pressure_values(L: np.ndarray) -> np.ndarray:
    return L.copy()


def p1_pressure_gradients(vertex_grads: np.ndarray) -> np.ndarray:
    return vertex_grads.copy()


def linear_tet_volume_and_grads(nodes4: np.ndarray) -> tuple[float, np.ndarray]:
    return _tet_gradients(nodes4)
