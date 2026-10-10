"""Dirichlet elimination for P1 Galerkin systems."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def apply_dirichlet_elimination(
    a: sparse.spmatrix,
    b: np.ndarray,
    node_values: list[tuple[int, float]],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Zero rows/cols for constrained nodes after moving contributions to the RHS."""
    a = a.tolil()
    b = b.copy()
    for i, val in node_values:
        b -= a[:, i].toarray().ravel() * val
        a[:, i] = 0
        a[i, :] = 0
        a[i, i] = 1.0
        b[i] = val
    return a.tocsr(), b
