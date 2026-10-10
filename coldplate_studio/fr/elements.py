"""Reference-element utilities for 1D Flux Reconstruction (milestone 6 foundation)."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss


def legendre_gauss_lobatto_nodes(degree: int) -> np.ndarray:
    """Return degree+1 LGL nodes on [-1, 1] for polynomial degree `degree`."""
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if degree == 1:
        return np.array([-1.0, 1.0])
    interior, _ = leggauss(degree - 1)
    return np.concatenate(([-1.0], interior, [1.0]))


def lagrange_basis(nodes: np.ndarray, xi: float) -> np.ndarray:
    n = len(nodes)
    out = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            out[i] *= (xi - nodes[j]) / (nodes[i] - nodes[j])
    return out


def _barycentric_weights(nodes: np.ndarray) -> np.ndarray:
    w = np.ones(len(nodes))
    for j, xj in enumerate(nodes):
        for k, xk in enumerate(nodes):
            if k != j:
                w[j] *= xj - xk
    return 1.0 / w


def lagrange_derivative_matrix(nodes: np.ndarray) -> np.ndarray:
    """Differentiation matrix D_ij = dL_j/dxi(xi_i) (barycentric form, Kopriva 2003)."""
    x = nodes
    w = _barycentric_weights(x)
    n = len(x)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i, j] = (w[j] / w[i]) / (x[i] - x[j])
        d[i, i] = -np.sum(d[i, :])
    return d


def radau_right_correction_values(nodes: np.ndarray) -> np.ndarray:
    """Right Radau correction at solution points (DG-equivalent FR branch)."""
    xi_r = 1.0
    n = len(nodes)
    g = np.zeros(n)
    for i in range(n):
        prod = 1.0
        for j in range(n):
            if j == i:
                continue
            prod *= (xi_r - nodes[j]) / (nodes[i] - nodes[j])
        g[i] = prod
    return g


def radau_left_correction_values(nodes: np.ndarray) -> np.ndarray:
    xi_l = -1.0
    n = len(nodes)
    g = np.zeros(n)
    for i in range(n):
        prod = 1.0
        for j in range(n):
            if j == i:
                continue
            prod *= (xi_l - nodes[j]) / (nodes[i] - nodes[j])
        g[i] = prod
    return g
