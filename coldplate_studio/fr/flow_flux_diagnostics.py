"""Consistent volumetric flow rate and mass diagnostics (m³/s)."""

from __future__ import annotations

import numpy as np

from coldplate_studio.mesh.integral_model import IntegralMesh


def cross_section_flow_rate_yz_plane(
    mesh: IntegralMesh,
    velocity: np.ndarray,
    x_plane: float,
    width: float,
    x_tol: float = 1e-5,
) -> float:
    """
    ∫ u·e_x dA on plane x=x_plane by trapezoidal quadrature in y (assumes u_x uniform in z).
    """
    nodes = mesh.nodes
    mask = np.abs(nodes[:, 0] - x_plane) <= x_tol
    if not np.any(mask):
        return 0.0
    y = nodes[mask, 1]
    ux = velocity[mask, 0]
    order = np.argsort(y)
    y_s = y[order]
    ux_s = ux[order]
    if y_s.size < 2:
        return float(ux_s[0] * width * (y_s.max() - y_s.min() + 1e-12))
    return float(np.trapezoid(ux_s, y_s) * width)


def analytical_parabolic_flow_rate(height: float, width: float, g: float, nu: float) -> float:
    """Q = G W H³ / (12 ν) for parallel-plate Poiseuille (developed, no-slip on y walls)."""
    return g * width * (height**3) / (12.0 * nu)
