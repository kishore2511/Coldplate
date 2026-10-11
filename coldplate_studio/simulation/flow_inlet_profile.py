"""Inlet velocity profiles for integral channel."""

from __future__ import annotations

import numpy as np

from coldplate_studio.mesh.integral_model import IntegralMesh


def parabolic_inlet_ux(
    mesh: IntegralMesh,
    inlet_nodes: np.ndarray,
    *,
    width_y: float = 0.2,
    u_max: float = 0.5,
) -> dict[int, tuple[float, float, float]]:
    """
    No-slip-compatible parabolic profile in y on x=0 inlet plane: u_x = u_max * 4 y (W-y) / W².
    """
    values: dict[int, tuple[float, float, float]] = {}
    y = mesh.nodes[:, 1]
    for i in inlet_nodes:
        i = int(i)
        yi = float(y[i])
        ux = u_max * 4.0 * yi * (width_y - yi) / max(width_y**2, 1e-30)
        values[i] = (ux, 0.0, 0.0)
    return values
