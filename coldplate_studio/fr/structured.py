"""Structured conforming 1D/3D node coordinates for FR hexahedral meshes."""

from __future__ import annotations

import numpy as np

from coldplate_studio.fr.elements import legendre_gauss_lobatto_nodes


def conforming_axis_coords(ne: int, degree: int, x0: float, x1: float) -> np.ndarray:
    """Merge element-local LGL nodes into a conforming 1D axis."""
    if ne < 1:
        raise ValueError("ne must be >= 1")
    ref = legendre_gauss_lobatto_nodes(degree)
    dx = (x1 - x0) / ne
    chunks: list[np.ndarray] = []
    for e in range(ne):
        xl = x0 + e * dx
        xr = x0 + (e + 1) * dx
        chunks.append(0.5 * (xr - xl) * ref + 0.5 * (xr + xl))
    merged = chunks[0]
    for c in chunks[1:]:
        merged = np.concatenate((merged, c[1:]))
    return merged


def structured_fr_grid_3d(
    ne_x: int,
    ne_y: int,
    ne_z: int,
    degree: int,
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Return (x, y, z) node coordinates and element sizes (dx, dy, dz)."""
    x0, x1, y0, y1, z0, z1 = bounds
    x = conforming_axis_coords(ne_x, degree, x0, x1)
    y = conforming_axis_coords(ne_y, degree, y0, y1)
    z = conforming_axis_coords(ne_z, degree, z0, z1)
    dx = (x1 - x0) / ne_x
    dy = (y1 - y0) / ne_y
    dz = (z1 - z0) / ne_z
    return x, y, z, dx, dy, dz
