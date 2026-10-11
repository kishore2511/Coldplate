"""3D FR spatial derivatives via tensor-product 1D operators."""

from __future__ import annotations

import numpy as np

from coldplate_studio.fr.operators import element_residual_fr1d, upwind_interface_flux


def derivative_along_axis(
    u: np.ndarray,
    axis: int,
    ne: int,
    degree: int,
    elem_length: float,
    *,
    advection_speed: float = 0.0,
    symmetric: bool = False,
) -> np.ndarray:
    """
    Compute ∂u/∂x_axis with FR 1D operators on a structured conforming grid.

    When `symmetric` is True, interface states use arithmetic averages (diffusion gradients).
    Otherwise upwind fluxes use `advection_speed`.
    """
    p = degree
    u_m = np.moveaxis(u, axis, 0)
    rest_shape = u_m.shape[1:]
    expected = ne * p + 1
    if u_m.shape[0] != expected:
        raise ValueError(f"axis length {u_m.shape[0]} != expected {expected} for ne={ne}, p={p}")
    out_m = np.zeros_like(u_m)
    a = advection_speed
    for e in range(ne):
        sl = slice(e * p, e * p + p + 1)
        for idx in np.ndindex(rest_shape):
            ue = u_m[sl, *idx]
            fe = ue if symmetric else a * ue
            if e == 0:
                f_l = ue[0] if symmetric else upwind_interface_flux(ue[0], ue[0], a)
            else:
                left = u_m[e * p - 1, *idx]
                f_l = 0.5 * (left + ue[0]) if symmetric else upwind_interface_flux(left, ue[0], a)
            if e == ne - 1:
                f_r = ue[-1] if symmetric else upwind_interface_flux(ue[-1], ue[-1], a)
            else:
                right = u_m[(e + 1) * p, *idx]
                f_r = 0.5 * (ue[-1] + right) if symmetric else upwind_interface_flux(ue[-1], right, a)
            residual = (
                -element_residual_fr1d(ue, ue, degree, elem_length, f_l, f_r)
                if symmetric
                else -element_residual_fr1d(ue, fe, degree, elem_length, f_l, f_r) / max(abs(a), 1e-30)
            )
            out_m[sl, *idx] = residual
    return np.moveaxis(out_m, 0, axis)


def laplacian_fr(
    u: np.ndarray,
    ne: tuple[int, int, int],
    degree: int,
    lengths: tuple[float, float, float],
) -> np.ndarray:
    """∇²u via successive symmetric FR first derivatives."""
    ne_x, ne_y, ne_z = ne
    dx, dy, dz = lengths
    gx = derivative_along_axis(u, 0, ne_x, degree, dx, symmetric=True)
    gy = derivative_along_axis(u, 1, ne_y, degree, dy, symmetric=True)
    gz = derivative_along_axis(u, 2, ne_z, degree, dz, symmetric=True)
    d2x = derivative_along_axis(gx, 0, ne_x, degree, dx, symmetric=True)
    d2y = derivative_along_axis(gy, 1, ne_y, degree, dy, symmetric=True)
    d2z = derivative_along_axis(gz, 2, ne_z, degree, dz, symmetric=True)
    return d2x + d2y + d2z
