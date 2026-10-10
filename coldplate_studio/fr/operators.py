"""1D Flux Reconstruction spatial operators."""

from __future__ import annotations

import numpy as np

from coldplate_studio.fr.elements import (
    lagrange_derivative_matrix,
    legendre_gauss_lobatto_nodes,
    radau_left_correction_values,
    radau_right_correction_values,
)


def element_residual_fr1d(
    u: np.ndarray,
    flux: np.ndarray,
    degree: int,
    dx: float,
    f_star_left: float,
    f_star_right: float,
) -> np.ndarray:
    """
    Semi-discrete FR residual du/dt = -dF/dx on one element (Huynh 2007 / Radau FR = nodal DG).

    `u` and `flux` are element-local values at LGL solution points; `flux` is the physical flux F(u).
    """
    nodes = legendre_gauss_lobatto_nodes(degree)
    d = lagrange_derivative_matrix(nodes)
    g_l = radau_left_correction_values(nodes)
    g_r = radau_right_correction_values(nodes)
    f_left = flux[0]
    f_right = flux[-1]
    # Reference element xi in [-1,1] maps to physical length dx: d/dx = (2/dx) d/dxi.
    dfdx = (d @ flux + g_l * (f_star_left - f_left) + g_r * (f_star_right - f_right)) * (2.0 / dx)
    return -dfdx


def upwind_interface_flux(u_left: float, u_right: float, speed: float) -> float:
    """Linear advection flux a*u with upwind choice at interfaces."""
    if speed >= 0.0:
        return speed * u_left
    return speed * u_right
