"""Map mesh entities into FR reference-element data (spec C2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.fr.elements import legendre_gauss_lobatto_nodes


@dataclass
class FRElementMapping1D:
    element_index: int
    x_left: float
    x_right: float
    degree: int
    solution_points_x: np.ndarray
    jacobian_dx_dxi: float

    @property
    def valid(self) -> bool:
        return self.x_right > self.x_left and self.jacobian_dx_dxi > 0.0


def map_uniform_1d_elements(x0: float, x1: float, n_elements: int, degree: int) -> list[FRElementMapping1D]:
    nodes = legendre_gauss_lobatto_nodes(degree)
    mappings: list[FRElementMapping1D] = []
    dx = (x1 - x0) / n_elements
    for e in range(n_elements):
        xl = x0 + e * dx
        xr = x0 + (e + 1) * dx
        x_pts = 0.5 * (xr - xl) * nodes + 0.5 * (xr + xl)
        mappings.append(
            FRElementMapping1D(
                element_index=e,
                x_left=xl,
                x_right=xr,
                degree=degree,
                solution_points_x=x_pts,
                jacobian_dx_dxi=0.5 * (xr - xl),
            )
        )
    return mappings
