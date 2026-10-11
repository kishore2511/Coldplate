"""1D scalar advection–diffusion FR solver (CPU reference path)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.fr.operators import element_residual_fr1d, upwind_interface_flux
from coldplate_studio.fr.elements import legendre_gauss_lobatto_nodes


@dataclass
class ScalarFR1DConfig:
    x0: float = 0.0
    x1: float = 1.0
    n_elements: int = 4
    degree: int = 3
    advection_speed: float = 1.0
    diffusivity: float = 0.0


@dataclass
class ScalarFR1DState:
    x: np.ndarray
    u: np.ndarray
    degree: int
    n_elements: int


class ScalarFR1DSolver:
    """Tensor-product 1D FR discretization for du/dt + a du/dx = nu d2u/dx2."""

    def __init__(self, config: ScalarFR1DConfig) -> None:
        self.config = config
        if config.n_elements < 1:
            raise ValueError("n_elements must be >= 1")
        if config.degree < 1:
            raise ValueError("degree must be >= 1")
        npts = config.n_elements * (config.degree + 1)
        self._nodes = legendre_gauss_lobatto_nodes(config.degree)
        xs = []
        for e in range(config.n_elements):
            x_left = config.x0 + (config.x1 - config.x0) * e / config.n_elements
            x_right = config.x0 + (config.x1 - config.x0) * (e + 1) / config.n_elements
            xe = 0.5 * (x_right - x_left) * self._nodes + 0.5 * (x_right + x_left)
            xs.append(xe)
        self.state = ScalarFR1DState(
            x=np.concatenate(xs),
            u=np.zeros(npts),
            degree=config.degree,
            n_elements=config.n_elements,
        )
        self._dx = (config.x1 - config.x0) / config.n_elements

    def set_solution(self, values: np.ndarray) -> None:
        if values.shape != self.state.u.shape:
            raise ValueError("solution size mismatch")
        self.state.u = np.asarray(values, dtype=float)

    def _element_slice(self, e: int) -> slice:
        p1 = self.config.degree + 1
        start = e * p1
        return slice(start, start + p1)

    def rhs(self, u: np.ndarray) -> np.ndarray:
        cfg = self.config
        p1 = cfg.degree + 1
        du = np.zeros_like(u)
        a = cfg.advection_speed
        nu = cfg.diffusivity
        for e in range(cfg.n_elements):
            sl = self._element_slice(e)
            ue = u[sl]
            fe = a * ue
            if e == 0:
                f_l = upwind_interface_flux(ue[0], ue[0], a)
            else:
                u_prev = u[self._element_slice(e - 1)]
                f_l = upwind_interface_flux(u_prev[-1], ue[0], a)
            if e == cfg.n_elements - 1:
                f_r = upwind_interface_flux(ue[-1], ue[-1], a)
            else:
                u_next = u[self._element_slice(e + 1)]
                f_r = upwind_interface_flux(ue[-1], u_next[0], a)
            du[sl] = element_residual_fr1d(ue, fe, cfg.degree, self._dx, f_l, f_r)
            if nu > 0.0:
                # BR2-style simplified interior penalty for diffusion (milestone verification).
                du[sl] += nu * self._diffusion_element(ue)
        return du

    def _diffusion_element(self, ue: np.ndarray) -> np.ndarray:
        # Second-derivative via compact FD on LGL nodes (verification-focused; 2D/3D BR2 later).
        from coldplate_studio.fr.elements import lagrange_derivative_matrix

        d = lagrange_derivative_matrix(self._nodes)
        d2 = d @ (d @ ue)
        return d2 / (self._dx**2)

    def step_rk4(self, dt: float) -> None:
        u0 = self.state.u
        k1 = self.rhs(u0)
        k2 = self.rhs(u0 + 0.5 * dt * k1)
        k3 = self.rhs(u0 + 0.5 * dt * k2)
        k4 = self.rhs(u0 + dt * k3)
        self.state.u = u0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
