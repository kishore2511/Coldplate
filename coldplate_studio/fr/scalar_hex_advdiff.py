"""Scalar advection–diffusion on structured hex FR grids (Phase 4.2 entry)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.fr.deriv3d import derivative_along_axis, laplacian_fr
from coldplate_studio.fr.structured import structured_fr_grid_3d


@dataclass
class HexAdvDiffConfig:
    ne_x: int = 4
    ne_y: int = 3
    ne_z: int = 2
    degree: int = 2
    bounds: tuple[float, float, float, float, float, float] = (0.0, 1.0, 0.0, 0.2, 0.0, 0.1)
    diffusivity: float = 0.05
    velocity_m_s: tuple[float, float, float] = (0.15, 0.0, 0.0)


@dataclass
class HexAdvDiffState:
    scalar: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    time: float = 0.0


@dataclass
class HexAdvDiffMMSReport:
    mesh_level: int
    l2_error: float
    ne_x: int
    ne_y: int
    ne_z: int


class ScalarHexAdvDiff3D:
    """
    Transient scalar transport ∂φ/∂t = α∇²φ − v·∇φ + S.

    Uses the same FR derivative operators as ``ScalarThermalVOF3D`` (no VOF).
    """

    def __init__(self, config: HexAdvDiffConfig) -> None:
        self.config = config
        x, y, z, dx, dy, dz = structured_fr_grid_3d(
            config.ne_x, config.ne_y, config.ne_z, config.degree, config.bounds
        )
        self.dx, self.dy, self.dz = dx, dy, dz
        self.state = HexAdvDiffState(
            scalar=np.zeros((len(x), len(y), len(z))),
            x=x,
            y=y,
            z=z,
        )

    def set_scalar(self, field: np.ndarray) -> None:
        if field.shape != self.state.scalar.shape:
            raise ValueError("scalar field shape mismatch")
        self.state.scalar = np.asarray(field, dtype=float)

    def apply_dirichlet_scalar(self, value: float = 0.0) -> None:
        phi = self.state.scalar
        phi[0, :, :] = value
        phi[-1, :, :] = value
        phi[:, 0, :] = value
        phi[:, -1, :] = value
        phi[:, :, 0] = value
        phi[:, :, -1] = value

    def _advection(self, scalar: np.ndarray) -> np.ndarray:
        cfg = self.config
        vx, vy, vz = cfg.velocity_m_s
        adv = np.zeros_like(scalar)
        if abs(vx) > 0:
            adv += vx * derivative_along_axis(
                scalar, 0, cfg.ne_x, cfg.degree, self.dx, advection_speed=vx
            )
        if abs(vy) > 0:
            adv += vy * derivative_along_axis(
                scalar, 1, cfg.ne_y, cfg.degree, self.dy, advection_speed=vy
            )
        if abs(vz) > 0:
            adv += vz * derivative_along_axis(
                scalar, 2, cfg.ne_z, cfg.degree, self.dz, advection_speed=vz
            )
        return adv

    def rhs(self, source: np.ndarray | None = None) -> np.ndarray:
        cfg = self.config
        phi = self.state.scalar
        lap = laplacian_fr(
            phi,
            (cfg.ne_x, cfg.ne_y, cfg.ne_z),
            cfg.degree,
            (self.dx, self.dy, self.dz),
        )
        s = source if source is not None else np.zeros_like(phi)
        return cfg.diffusivity * lap - self._advection(phi) + s

    def _interior_slices(self) -> tuple[slice, slice, slice]:
        return slice(1, -1), slice(1, -1), slice(1, -1)

    def _rk4_on_interior(self, phi0: np.ndarray, dt: float, source: np.ndarray | None) -> np.ndarray:
        sl = self._interior_slices()

        def increment(phi: np.ndarray) -> np.ndarray:
            self.state.scalar = phi
            self.apply_dirichlet_scalar(0.0)
            return self.rhs(source)

        k1 = increment(phi0)
        k2 = increment(phi0 + 0.5 * dt * k1)
        k3 = increment(phi0 + 0.5 * dt * k2)
        k4 = increment(phi0 + dt * k3)
        out = phi0.copy()
        out[sl] = phi0[sl] + (dt / 6.0) * (k1[sl] + 2 * k2[sl] + 2 * k3[sl] + k4[sl])
        return out

    def step_rk4(self, dt: float, source: np.ndarray | None = None) -> None:
        self.state.scalar = self._rk4_on_interior(self.state.scalar, dt, source)
        self.apply_dirichlet_scalar(0.0)
        self.state.time += dt

    def volume_mean_l2_error(self, exact: np.ndarray) -> float:
        err_sq = float(np.mean((self.state.scalar - exact) ** 2))
        return float(np.sqrt(err_sq))


def manufactured_scalar_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    lx: float,
    ly: float,
    lz: float,
) -> np.ndarray:
    """Smooth periodic-like mode; zero on x=0,L and y=0,H faces (Dirichlet-friendly)."""
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.sin(np.pi * X / lx) * np.sin(np.pi * Y / ly) * np.sin(np.pi * Z / lz)


def manufactured_scalar_source(
    solver: ScalarHexAdvDiff3D,
    exact: np.ndarray,
) -> np.ndarray:
    """S = −(α∇²φ − v·∇φ) so steady state satisfies α∇²φ − v·∇φ + S = 0."""
    cfg = solver.config
    solver.set_scalar(exact)
    lap = laplacian_fr(
        exact,
        (cfg.ne_x, cfg.ne_y, cfg.ne_z),
        cfg.degree,
        (solver.dx, solver.dy, solver.dz),
    )
    adv = solver._advection(exact)
    return -(cfg.diffusivity * lap - adv)


def laplacian_discretization_error_l2(
    solver: ScalarHexAdvDiff3D,
    exact: np.ndarray,
    *,
    lx: float,
    ly: float,
    lz: float,
) -> float:
    """‖∇²_h φ − ∇²φ‖₂ on interior FR nodes (analytic ∇² for the sin product)."""
    cfg = solver.config
    lap_h = laplacian_fr(
        exact,
        (cfg.ne_x, cfg.ne_y, cfg.ne_z),
        cfg.degree,
        (solver.dx, solver.dy, solver.dz),
    )
    x, y, z = solver.state.x, solver.state.y, solver.state.z
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    lap_ex = -(
        (np.pi / lx) ** 2 + (np.pi / ly) ** 2 + (np.pi / lz) ** 2
    ) * np.sin(np.pi * X / lx) * np.sin(np.pi * Y / ly) * np.sin(np.pi * Z / lz)
    sl = solver._interior_slices()
    diff = lap_h[sl] - lap_ex[sl]
    return float(np.sqrt(np.mean(diff**2)))


def run_hex_advdiff_mms_study(
    *,
    ne_levels: tuple[int, ...] = (3, 6, 12),
    degree: int = 2,
) -> list[HexAdvDiffMMSReport]:
    """
    Mesh-refinement audit for FR spatial operators on a smooth sin product field.

    Reports ‖∇²_h φ − ∇²φ‖₂ (interior); expect O(h) trend as h→0 for this FR laplacian.
    """
    bounds = (0.0, 1.0, 0.0, 0.2, 0.0, 0.1)
    lx, ly, lz = bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
    reports: list[HexAdvDiffMMSReport] = []
    for ne in ne_levels:
        cfg = HexAdvDiffConfig(
            ne_x=ne,
            ne_y=max(2, ne // 2),
            ne_z=2,
            degree=degree,
            bounds=bounds,
        )
        solver = ScalarHexAdvDiff3D(cfg)
        exact = manufactured_scalar_field(solver.state.x, solver.state.y, solver.state.z, lx=lx, ly=ly, lz=lz)
        err = laplacian_discretization_error_l2(solver, exact, lx=lx, ly=ly, lz=lz)
        reports.append(
            HexAdvDiffMMSReport(mesh_level=ne, l2_error=err, ne_x=ne, ne_y=cfg.ne_y, ne_z=cfg.ne_z)
        )
    return reports


def estimate_l2_convergence_order(reports: list[HexAdvDiffMMSReport]) -> float | None:
    """Crude order from last two levels: log(e1/e2)/log(h1/h2)."""
    if len(reports) < 2:
        return None
    r0, r1 = reports[-2], reports[-1]
    if r0.l2_error <= 0 or r1.l2_error <= 0:
        return None
    h0 = 1.0 / r0.ne_x
    h1 = 1.0 / r1.ne_x
    return float(np.log(r0.l2_error / r1.l2_error) / np.log(h0 / h1))
