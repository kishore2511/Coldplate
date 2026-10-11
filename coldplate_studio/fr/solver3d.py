"""3D FR scalar thermal transport and passive VOF on structured hexahedral grids."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from coldplate_studio.fr.deriv3d import derivative_along_axis, laplacian_fr
from coldplate_studio.fr.structured import structured_fr_grid_3d


@dataclass
class PhaseChangeConfig:
    enabled: bool = False
    saturation_temperature: float = 373.15
    latent_heat_j_kg: float = 2.26e6
    transfer_coefficient: float = 0.05
    rho_kg_m3: float = 1000.0
    cp_j_kg_k: float = 4180.0


@dataclass
class Solver3DConfig:
    ne_x: int = 3
    ne_y: int = 2
    ne_z: int = 2
    degree: int = 2
    bounds: tuple[float, float, float, float, float, float] = (0.0, 1.0, 0.0, 0.2, 0.0, 0.1)
    thermal_diffusivity: float = 1.0e-5
    velocity_m_s: tuple[float, float, float] = (0.1, 0.0, 0.0)
    phase_change: PhaseChangeConfig = field(default_factory=PhaseChangeConfig)


@dataclass
class ConservationReport:
    liquid_mass_kg: float
    total_thermal_energy_j: float
    phase_change_power_w: float
    alpha_min: float
    alpha_max: float


@dataclass
class Solver3DState:
    temperature: np.ndarray
    vof: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    time: float = 0.0


class ScalarThermalVOF3D:
    """Transient heat equation + passive VOF with optional phenomenological phase change."""

    def __init__(self, config: Solver3DConfig) -> None:
        self.config = config
        x, y, z, dx, dy, dz = structured_fr_grid_3d(
            config.ne_x, config.ne_y, config.ne_z, config.degree, config.bounds
        )
        self.dx, self.dy, self.dz = dx, dy, dz
        shape = (len(x), len(y), len(z))
        self.state = Solver3DState(
            temperature=np.zeros(shape),
            vof=np.zeros(shape),
            x=x,
            y=y,
            z=z,
        )
        self._cell_volume = dx * dy * dz

    def set_initial_temperature(self, field: np.ndarray) -> None:
        if field.shape != self.state.temperature.shape:
            raise ValueError("temperature field shape mismatch")
        self.state.temperature = np.asarray(field, dtype=float)

    def set_initial_vof(self, field: np.ndarray) -> None:
        if field.shape != self.state.vof.shape:
            raise ValueError("vof field shape mismatch")
        self.state.vof = np.clip(field, 0.0, 1.0)

    def apply_dirichlet_temperature(self, value: float = 0.0) -> None:
        T = self.state.temperature
        T[0, :, :] = value
        T[-1, :, :] = value
        T[:, 0, :] = value
        T[:, -1, :] = value
        T[:, :, 0] = value
        T[:, :, -1] = value

    def _advection(self, scalar: np.ndarray) -> np.ndarray:
        vx, vy, vz = self.config.velocity_m_s
        adv = np.zeros_like(scalar)
        if abs(vx) > 0:
            adv += vx * derivative_along_axis(scalar, 0, self.config.ne_x, self.config.degree, self.dx, advection_speed=vx)
        if abs(vy) > 0:
            adv += vy * derivative_along_axis(scalar, 1, self.config.ne_y, self.config.degree, self.dy, advection_speed=vy)
        if abs(vz) > 0:
            adv += vz * derivative_along_axis(scalar, 2, self.config.ne_z, self.config.degree, self.dz, advection_speed=vz)
        return adv

    def rhs(self) -> tuple[np.ndarray, np.ndarray, float]:
        cfg = self.config
        T = self.state.temperature
        alpha = self.state.vof
        lap = laplacian_fr(
            T,
            (cfg.ne_x, cfg.ne_y, cfg.ne_z),
            cfg.degree,
            (self.dx, self.dy, self.dz),
        )
        dT = cfg.thermal_diffusivity * lap - self._advection(T)
        dalpha = -self._advection(alpha)
        pc_power = 0.0
        if cfg.phase_change.enabled:
            pc = cfg.phase_change
            iface_mask = alpha * (1.0 - alpha)
            m_dot = pc.transfer_coefficient * np.maximum(T - pc.saturation_temperature, 0.0) * iface_mask
            dalpha -= m_dot / pc.rho_kg_m3
            latent = pc.latent_heat_j_kg * m_dot
            dT -= latent / (pc.rho_kg_m3 * pc.cp_j_kg_k)
            pc_power = float(np.sum(latent) * self._cell_volume)
        return dT, dalpha, pc_power

    def step_rk4(self, dt: float) -> ConservationReport:
        T0 = self.state.temperature
        a0 = self.state.vof

        def increment(T, a):
            self.state.temperature = T
            self.state.vof = a
            dT, da, _ = self.rhs()
            return dT, da

        kT1, ka1 = increment(T0, a0)
        kT2, ka2 = increment(T0 + 0.5 * dt * kT1, a0 + 0.5 * dt * ka1)
        kT3, ka3 = increment(T0 + 0.5 * dt * kT2, a0 + 0.5 * dt * ka2)
        kT4, ka4 = increment(T0 + dt * kT3, a0 + dt * ka3)
        self.state.temperature = T0 + (dt / 6.0) * (kT1 + 2 * kT2 + 2 * kT3 + kT4)
        self.state.vof = np.clip(a0 + (dt / 6.0) * (ka1 + 2 * ka2 + 2 * ka3 + ka4), 0.0, 1.0)
        self.apply_dirichlet_temperature(0.0)
        self.state.time += dt
        _, _, pc_power = self.rhs()
        return self.conservation_report(pc_power)

    def conservation_report(self, phase_change_power: float = 0.0) -> ConservationReport:
        pc = self.config.phase_change
        alpha = self.state.vof
        mass = float(np.sum(alpha * pc.rho_kg_m3) * self._cell_volume)
        energy = float(np.sum(self.state.temperature * pc.cp_j_kg_k * pc.rho_kg_m3) * self._cell_volume)
        return ConservationReport(
            liquid_mass_kg=mass,
            total_thermal_energy_j=energy,
            phase_change_power_w=phase_change_power,
            alpha_min=float(alpha.min()),
            alpha_max=float(alpha.max()),
        )
