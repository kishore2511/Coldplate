"""3D scalar thermal + VOF on the integral Gmsh mesh (in-house FR, not PyFR)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from coldplate_studio.fr.solver3d import ConservationReport, PhaseChangeConfig
from coldplate_studio.fr.tet_p1 import assemble_p1_operators
from coldplate_studio.mesh.integral_model import IntegralModel


@dataclass
class IntegralSolverConfig:
    thermal_diffusivity: float = 2.0e-5
    velocity_m_s: tuple[float, float, float] = (0.05, 0.0, 0.0)
    phase_change: PhaseChangeConfig = field(default_factory=PhaseChangeConfig)
    dirichlet_temperature_k: float = 300.0


@dataclass
class IntegralSolverState:
    temperature: np.ndarray  # nodal, length n_nodes
    vof: np.ndarray
    time: float = 0.0


class IntegralScalarThermalVOF3D:
    """
    Transient transport on the same tetrahedral mesh produced by the integral pipeline.

    Spatial discretization: P1 simplex (nodal FR/DG order p=1). High-order FR hex operators
    are added separately on supported element types in `fr/hex` modules.
    """

    def __init__(self, model: IntegralModel, config: IntegralSolverConfig) -> None:
        self.model = model
        self.config = config
        pc = config.phase_change
        self._m, self._k, self._c, self._tet_vol = assemble_p1_operators(
            model.mesh,
            config.thermal_diffusivity,
            config.velocity_m_s,
            pc.rho_kg_m3 * pc.cp_j_kg_k,
        )
        n = model.mesh.nodes.shape[0]
        self.state = IntegralSolverState(temperature=np.full(n, config.dirichlet_temperature_k), vof=np.zeros(n))
        self._boundary = model.mesh.boundary_fluid_nodes
        self._fluid_node_mask = self._fluid_nodes_mask()

    def _fluid_nodes_mask(self) -> np.ndarray:
        mask = np.zeros(self.model.mesh.nodes.shape[0], dtype=bool)
        for tet in self.model.mesh.fluid_tets:
            mask[tet] = True
        return mask

    def set_initial_temperature(self, nodal: np.ndarray) -> None:
        if nodal.shape[0] != self.state.temperature.shape[0]:
            raise ValueError("nodal temperature length mismatch")
        self.state.temperature = np.asarray(nodal, dtype=float)

    def set_initial_vof(self, nodal: np.ndarray) -> None:
        if nodal.shape[0] != self.state.vof.shape[0]:
            raise ValueError("nodal vof length mismatch")
        self.state.vof = np.clip(nodal, 0.0, 1.0)

    def _apply_dirichlet(self, vector: np.ndarray) -> np.ndarray:
        vector = vector.copy()
        vector[self._boundary] = self.config.dirichlet_temperature_k
        return vector

    def rhs(self) -> tuple[np.ndarray, np.ndarray, float]:
        t = self.state.temperature
        a = self.state.vof
        inv_m = np.zeros_like(t)
        diag = self._m.diagonal()
        np.divide(1.0, diag, out=inv_m, where=diag > 0)
        dT = (-self._k @ t - self._c @ t) * inv_m
        dalpha = (-self._c @ a) * inv_m
        dT = np.where(self._fluid_node_mask, dT, 0.0)
        dalpha = np.where(self._fluid_node_mask, dalpha, 0.0)
        pc_power = 0.0
        if self.config.phase_change.enabled:
            pc = self.config.phase_change
            iface = a * (1.0 - a)
            m_dot = pc.transfer_coefficient * np.maximum(t - pc.saturation_temperature, 0.0) * iface
            m_dot = np.where(self._fluid_node_mask, m_dot, 0.0)
            dalpha -= m_dot / pc.rho_kg_m3
            latent = pc.latent_heat_j_kg * m_dot
            dT -= latent / (pc.rho_kg_m3 * pc.cp_j_kg_k)
            pc_power = float(np.sum(latent * self._tet_volume_per_node()))
        return dT, dalpha, pc_power

    def _tet_volume_per_node(self) -> np.ndarray:
        contrib = np.zeros_like(self.state.temperature)
        counts = np.zeros_like(self.state.temperature)
        for e, tet in enumerate(self.model.mesh.fluid_tets):
            v = self._tet_vol[e] / 4.0
            for n in tet:
                contrib[n] += v
                counts[n] += 1
        counts = np.maximum(counts, 1.0)
        return contrib / counts

    def step_rk4(self, dt: float) -> ConservationReport:
        t0 = self.state.temperature
        a0 = self.state.vof

        def inc(T, A):
            self.state.temperature = T
            self.state.vof = A
            return self.rhs()

        kT1, ka1, _ = inc(t0, a0)
        kT2, ka2, _ = inc(t0 + 0.5 * dt * kT1, a0 + 0.5 * dt * ka1)
        kT3, ka3, _ = inc(t0 + 0.5 * dt * kT2, a0 + 0.5 * dt * ka2)
        kT4, ka4, _ = inc(t0 + dt * kT3, a0 + dt * ka3)
        self.state.temperature = self._apply_dirichlet(t0 + (dt / 6.0) * (kT1 + 2 * kT2 + 2 * kT3 + kT4))
        self.state.vof = np.clip(a0 + (dt / 6.0) * (ka1 + 2 * ka2 + 2 * ka3 + ka4), 0.0, 1.0)
        self.state.vof = np.where(self._fluid_node_mask, self.state.vof, 0.0)
        self.state.time += dt
        _, _, pc_power = self.rhs()
        return self.conservation_report(pc_power)

    def conservation_report(self, phase_change_power: float = 0.0) -> ConservationReport:
        pc = self.config.phase_change
        nodal_vol = self._tet_volume_per_node()
        alpha = self.state.vof
        mass = float(np.sum(alpha * pc.rho_kg_m3 * nodal_vol))
        energy = float(np.sum(self.state.temperature * pc.cp_j_kg_k * pc.rho_kg_m3 * nodal_vol))
        fluid_alpha = alpha[self._fluid_node_mask]
        return ConservationReport(
            liquid_mass_kg=mass,
            total_thermal_energy_j=energy,
            phase_change_power_w=phase_change_power,
            alpha_min=float(fluid_alpha.min()) if fluid_alpha.size else 0.0,
            alpha_max=float(fluid_alpha.max()) if fluid_alpha.size else 0.0,
        )
