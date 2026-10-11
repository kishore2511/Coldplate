"""Conjugate heat transfer: coupled fluid advection-diffusion + solid conduction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from coldplate_studio.fr.dirichlet_p1 import apply_dirichlet_elimination
from coldplate_studio.fr.thermal_p1 import (
    assemble_advection,
    assemble_region_lumped_mass,
    assemble_region_stiffness,
    interface_heat_flux,
    region_masks,
)
from coldplate_studio.fr.thermal_p2_advection import assemble_p2_velocity_advection
from coldplate_studio.mesh.integral_model import IntegralMesh


@dataclass
class ThermalMaterialProperties:
    rho_cp_fluid: float = 4180.0 * 1000.0  # water-like
    k_fluid: float = 0.6
    rho_cp_solid: float = 2700.0 * 900.0
    k_solid: float = 200.0


@dataclass
class ThermalBoundaryConditions:
    fluid_inlet_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    fluid_inlet_temperature_k: float = 350.0
    solid_outer_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    solid_outer_temperature_k: float = 300.0
    solid_volumetric_heat_w_m3: float = 0.0


@dataclass
class CHTState:
    temperature: np.ndarray
    time: float = 0.0


@dataclass
class CHTDiagnostics:
    total_energy_j: float
    fluid_energy_j: float
    solid_energy_j: float
    interface_flux_fluid_w: float
    interface_flux_solid_w: float
    interface_flux_imbalance: float


class ConjugateHeatTransferSolver:
    """
    Single nodal temperature field on conforming integral mesh (shared FSI nodes).
    Semi-implicit: (M/dt + K + C(u)) T^{n+1} = M/dt T^n with Dirichlet BC.
    """

    def __init__(
        self,
        mesh: IntegralMesh,
        velocity: np.ndarray,
        props: ThermalMaterialProperties,
        bcs: ThermalBoundaryConditions,
        *,
        p2_velocity_field=None,
    ) -> None:
        self.mesh = mesh
        self.velocity = velocity
        self.p2_velocity_field = p2_velocity_field
        self.props = props
        self.bcs = bcs
        self.fluid_mask, self.solid_mask = region_masks(mesh)
        n = mesh.nodes.shape[0]
        m_f = assemble_region_lumped_mass(mesh, mesh.fluid_tets, props.rho_cp_fluid)
        m_s = assemble_region_lumped_mass(mesh, mesh.solid_tets, props.rho_cp_solid)
        self._lumped = m_f + m_s
        k_f = assemble_region_stiffness(mesh, mesh.fluid_tets, props.k_fluid)
        k_s = assemble_region_stiffness(mesh, mesh.solid_tets, props.k_solid)
        self._k = k_f + k_s
        if p2_velocity_field is not None:
            self._c = assemble_p2_velocity_advection(mesh, p2_velocity_field, props.rho_cp_fluid)
        else:
            self._c = assemble_advection(mesh, velocity, props.rho_cp_fluid)
        self.state = CHTState(temperature=np.full(n, bcs.solid_outer_temperature_k))

    def _dirichlet_pairs(self) -> list[tuple[int, float]]:
        pairs: list[tuple[int, float]] = []
        for idx in self.bcs.fluid_inlet_nodes:
            pairs.append((int(idx), self.bcs.fluid_inlet_temperature_k))
        for idx in self.bcs.solid_outer_nodes:
            pairs.append((int(idx), self.bcs.solid_outer_temperature_k))
        return pairs

    def step(self, dt: float) -> CHTDiagnostics:
        m = sparse.diags(self._lumped, format="csr")
        a = (m / dt + self._k + self._c).tolil()
        b = m @ self.state.temperature / dt
        if self.bcs.solid_volumetric_heat_w_m3 != 0.0:
            vol_lump = assemble_region_lumped_mass(self.mesh, self.mesh.solid_tets, 1.0)
            b = b + self.bcs.solid_volumetric_heat_w_m3 * vol_lump
        a, b = apply_dirichlet_elimination(a.tocsr(), b, self._dirichlet_pairs())
        t_new = spsolve(a, b)
        self.state.temperature = t_new
        self.state.time += dt
        return self.diagnostics()

    def diagnostics(self) -> CHTDiagnostics:
        t = self.state.temperature
        e_f = float(np.sum(t[self.fluid_mask] * self.props.rho_cp_fluid * self._lumped[self.fluid_mask] / max(self.props.rho_cp_fluid, 1.0)))
        # energy uses rho_cp lumped: lumped_i = rho_cp * vol_i
        fluid_lump = assemble_region_lumped_mass(self.mesh, self.mesh.fluid_tets, self.props.rho_cp_fluid)
        solid_lump = assemble_region_lumped_mass(self.mesh, self.mesh.solid_tets, self.props.rho_cp_solid)
        e_f = float(np.sum(t * fluid_lump))
        e_s = float(np.sum(t * solid_lump))
        qf, qs = interface_heat_flux(
            self.mesh,
            t,
            self.props.k_fluid,
            self.props.k_solid,
        )
        from coldplate_studio.fr.thermal_p1 import interface_flux_imbalance

        imb = interface_flux_imbalance(qf, qs)
        return CHTDiagnostics(
            total_energy_j=e_f + e_s,
            fluid_energy_j=e_f,
            solid_energy_j=e_s,
            interface_flux_fluid_w=qf,
            interface_flux_solid_w=qs,
            interface_flux_imbalance=imb,
        )
