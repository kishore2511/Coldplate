"""
Deterministic coupled CHT on integral fin channel (verified flow + documented BCs).

Uses tagged port nodes for hot fluid inlet, solid-base Dirichlet, and optional
volumetric heating in the solid to guarantee measurable interface flux on coarse meshes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.fr.cht_energy import transient_energy_balance
from coldplate_studio.fr.cht_transient import (
    ConjugateHeatTransferSolver,
    ThermalBoundaryConditions,
    ThermalMaterialProperties,
)
from coldplate_studio.fr.thermal_interface_flux import interface_flux_report
from coldplate_studio.fr.thermal_p1 import interface_flux_imbalance
from coldplate_studio.mesh.integral_model import IntegralModel
from coldplate_studio.simulation.flow_channel_boundaries import (
    channel_port_fluid_nodes,
    solid_base_dirichlet_nodes,
)
from coldplate_studio.simulation.flow_channel import ChannelFlowRunResult


@dataclass
class FinChannelCHTCaseResult:
    fluid_temperature_min_k: float
    fluid_temperature_max_k: float
    solid_temperature_min_k: float
    solid_temperature_max_k: float
    interface_flux_fluid_w: float
    interface_flux_solid_w: float
    interface_flux_imbalance: float
    interface_flux_fluid_implied_w: float
    interface_flux_side_ratio: float
    galerkin_flux_fluid_w: float
    galerkin_flux_solid_w: float
    galerkin_flux_imbalance: float
    energy_relative_residual: float
    delta_stored_energy_j: float
    enthalpy_in_w: float
    enthalpy_out_w: float
    solid_volumetric_heat_w: float
    inlet_nodes_count: int


def run_fin_channel_cht_case(
    model: IntegralModel,
    flow_run: ChannelFlowRunResult,
    boundaries,
    *,
    thermal_steps: int = 40,
    thermal_dt: float = 0.05,
    inlet_temperature_k: float = 360.0,
    solid_base_temperature_k: float = 300.0,
    solid_volumetric_heat_w_m3: float = 5.0e5,
) -> tuple[ConjugateHeatTransferSolver, FinChannelCHTCaseResult]:
    mesh = model.mesh
    inlet, outlet = channel_port_fluid_nodes(boundaries)
    solid_base = solid_base_dirichlet_nodes(mesh)
    props = ThermalMaterialProperties()
    bcs = ThermalBoundaryConditions(
        fluid_inlet_nodes=inlet,
        fluid_inlet_temperature_k=inlet_temperature_k,
        solid_outer_nodes=solid_base,
        solid_outer_temperature_k=solid_base_temperature_k,
        solid_volumetric_heat_w_m3=solid_volumetric_heat_w_m3,
    )
    cht = ConjugateHeatTransferSolver(
        mesh,
        flow_run.flow.velocity,
        props,
        bcs,
        p2_velocity_field=getattr(flow_run.flow, "p2_velocity_field", None),
    )
    t_prev = cht.state.temperature.copy()
    vol_lump = np.zeros(mesh.nodes.shape[0])
    from coldplate_studio.fr.thermal_p1 import assemble_region_lumped_mass

    vol_lump = assemble_region_lumped_mass(mesh, mesh.solid_tets, 1.0)
    q_vol_w = float(solid_volumetric_heat_w_m3 * np.sum(vol_lump))
    for _ in range(thermal_steps):
        t_prev = cht.state.temperature.copy()
        cht.step(thermal_dt)
    diag = cht.diagnostics()
    bal = transient_energy_balance(
        mesh,
        t_prev,
        cht.state.temperature,
        thermal_dt,
        rho_cp_fluid=props.rho_cp_fluid,
        rho_cp_solid=props.rho_cp_solid,
        k_fluid=props.k_fluid,
        k_solid=props.k_solid,
        inlet_nodes=inlet,
        inlet_temperature=inlet_temperature_k,
        outlet_nodes=outlet,
        velocity=flow_run.flow.velocity,
        solid_volumetric_heat_w=q_vol_w,
    )
    fluid_mask = cht.fluid_mask
    solid_mask = cht.solid_mask
    t = cht.state.temperature
    qf, qs = diag.interface_flux_fluid_w, diag.interface_flux_solid_w
    flux_rep = interface_flux_report(mesh, t, props.k_fluid, props.k_solid)
    side_ratio = min(abs(qf), abs(qs)) / max(abs(qf), abs(qs), 1e-12)
    case = FinChannelCHTCaseResult(
        fluid_temperature_min_k=float(np.min(t[fluid_mask])) if fluid_mask.any() else inlet_temperature_k,
        fluid_temperature_max_k=float(np.max(t[fluid_mask])) if fluid_mask.any() else inlet_temperature_k,
        solid_temperature_min_k=float(np.min(t[solid_mask])) if solid_mask.any() else solid_base_temperature_k,
        solid_temperature_max_k=float(np.max(t[solid_mask])) if solid_mask.any() else solid_base_temperature_k,
        interface_flux_fluid_w=qf,
        interface_flux_solid_w=qs,
        interface_flux_imbalance=interface_flux_imbalance(qf, qs),
        interface_flux_fluid_implied_w=bal.interface_flux_fluid_implied_w,
        interface_flux_side_ratio=float(side_ratio),
        galerkin_flux_fluid_w=float(flux_rep["galerkin_flux_fluid_w"]),
        galerkin_flux_solid_w=float(flux_rep["galerkin_flux_solid_w"]),
        galerkin_flux_imbalance=float(flux_rep["galerkin_imbalance"]),
        energy_relative_residual=bal.relative_residual,
        delta_stored_energy_j=bal.delta_stored_j,
        enthalpy_in_w=bal.boundary_heat_in_w,
        enthalpy_out_w=bal.boundary_heat_out_w,
        solid_volumetric_heat_w=q_vol_w,
        inlet_nodes_count=int(inlet.size),
    )
    return cht, case
