"""Integral fin-channel CHT with zero velocity (conduction-dominated interface check)."""

from __future__ import annotations

import numpy as np

from coldplate_studio.fr.cht_energy import transient_energy_balance
from coldplate_studio.fr.cht_transient import (
    ConjugateHeatTransferSolver,
    ThermalBoundaryConditions,
    ThermalMaterialProperties,
)
from coldplate_studio.fr.thermal_p1 import (
    assemble_region_lumped_mass,
    interface_flux_imbalance,
    interface_heat_flux,
)
from coldplate_studio.mesh.integral_model import IntegralModel
from coldplate_studio.simulation.flow_channel_boundaries import (
    channel_port_fluid_nodes,
    solid_base_dirichlet_nodes,
)


def run_fin_channel_conduction_case(
    model: IntegralModel,
    boundaries,
    *,
    thermal_steps: int = 30,
    thermal_dt: float = 0.1,
    inlet_temperature_k: float = 380.0,
) -> dict:
    """
    Hot fluid inlet, cold solid-only base, u=0, mild solid volumetric heat.

    Mild volumetric heating guarantees measurable interface flux on the coarse integral
    fin mesh while flow remains zero; local side fluxes are more comparable than the
    advection-dominated coupled demo.
    """
    mesh = model.mesh
    inlet, outlet = channel_port_fluid_nodes(boundaries)
    solid_base = solid_base_dirichlet_nodes(mesh)
    props = ThermalMaterialProperties()
    bcs = ThermalBoundaryConditions(
        fluid_inlet_nodes=inlet,
        fluid_inlet_temperature_k=inlet_temperature_k,
        solid_outer_nodes=solid_base,
        solid_outer_temperature_k=300.0,
        solid_volumetric_heat_w_m3=2.0e4,
    )
    zero_u = np.zeros_like(mesh.nodes)
    vol_lump = assemble_region_lumped_mass(mesh, mesh.solid_tets, 1.0)
    q_vol_w = float(bcs.solid_volumetric_heat_w_m3 * np.sum(vol_lump))
    cht = ConjugateHeatTransferSolver(mesh, zero_u, props, bcs)
    t_prev = cht.state.temperature.copy()
    for _ in range(thermal_steps):
        t_prev = cht.state.temperature.copy()
        cht.step(thermal_dt)
    t = cht.state.temperature
    qf, qs = interface_heat_flux(mesh, t, props.k_fluid, props.k_solid)
    bal = transient_energy_balance(
        mesh,
        t_prev,
        t,
        thermal_dt,
        rho_cp_fluid=props.rho_cp_fluid,
        rho_cp_solid=props.rho_cp_solid,
        k_fluid=props.k_fluid,
        k_solid=props.k_solid,
        inlet_nodes=inlet,
        inlet_temperature=inlet_temperature_k,
        outlet_nodes=outlet,
        velocity=zero_u,
        solid_volumetric_heat_w=q_vol_w,
    )
    imb = interface_flux_imbalance(qf, qs)
    side_ratio = min(abs(qf), abs(qs)) / max(abs(qf), abs(qs), 1e-12)
    return {
        "interface_flux_fluid_w": qf,
        "interface_flux_solid_w": qs,
        "interface_flux_imbalance": imb,
        "interface_flux_fluid_implied_w": bal.interface_flux_fluid_implied_w,
        "interface_flux_side_ratio": side_ratio,
        "energy_relative_residual": bal.relative_residual,
        "max_solid_temperature_k": float(np.max(t[cht.solid_mask])),
    }
