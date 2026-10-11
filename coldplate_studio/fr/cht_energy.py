"""Global CHT energy balance accounting (semi-implicit BE step)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.fr.thermal_p1 import assemble_region_lumped_mass, interface_heat_flux
from coldplate_studio.mesh.integral_model import IntegralMesh


@dataclass
class EnergyBalanceReport:
    stored_energy_j: float
    stored_energy_prev_j: float
    delta_stored_j: float
    interface_flux_fluid_w: float
    interface_flux_solid_w: float
    boundary_heat_in_w: float
    boundary_heat_out_w: float
    residual_w: float
    relative_residual: float
    interface_flux_fluid_implied_w: float = 0.0


def nodal_stored_energy(mesh: IntegralMesh, temperature: np.ndarray, rho_cp_fluid: float, rho_cp_solid: float) -> float:
    m_f = assemble_region_lumped_mass(mesh, mesh.fluid_tets, rho_cp_fluid)
    m_s = assemble_region_lumped_mass(mesh, mesh.solid_tets, rho_cp_solid)
    return float(np.sum(temperature * (m_f + m_s)))


def boundary_conduction_power(
    mesh: IntegralMesh,
    temperature: np.ndarray,
    boundary_nodes: np.ndarray,
    k: float,
    t_ambient: float,
) -> float:
    """Approximate Robin-like flux k (T-T_amb)/h_char on listed nodes (diagnostic)."""
    if boundary_nodes.size == 0:
        return 0.0
    h_char = 0.01
    return float(np.sum(k * (temperature[boundary_nodes] - t_ambient) * h_char))


def transient_energy_balance(
    mesh: IntegralMesh,
    t_prev: np.ndarray,
    t_new: np.ndarray,
    dt: float,
    *,
    rho_cp_fluid: float,
    rho_cp_solid: float,
    k_fluid: float,
    k_solid: float,
    inlet_nodes: np.ndarray,
    inlet_temperature: float,
    outlet_nodes: np.ndarray,
    velocity: np.ndarray,
    solid_volumetric_heat_w: float = 0.0,
) -> EnergyBalanceReport:
    e_prev = nodal_stored_energy(mesh, t_prev, rho_cp_fluid, rho_cp_solid)
    e_new = nodal_stored_energy(mesh, t_new, rho_cp_fluid, rho_cp_solid)
    qf, qs = interface_heat_flux(mesh, t_new, k_fluid, k_solid)
    # Enthalpy advection through inlet/outlet (ρ c_p u_x T at port nodes)
    rho_cp = rho_cp_fluid
    phi_in = float(np.sum(rho_cp * velocity[inlet_nodes, 0] * t_new[inlet_nodes])) if inlet_nodes.size else 0.0
    phi_out = float(np.sum(rho_cp * velocity[outlet_nodes, 0] * t_new[outlet_nodes])) if outlet_nodes.size else 0.0
    delta = e_new - e_prev
    # BE: ΔE/dt ≈ -Q_interface + net enthalpy flux (sign convention documented)
    residual = delta / dt - (qf + qs) - (phi_in - phi_out) - solid_volumetric_heat_w
    scale = max(abs(delta / dt), abs(qf), abs(phi_in), abs(solid_volumetric_heat_w), 1e-12)
    qf_implied = float(-(qs + delta / dt - (phi_in - phi_out) - solid_volumetric_heat_w))
    return EnergyBalanceReport(
        stored_energy_j=e_new,
        stored_energy_prev_j=e_prev,
        delta_stored_j=delta,
        interface_flux_fluid_w=qf,
        interface_flux_solid_w=qs,
        boundary_heat_in_w=phi_in,
        boundary_heat_out_w=phi_out,
        residual_w=float(residual),
        relative_residual=float(abs(residual) / scale),
        interface_flux_fluid_implied_w=qf_implied,
    )
