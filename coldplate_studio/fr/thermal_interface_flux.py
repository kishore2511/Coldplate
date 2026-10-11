"""Interface heat transfer: local P1 gradients vs Galerkin (weak-form) flux."""

from __future__ import annotations

import numpy as np

from coldplate_studio.fr.thermal_p1 import assemble_region_stiffness, interface_heat_flux, interface_flux_imbalance
from coldplate_studio.mesh.integral_model import IntegralMesh


def galerkin_interface_conduction_rates(
    mesh: IntegralMesh,
    temperature: np.ndarray,
    k_fluid: float,
    k_solid: float,
) -> tuple[float, float]:
    """
    Discrete weak-form conductive exchange at shared interface nodes.

    For each interface node i, fluid-side power leaving the fluid subdomain is
    (K_f T)_i; solid-side power leaving the solid subdomain is (K_s T)_i. At a
    conforming interface with shared temperature DOFs, K_f + K_s is the assembled
    conduction operator and (K_f T)_i + (K_s T)_i is the net discrete balance at i.
    Summing over interface nodes yields equal-and-opposite transfer when the
    global solve is consistent.
    """
    from coldplate_studio.mesh.interface_geometry import analyze_interface_geometry

    k_f = assemble_region_stiffness(mesh, mesh.fluid_tets, k_fluid)
    k_s = assemble_region_stiffness(mesh, mesh.solid_tets, k_solid)
    kt = temperature
    qf_nodes = k_f @ kt
    qs_nodes = k_s @ kt
    geom = analyze_interface_geometry(mesh)
    iface_nodes = np.array(sorted({int(i) for tri in mesh.interface_triangles for i in tri}), dtype=int)
    q_fluid = float(np.sum(qf_nodes[iface_nodes]))
    q_solid = float(np.sum(qs_nodes[iface_nodes]))
    return q_fluid, q_solid


def interface_flux_report(
    mesh: IntegralMesh,
    temperature: np.ndarray,
    k_fluid: float,
    k_solid: float,
) -> dict:
    """Compare local gradient fluxes and Galerkin nodal exchange."""
    qf_loc, qs_loc = interface_heat_flux(mesh, temperature, k_fluid, k_solid)
    qf_g, qs_g = galerkin_interface_conduction_rates(mesh, temperature, k_fluid, k_solid)
    return {
        "local_flux_fluid_w": qf_loc,
        "local_flux_solid_w": qs_loc,
        "local_imbalance": interface_flux_imbalance(qf_loc, qs_loc),
        "galerkin_flux_fluid_w": qf_g,
        "galerkin_flux_solid_w": qs_g,
        "galerkin_imbalance": interface_flux_imbalance(qf_g, qs_g),
    }
