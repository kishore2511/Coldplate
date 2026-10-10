"""Channel flow vertical slice: user ports + P1 penalty Stokes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.connectivity.ports import channel_ports_from_user_confirmation
from coldplate_studio.fr.flow_p1_penalty import FlowSolveResult, solve_steady_channel_flow
from coldplate_studio.mesh.boundary_faces import classify_boundary_faces
from coldplate_studio.mesh.integral_model import IntegralModel


@dataclass
class ChannelFlowRunResult:
    flow: FlowSolveResult
    ports_ok: bool
    mass_flux_relative_imbalance: float


def _select_channel_inlet_outlet_faces(model: IntegralModel, x_in: float, x_out: float, tol: float = 0.02):
    """User confirmation helper: faces whose centroid x ≈ x_in / x_out (explicit tolerances)."""
    report = classify_boundary_faces(model.mesh)
    inlet = []
    outlet = []
    for f in report.faces:
        if f.face_type != "exterior_fluid":
            continue
        if abs(f.centroid[0] - x_in) <= tol:
            inlet.append(f.nodes)
        if abs(f.centroid[0] - x_out) <= tol:
            outlet.append(f.nodes)
    return inlet, outlet


def run_channel_flow(model: IntegralModel, length: float = 1.0) -> ChannelFlowRunResult:
    inlet, outlet = _select_channel_inlet_outlet_faces(model, 0.0, length)
    ports = channel_ports_from_user_confirmation(model.mesh, inlet, outlet)
    if not ports.ok_for_flow_solve:
        raise RuntimeError(f"User port confirmation failed: {ports.notes}")
    inlet_nodes = {i for face in inlet for i in face}
    outlet_nodes = {i for face in outlet for i in face}
    port_nodes = inlet_nodes | outlet_nodes
    wall_nodes = np.array(
        [i for i in model.mesh.boundary_fluid_nodes if int(i) not in port_nodes],
        dtype=int,
    )
    fluid_nodes = {int(t) for tet in model.mesh.fluid_tets for t in tet}
    solid_only = np.array(
        [i for i in range(model.mesh.nodes.shape[0]) if i not in fluid_nodes],
        dtype=int,
    )
    flow = solve_steady_channel_flow(
        model.mesh,
        inlet_faces=inlet,
        outlet_faces=outlet,
        no_slip_nodes=wall_nodes,
        zero_velocity_nodes=solid_only,
    )
    mean_flux = 0.5 * (abs(flow.inlet_flux_m3_s) + abs(flow.outlet_flux_m3_s))
    rel = abs(flow.inlet_flux_m3_s + flow.outlet_flux_m3_s) / max(mean_flux, 1e-12)
    return ChannelFlowRunResult(flow=flow, ports_ok=True, mass_flux_relative_imbalance=float(rel))
