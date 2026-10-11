"""Channel flow on integral fin geometry (PSPG production path + penalty legacy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from coldplate_studio.connectivity.ports import channel_ports_from_user_confirmation
from coldplate_studio.fr.flow_p1_penalty import FlowSolveResult, solve_steady_channel_flow
from coldplate_studio.fr.flow_p1_pspg_mixed import solve_integral_channel_pspg
from coldplate_studio.mesh.integral_model import IntegralModel
from coldplate_studio.simulation.flow_channel_boundaries import (
    _select_channel_inlet_outlet_faces,
    build_integral_channel_flow_boundaries,
)


@dataclass
class ChannelFlowRunResult:
    flow: FlowSolveResult
    ports_ok: bool
    mass_flux_relative_imbalance: float
    formulation: str


def run_channel_flow(
    model: IntegralModel,
    length: float = 1.0,
    *,
    formulation: Literal["pspg", "penalty", "pspg_inlet_parabolic", "pspg_legacy_body_force"] = "pspg",
    momentum_grad_div_beta: float = 100.0,
) -> ChannelFlowRunResult:
    boundaries = build_integral_channel_flow_boundaries(model, length)
    ports = channel_ports_from_user_confirmation(model.mesh, boundaries.inlet_faces, boundaries.outlet_faces)
    if not ports.ok_for_flow_solve:
        raise RuntimeError(f"User port confirmation failed: {ports.notes}")
    if formulation == "penalty":
        flow = solve_steady_channel_flow(
            model.mesh,
            inlet_faces=boundaries.inlet_faces,
            outlet_faces=boundaries.outlet_faces,
            no_slip_nodes=boundaries.no_slip_nodes,
            zero_velocity_nodes=boundaries.solid_only_nodes,
        )
        rel = abs(flow.inlet_flux_m3_s + flow.outlet_flux_m3_s) / max(
            0.5 * (abs(flow.inlet_flux_m3_s) + abs(flow.outlet_flux_m3_s)),
            1e-12,
        )
        return ChannelFlowRunResult(
            flow=flow,
            ports_ok=True,
            mass_flux_relative_imbalance=float(rel),
            formulation="penalty_stokes",
        )
    if formulation == "pspg_inlet_parabolic":
        pspg = solve_integral_channel_pspg(model.mesh, boundaries, drive="inlet_parabolic")
        formulation_name = "pspg_inlet_parabolic"
    else:
        pspg = solve_integral_channel_pspg(
            model.mesh,
            boundaries,
            drive="body_force_legacy",
            dp_dx=-1.0,
            momentum_grad_div_beta=momentum_grad_div_beta,
        )
        formulation_name = "pspg_mixed_stokes"
    fluid_mask = np.zeros(model.mesh.nodes.shape[0], dtype=bool)
    for tet in model.mesh.fluid_tets:
        fluid_mask[tet] = True
    flow = FlowSolveResult(
        velocity=pspg.velocity,
        divergence_l2=pspg.divergence_l2,
        inlet_flux_m3_s=float(pspg.inlet_flux_m3_s or 0.0),
        outlet_flux_m3_s=float(pspg.outlet_flux_m3_s or 0.0),
        mean_velocity_x=float(np.mean(pspg.velocity[fluid_mask, 0])),
        formulation=formulation_name,
        pressure=pspg.pressure,
        pressure_drop=pspg.pressure_drop,
        solver_residual_norm=pspg.solver_residual_norm,
    )
    return ChannelFlowRunResult(
        flow=flow,
        ports_ok=True,
        mass_flux_relative_imbalance=float(pspg.mass_flux_relative_imbalance or 0.0),
        formulation=formulation_name,
    )
