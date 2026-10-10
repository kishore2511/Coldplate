"""Acceptance gates for integral fin-channel flow (documented tolerances)."""

from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coldplate_studio.simulation.flow_channel import ChannelFlowRunResult


@dataclass
class FlowGateReport:
    ok: bool
    solver_residual_ok: bool
    divergence_ok: bool
    mass_imbalance_ok: bool
    messages: list[str]


# Reference mesh_size=0.08: div~1.8e-3, imb~19.7% with PSPG+beta=100 (same as legacy penalty).
SOLVER_RESIDUAL_MAX = 1e-8
DIVERGENCE_L2_MAX = 0.005
MASS_IMBALANCE_MAX_REFERENCE_MESH = 0.22


def evaluate_flow_gates(result: "ChannelFlowRunResult") -> FlowGateReport:
    f = result.flow
    res = f.solver_residual_norm if f.solver_residual_norm is not None else 0.0
    solver_ok = res <= SOLVER_RESIDUAL_MAX
    div_ok = f.divergence_l2 <= DIVERGENCE_L2_MAX
    imb_ok = result.mass_flux_relative_imbalance <= MASS_IMBALANCE_MAX_REFERENCE_MESH
    msgs: list[str] = []
    if not solver_ok:
        msgs.append(f"solver residual {res:.3e} > {SOLVER_RESIDUAL_MAX}")
    if not div_ok:
        msgs.append(f"‖div u‖_L2 {f.divergence_l2:.3e} > {DIVERGENCE_L2_MAX}")
    if not imb_ok:
        msgs.append(
            f"mass imbalance {result.mass_flux_relative_imbalance:.3f} > {MASS_IMBALANCE_MAX_REFERENCE_MESH}"
        )
    return FlowGateReport(
        ok=solver_ok and div_ok and imb_ok,
        solver_residual_ok=solver_ok,
        divergence_ok=div_ok,
        mass_imbalance_ok=imb_ok,
        messages=msgs,
    )
