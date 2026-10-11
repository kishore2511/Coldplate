"""Mesh-refinement flow metrics for integral fin channel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.fr.flow_p1_operators import fluid_node_mask
from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.simulation.flow_channel import run_channel_flow

@dataclass
class FlowRefinementRow:
    mesh_size: float
    node_count: int
    fluid_tet_count: int
    velocity_dofs: int
    pressure_dofs: int
    inlet_flux_m3_s: float
    outlet_flux_m3_s: float
    mass_flux_relative_imbalance: float
    divergence_l2: float
    pressure_drop: float | None
    solver_residual_norm: float | None
    max_velocity_magnitude: float
    flow_gates_ok: bool = False


def run_fin_channel_flow_refinement(
    mesh_sizes: tuple[float, ...] = (0.12, 0.08, 0.06),
    *,
    formulation: str = "p2p1",
) -> list[FlowRefinementRow]:
    rows: list[FlowRefinementRow] = []
    for ms in mesh_sizes:
        model = build_integral_channel_model(mesh_size=ms)
        mesh = model.mesh
        n = mesh.nodes.shape[0]
        run = run_channel_flow(model, formulation=formulation)
        f = run.flow
        fluid = fluid_node_mask(mesh)
        n_p_fluid = int(np.sum(fluid))
        umax = float(max((float((f.velocity[i] ** 2).sum()) ** 0.5 for i in range(n) if fluid[i]), default=0.0))
        from coldplate_studio.diagnostics.flow_gates import evaluate_flow_gates

        gates_ok = evaluate_flow_gates(run).ok
        rows.append(
            FlowRefinementRow(
                mesh_size=ms,
                node_count=n,
                fluid_tet_count=int(mesh.fluid_tets.shape[0]),
                velocity_dofs=3 * n,
                pressure_dofs=n_p_fluid,
                inlet_flux_m3_s=float(f.inlet_flux_m3_s),
                outlet_flux_m3_s=float(f.outlet_flux_m3_s),
                mass_flux_relative_imbalance=float(run.mass_flux_relative_imbalance),
                divergence_l2=float(f.divergence_l2),
                pressure_drop=f.pressure_drop,
                solver_residual_norm=f.solver_residual_norm,
                max_velocity_magnitude=umax,
                flow_gates_ok=gates_ok,
            )
        )
    return rows


def refinement_table_markdown(rows: list[FlowRefinementRow]) -> str:
    lines = [
        "| mesh_size | nodes | fluid tets | p DOFs | Φ_in (m³/s) | Φ_out (m³/s) | imbalance | ‖div u‖ | gates |",
        "|-----------|-------|------------|--------|-------------|--------------|-----------|---------|-------|",
    ]
    for r in rows:
        lines.append(
            f"| {r.mesh_size} | {r.node_count} | {r.fluid_tet_count} | {r.pressure_dofs} | "
            f"{r.inlet_flux_m3_s:.4e} | {r.outlet_flux_m3_s:.4e} | "
            f"{r.mass_flux_relative_imbalance:.3f} | {r.divergence_l2:.4e} | {r.flow_gates_ok} |"
        )
    return "\n".join(lines)
