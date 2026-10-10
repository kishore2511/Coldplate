"""
End-to-end single-phase fin-channel: mesh → flow (PSPG) → CHT → gated report.

Stops with a non-zero exit code when a gate fails (see ``main``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from coldplate_studio.diagnostics.cht_gates import evaluate_cht_gates
from coldplate_studio.diagnostics.flow_gates import evaluate_flow_gates
from coldplate_studio.diagnostics.preflight import run_integral_preflight
from coldplate_studio.fr.cht_energy import transient_energy_balance
from coldplate_studio.fr.cht_transient import (
    ConjugateHeatTransferSolver,
    ThermalBoundaryConditions,
    ThermalMaterialProperties,
)
from coldplate_studio.fr.thermal_p1 import interface_heat_flux
from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.simulation.flow_channel import run_channel_flow


@dataclass
class FinChannelE2EResult:
    mesh_size: float
    node_count: int
    fluid_tet_count: int
    solid_tet_count: int
    formulation: str
    flow: dict
    flow_gates: dict
    cht: dict
    cht_gates: dict
    ok: bool
    report_path: str | None = None


def _inlet_outlet_node_sets(mesh, length: float = 1.0):
    inlet = np.array(
        [i for i in mesh.boundary_fluid_nodes if mesh.nodes[i, 0] < 0.02],
        dtype=int,
    )
    outlet = np.array(
        [i for i in mesh.boundary_fluid_nodes if mesh.nodes[i, 0] > length - 0.02],
        dtype=int,
    )
    return inlet, outlet


def run_fin_channel_e2e(
    output_dir: Path | None = None,
    mesh_size: float = 0.08,
    thermal_steps: int = 25,
    thermal_dt: float = 0.02,
) -> FinChannelE2EResult:
    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    model = build_integral_channel_model(mesh_size=mesh_size)
    preflight = run_integral_preflight(model)
    if not preflight.ok:
        raise RuntimeError(f"Preflight failed: {preflight.items}")

    flow_run = run_channel_flow(model, formulation="pspg")
    flow_gates = evaluate_flow_gates(flow_run)

    mesh = model.mesh
    inlet, outlet = _inlet_outlet_node_sets(mesh)
    props = ThermalMaterialProperties()
    bcs = ThermalBoundaryConditions(
        fluid_inlet_nodes=inlet[: min(12, inlet.size)],
        fluid_inlet_temperature_k=360.0,
    )
    cht = ConjugateHeatTransferSolver(mesh, flow_run.flow.velocity, props, bcs)
    t_prev = cht.state.temperature.copy()
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
        inlet_nodes=inlet[: min(12, inlet.size)],
        inlet_temperature=360.0,
        outlet_nodes=outlet[: min(12, outlet.size)],
        velocity=flow_run.flow.velocity,
    )
    flux_mag = max(abs(diag.interface_flux_fluid_w), abs(diag.interface_flux_solid_w))
    cht_gates = evaluate_cht_gates(
        diag.interface_flux_imbalance,
        bal,
        interface_flux_magnitude_w=flux_mag,
    )

    t_solid = cht.state.temperature[cht.solid_mask]
    q_in = props.k_solid * (360.0 - 300.0)  # diagnostic scale; BC is nodal Dirichlet
    r_th = (float(np.max(t_solid)) - 300.0) / max(q_in, 1e-12)

    flow_dict = {
        "divergence_l2": flow_run.flow.divergence_l2,
        "inlet_flux_m3_s": flow_run.flow.inlet_flux_m3_s,
        "outlet_flux_m3_s": flow_run.flow.outlet_flux_m3_s,
        "mass_flux_relative_imbalance": flow_run.mass_flux_relative_imbalance,
        "pressure_drop": flow_run.flow.pressure_drop,
        "solver_residual_norm": flow_run.flow.solver_residual_norm,
        "max_velocity_magnitude": float(np.max(np.linalg.norm(flow_run.flow.velocity, axis=1))),
    }
    cht_dict = {
        "interface_flux_fluid_w": diag.interface_flux_fluid_w,
        "interface_flux_solid_w": diag.interface_flux_solid_w,
        "interface_flux_imbalance": diag.interface_flux_imbalance,
        "energy_relative_residual": bal.relative_residual,
        "max_solid_temperature_k": float(np.max(t_solid)) if t_solid.size else None,
        "thermal_resistance_proxy_k_per_w": r_th,
        "thermal_resistance_reference": "T_ref=300 K solid outer Dirichlet; heat scale k_solid*(T_inlet-T_ref)",
    }

    ok = flow_gates.ok and cht_gates.ok
    result = FinChannelE2EResult(
        mesh_size=mesh_size,
        node_count=int(mesh.nodes.shape[0]),
        fluid_tet_count=int(mesh.fluid_tets.shape[0]),
        solid_tet_count=int(mesh.solid_tets.shape[0]),
        formulation=flow_run.formulation,
        flow=flow_dict,
        flow_gates=asdict(flow_gates),
        cht=cht_dict,
        cht_gates=asdict(cht_gates),
        ok=ok,
    )
    if out:
        summary = asdict(result)
        json_path = out / "fin_channel_e2e_summary.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        md_path = out / "fin_channel_e2e_report.md"
        md_path.write_text(_markdown_report(result), encoding="utf-8")
        result.report_path = str(md_path)
    return result


def _markdown_report(r: FinChannelE2EResult) -> str:
    lines = [
        "# Fin-channel E2E report",
        "",
        f"- Mesh size: {r.mesh_size}",
        f"- Nodes: {r.node_count}, fluid tets: {r.fluid_tet_count}, solid tets: {r.solid_tet_count}",
        f"- Flow formulation: {r.formulation}",
        "",
        "## Flow",
        "",
    ]
    for k, v in r.flow.items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Flow gates", "", f"- OK: {r.flow_gates['ok']}", ""])
    for m in r.flow_gates.get("messages", []):
        lines.append(f"- FAIL: {m}")
    lines.extend(["", "## CHT", ""])
    for k, v in r.cht.items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## CHT gates", "", f"- OK: {r.cht_gates['ok']}", ""])
    for m in r.cht_gates.get("messages", []):
        lines.append(f"- FAIL: {m}")
    lines.append("")
    lines.append(f"**Overall OK:** {r.ok}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fin-channel single-phase E2E with conservation gates")
    parser.add_argument("--output", type=Path, default=Path("output/fin_channel_e2e"))
    parser.add_argument("--mesh-size", type=float, default=0.08)
    args = parser.parse_args(argv)
    result = run_fin_channel_e2e(args.output, mesh_size=args.mesh_size)
    print(json.dumps(asdict(result), indent=2, default=str))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
