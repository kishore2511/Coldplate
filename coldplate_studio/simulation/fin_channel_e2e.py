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
from coldplate_studio.diagnostics.flow_refinement import (
    refinement_table_markdown,
    run_fin_channel_flow_refinement,
)
from coldplate_studio.diagnostics.preflight import run_integral_preflight
from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.simulation.cht_fin_channel_case import run_fin_channel_cht_case
from coldplate_studio.simulation.flow_channel import run_channel_flow
from coldplate_studio.simulation.flow_channel_boundaries import build_integral_channel_flow_boundaries


@dataclass
class FinChannelE2EResult:
    mesh_size: float
    node_count: int
    fluid_tet_count: int
    solid_tet_count: int
    formulation: str
    flow: dict
    flow_gates: dict
    flow_refinement: list[dict]
    cht: dict | None
    cht_gates: dict | None
    cht_skipped: bool
    ok: bool
    report_path: str | None = None


def run_fin_channel_e2e(
    output_dir: Path | None = None,
    mesh_size: float = 0.08,
    thermal_steps: int = 40,
    thermal_dt: float = 0.05,
    run_refinement_study: bool = True,
) -> FinChannelE2EResult:
    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    model = build_integral_channel_model(mesh_size=mesh_size)
    preflight = run_integral_preflight(model)
    if not preflight.ok:
        raise RuntimeError(f"Preflight failed: {preflight.items}")

    boundaries = build_integral_channel_flow_boundaries(model)
    if not boundaries.inlet_faces or not boundaries.outlet_faces:
        raise RuntimeError(
            f"Port face integrity failed: inlet={len(boundaries.inlet_faces)} outlet={len(boundaries.outlet_faces)}"
        )

    flow_run = run_channel_flow(model, formulation="p2p1")
    flow_gates = evaluate_flow_gates(flow_run)

    refinement_rows = (
        run_fin_channel_flow_refinement((0.12, 0.08, 0.06)) if run_refinement_study else []
    )
    refinement_dicts = [asdict(r) for r in refinement_rows]

    mesh = model.mesh
    flow_dict = {
        "divergence_l2": flow_run.flow.divergence_l2,
        "inlet_flux_m3_s": flow_run.flow.inlet_flux_m3_s,
        "outlet_flux_m3_s": flow_run.flow.outlet_flux_m3_s,
        "mass_flux_relative_imbalance": flow_run.mass_flux_relative_imbalance,
        "pressure_drop": flow_run.flow.pressure_drop,
        "solver_residual_norm": flow_run.flow.solver_residual_norm,
        "max_velocity_magnitude": float(np.max(np.linalg.norm(flow_run.flow.velocity, axis=1))),
        "inlet_port_faces": len(boundaries.inlet_faces),
        "outlet_port_faces": len(boundaries.outlet_faces),
    }

    cht_dict: dict | None = None
    cht_gates_dict: dict | None = None
    cht_skipped = not flow_gates.ok
    cht_ok = True
    if flow_gates.ok:
        _, case = run_fin_channel_cht_case(
            model,
            flow_run,
            boundaries,
            thermal_steps=thermal_steps,
            thermal_dt=thermal_dt,
        )
        flux_mag = max(abs(case.interface_flux_fluid_w), abs(case.interface_flux_solid_w))
        from coldplate_studio.fr.cht_energy import EnergyBalanceReport

        bal_stub = EnergyBalanceReport(
            stored_energy_j=0.0,
            stored_energy_prev_j=0.0,
            delta_stored_j=case.delta_stored_energy_j,
            interface_flux_fluid_w=case.interface_flux_fluid_w,
            interface_flux_solid_w=case.interface_flux_solid_w,
            boundary_heat_in_w=case.enthalpy_in_w,
            boundary_heat_out_w=case.enthalpy_out_w,
            residual_w=0.0,
            relative_residual=case.energy_relative_residual,
            interface_flux_fluid_implied_w=case.interface_flux_fluid_implied_w,
        )
        cht_gates_report = evaluate_cht_gates(
            case.interface_flux_imbalance,
            bal_stub,
            interface_flux_magnitude_w=flux_mag,
            interface_flux_fluid_w=case.interface_flux_fluid_w,
            interface_flux_solid_w=case.interface_flux_solid_w,
            enforce_interface_consistency=True,
        )
        cht_gates_dict = asdict(cht_gates_report)
        cht_ok = cht_gates_report.ok
        cht_dict = {
            "fluid_temperature_k": [case.fluid_temperature_min_k, case.fluid_temperature_max_k],
            "solid_temperature_k": [case.solid_temperature_min_k, case.solid_temperature_max_k],
            "interface_flux_fluid_w": case.interface_flux_fluid_w,
            "interface_flux_solid_w": case.interface_flux_solid_w,
            "interface_flux_imbalance": case.interface_flux_imbalance,
            "interface_flux_side_ratio": case.interface_flux_side_ratio,
            "galerkin_flux_fluid_w": case.galerkin_flux_fluid_w,
            "galerkin_flux_solid_w": case.galerkin_flux_solid_w,
            "galerkin_flux_imbalance": case.galerkin_flux_imbalance,
            "interface_flux_fluid_implied_w": case.interface_flux_fluid_implied_w,
            "delta_stored_energy_j": case.delta_stored_energy_j,
            "enthalpy_in_w": case.enthalpy_in_w,
            "enthalpy_out_w": case.enthalpy_out_w,
            "energy_relative_residual": case.energy_relative_residual,
            "solid_volumetric_heat_w": case.solid_volumetric_heat_w,
            "fluid_inlet_nodes": case.inlet_nodes_count,
            "thermal_drive": "port-face Dirichlet + solid base 300 K + volumetric solid source (W/m³)",
        }
    else:
        cht_dict = {"skipped_reason": "flow gates failed", "flow_gate_messages": flow_gates.messages}

    ok = flow_gates.ok and cht_ok
    result = FinChannelE2EResult(
        mesh_size=mesh_size,
        node_count=int(mesh.nodes.shape[0]),
        fluid_tet_count=int(mesh.fluid_tets.shape[0]),
        solid_tet_count=int(mesh.solid_tets.shape[0]),
        formulation=flow_run.formulation,
        flow=flow_dict,
        flow_gates=asdict(flow_gates),
        flow_refinement=refinement_dicts,
        cht=cht_dict,
        cht_gates=cht_gates_dict,
        cht_skipped=cht_skipped,
        ok=ok,
    )
    if out:
        summary = asdict(result)
        json_path = out / "fin_channel_e2e_summary.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        md_path = out / "fin_channel_e2e_report.md"
        md_path.write_text(_markdown_report(result, refinement_rows), encoding="utf-8")
        result.report_path = str(md_path)
    return result


def _markdown_report(r: FinChannelE2EResult, refinement_rows) -> str:
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
    if refinement_rows:
        lines.extend(["", "## Flow refinement (PSPG, three mesh sizes)", "", refinement_table_markdown(refinement_rows), ""])
    lines.extend(["", "## CHT", ""])
    if r.cht:
        for k, v in r.cht.items():
            lines.append(f"- {k}: {v}")
    if r.cht_gates:
        lines.extend(["", "## CHT gates", "", f"- OK: {r.cht_gates['ok']}", ""])
        for m in r.cht_gates.get("messages", []):
            lines.append(f"- FAIL: {m}")
    elif r.cht_skipped:
        lines.append("- CHT not run (flow gates failed).")
    lines.append("")
    lines.append(f"**Overall OK:** {r.ok}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fin-channel single-phase E2E with conservation gates")
    parser.add_argument("--output", type=Path, default=Path("output/fin_channel_e2e"))
    parser.add_argument("--mesh-size", type=float, default=0.08)
    parser.add_argument("--no-refinement-study", action="store_true")
    args = parser.parse_args(argv)
    result = run_fin_channel_e2e(
        args.output,
        mesh_size=args.mesh_size,
        run_refinement_study=not args.no_refinement_study,
    )
    print(json.dumps(asdict(result), indent=2, default=str))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
