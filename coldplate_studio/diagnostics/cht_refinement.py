"""Coupled fin-channel CHT mesh refinement (production P2–P1 flow)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from coldplate_studio.diagnostics.cht_gates import evaluate_cht_gates
from coldplate_studio.diagnostics.flow_gates import evaluate_flow_gates
from coldplate_studio.fr.cht_energy import EnergyBalanceReport
from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.simulation.cht_fin_channel_case import run_fin_channel_cht_case
from coldplate_studio.simulation.flow_channel import run_channel_flow
from coldplate_studio.simulation.flow_channel_boundaries import build_integral_channel_flow_boundaries


@dataclass
class CHTRefinementRow:
    mesh_size: float
    solver_method: str | None
    formulation: str
    mass_flux_relative_imbalance: float
    divergence_l2: float
    flow_gates_ok: bool
    interface_flux_fluid_w: float
    interface_flux_solid_w: float
    interface_flux_imbalance: float
    galerkin_flux_fluid_w: float
    galerkin_flux_solid_w: float
    galerkin_flux_imbalance: float
    enthalpy_in_w: float
    enthalpy_out_w: float
    solid_volumetric_heat_w: float
    delta_stored_energy_j: float
    energy_relative_residual: float
    solid_temperature_max_k: float
    cht_gates_ok: bool


def run_fin_channel_cht_refinement(
    mesh_sizes: tuple[float, ...] = (0.12, 0.08, 0.06),
    *,
    thermal_steps: int = 8,
    thermal_dt: float = 0.05,
) -> list[CHTRefinementRow]:
    rows: list[CHTRefinementRow] = []
    for ms in mesh_sizes:
        model = build_integral_channel_model(mesh_size=ms)
        boundaries = build_integral_channel_flow_boundaries(model)
        flow = run_channel_flow(model, formulation="p2p1")
        flow_ok = evaluate_flow_gates(flow).ok
        _, case = run_fin_channel_cht_case(
            model,
            flow,
            boundaries,
            thermal_steps=thermal_steps,
            thermal_dt=thermal_dt,
        )
        flux_mag = max(abs(case.interface_flux_fluid_w), abs(case.interface_flux_solid_w))
        bal = EnergyBalanceReport(
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
        cht_ok = evaluate_cht_gates(
            case.interface_flux_imbalance,
            bal,
            interface_flux_magnitude_w=flux_mag,
            interface_flux_fluid_w=case.interface_flux_fluid_w,
            interface_flux_solid_w=case.interface_flux_solid_w,
        ).ok
        rows.append(
            CHTRefinementRow(
                mesh_size=ms,
                solver_method=flow.flow.solver_method,
                formulation=flow.formulation,
                mass_flux_relative_imbalance=flow.mass_flux_relative_imbalance,
                divergence_l2=flow.flow.divergence_l2,
                flow_gates_ok=flow_ok,
                interface_flux_fluid_w=case.interface_flux_fluid_w,
                interface_flux_solid_w=case.interface_flux_solid_w,
                interface_flux_imbalance=case.interface_flux_imbalance,
                galerkin_flux_fluid_w=case.galerkin_flux_fluid_w,
                galerkin_flux_solid_w=case.galerkin_flux_solid_w,
                galerkin_flux_imbalance=case.galerkin_flux_imbalance,
                enthalpy_in_w=case.enthalpy_in_w,
                enthalpy_out_w=case.enthalpy_out_w,
                solid_volumetric_heat_w=case.solid_volumetric_heat_w,
                delta_stored_energy_j=case.delta_stored_energy_j,
                energy_relative_residual=case.energy_relative_residual,
                solid_temperature_max_k=case.solid_temperature_max_k,
                cht_gates_ok=cht_ok,
            )
        )
    return rows


def write_cht_refinement_json(path: Path, rows: list[CHTRefinementRow]) -> None:
    path.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
