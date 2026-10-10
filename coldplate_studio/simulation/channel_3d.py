"""End-to-end integral channel case: one mesh from CAD through solver to VTK/contours."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coldplate_studio.connectivity.diagnostics import run_connectivity_diagnostics
from coldplate_studio.connectivity.graph import build_connectivity_from_extraction
from coldplate_studio.connectivity.interface_maps import build_interface_connectivity
from coldplate_studio.fr.solver3d import PhaseChangeConfig
from coldplate_studio.fr.solver_integral import IntegralScalarThermalVOF3D, IntegralSolverConfig
from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.postprocess.contours import write_integral_midplane_contours
from coldplate_studio.postprocess.vtk_io import write_vtk_unstructured_tets


@dataclass
class Channel3DRunResult:
    conservation_initial: dict[str, float]
    conservation_final: dict[str, float]
    vtk_path: Path
    temperature_contour: Path
    vof_contour: Path
    connectivity_ok: bool
    interface_triangles: int
    fr_element_family: str


def run_channel_3d_case(output_dir: Path) -> Channel3DRunResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_integral_channel_model(mesh_size=0.06)
    graph = build_connectivity_from_extraction(model.extraction)
    diag = run_connectivity_diagnostics(graph)
    iface = build_interface_connectivity(model.mesh)
    mesh = model.mesh
    nodes = mesh.nodes
    n = nodes.shape[0]
    fin_x0 = model.fin_x0
    fin_thick = model.fin_thickness
    fin_h = model.fin_height
    fluid_mask = np.zeros(n, dtype=bool)
    for tet in mesh.fluid_tets:
        fluid_mask[tet] = True
    T0 = np.full(n, 300.0)
    alpha0 = np.zeros(n)
    T0[fluid_mask] = 300.0 + 25.0 * np.sin(np.pi * nodes[fluid_mask, 0]) * np.sin(
        np.pi * nodes[fluid_mask, 1] / 0.2
    ) * np.sin(np.pi * nodes[fluid_mask, 2] / 0.1)
    in_fin = (
        (nodes[:, 0] >= fin_x0)
        & (nodes[:, 0] <= fin_x0 + fin_thick)
        & (nodes[:, 2] <= fin_h + 1e-9)
    )
    alpha0[fluid_mask & ~in_fin] = 0.8
    solver = IntegralScalarThermalVOF3D(
        model,
        IntegralSolverConfig(
            thermal_diffusivity=2.0e-5,
            velocity_m_s=(0.05, 0.0, 0.0),
            phase_change=PhaseChangeConfig(enabled=True, saturation_temperature=320.0, transfer_coefficient=0.02),
            dirichlet_temperature_k=300.0,
        ),
    )
    solver.set_initial_temperature(T0)
    solver.set_initial_vof(alpha0)
    init = solver.conservation_report()
    dt = 0.25
    for _ in range(6):
        solver.step_rk4(dt)
    final = solver.conservation_report()
    vtk_path = output_dir / "channel_integral.vtk"
    write_vtk_unstructured_tets(
        vtk_path,
        mesh.nodes,
        mesh.fluid_tets,
        {"temperature": solver.state.temperature, "vof": solver.state.vof},
    )
    t_png, a_png = write_integral_midplane_contours(
        mesh.nodes,
        solver.state.temperature,
        solver.state.vof,
        output_dir,
    )
    return Channel3DRunResult(
        conservation_initial={
            "liquid_mass_kg": init.liquid_mass_kg,
            "thermal_energy_j": init.total_thermal_energy_j,
            "alpha_min": init.alpha_min,
            "alpha_max": init.alpha_max,
        },
        conservation_final={
            "liquid_mass_kg": final.liquid_mass_kg,
            "thermal_energy_j": final.total_thermal_energy_j,
            "alpha_min": final.alpha_min,
            "alpha_max": final.alpha_max,
            "phase_change_power_w": final.phase_change_power_w,
        },
        vtk_path=vtk_path,
        temperature_contour=t_png,
        vof_contour=a_png,
        connectivity_ok=diag.ok and iface.total_area_m2 > 0,
        interface_triangles=int(mesh.interface_triangles.shape[0]),
        fr_element_family=mesh.fr_element_family,
    )
