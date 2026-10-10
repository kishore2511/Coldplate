"""End-to-end 3D channel thermal + VOF demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coldplate_studio.connectivity.diagnostics import run_connectivity_diagnostics
from coldplate_studio.connectivity.graph import build_connectivity_from_extraction
from coldplate_studio.fr.solver3d import PhaseChangeConfig, ScalarThermalVOF3D, Solver3DConfig
from coldplate_studio.geometry.extraction import extract_channel_fluid_domain
from coldplate_studio.mesh.gmsh_export import export_channel_tet_mesh
from coldplate_studio.postprocess.contours import write_midplane_contours
from coldplate_studio.postprocess.vtk_io import write_vtk_structured, write_vtk_unstructured_tets


@dataclass
class Channel3DRunResult:
    conservation_initial: dict[str, float]
    conservation_final: dict[str, float]
    vtk_structured: Path
    vtk_tet: Path
    temperature_contour: Path
    vof_contour: Path
    connectivity_ok: bool
    interface_triangles: int


def run_channel_3d_case(output_dir: Path) -> Channel3DRunResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = (0.0, 1.0, 0.0, 0.2, 0.0, 0.1)
    extraction = extract_channel_fluid_domain(1.0, 0.2, 0.1, 0.02, 0.05, mesh_size=0.06)
    graph = build_connectivity_from_extraction(extraction)
    diag = run_connectivity_diagnostics(graph)
    tet = export_channel_tet_mesh(mesh_size=0.06)
    cfg = Solver3DConfig(
        ne_x=4,
        ne_y=2,
        ne_z=2,
        degree=2,
        bounds=bounds,
        thermal_diffusivity=2.0e-5,
        velocity_m_s=(0.05, 0.0, 0.0),
        phase_change=PhaseChangeConfig(enabled=True, saturation_temperature=320.0, transfer_coefficient=0.02),
    )
    solver = ScalarThermalVOF3D(cfg)
    x, y, z = solver.state.x, solver.state.y, solver.state.z
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    fin_x0 = 0.5 * (1.0 - 0.02)
    fin_mask = (X >= fin_x0) & (X <= fin_x0 + 0.02) & (Z <= 0.05)
    T0 = 300.0 + 25.0 * np.sin(np.pi * X) * np.sin(np.pi * Y / 0.2) * np.sin(np.pi * Z / 0.1)
    T0[fin_mask] = 300.0
    alpha0 = np.where(fin_mask, 0.0, 0.8)
    solver.set_initial_temperature(T0)
    solver.set_initial_vof(alpha0)
    solver.apply_dirichlet_temperature(300.0)
    init = solver.conservation_report()
    dt = 0.25
    for _ in range(6):
        solver.step_rk4(dt)
    final = solver.conservation_report()
    vtk_s = output_dir / "channel_fr_structured.vtk"
    write_vtk_structured(
        vtk_s,
        x,
        y,
        z,
        {"temperature": solver.state.temperature, "vof": solver.state.vof},
    )
    # Sample FR fields onto Gmsh nodes (nearest grid point) for tet visualization.
    nodes = tet.nodes
    ix = np.clip(np.searchsorted(x, nodes[:, 0]) - 1, 0, len(x) - 1)
    iy = np.clip(np.searchsorted(y, nodes[:, 1]) - 1, 0, len(y) - 1)
    iz = np.clip(np.searchsorted(z, nodes[:, 2]) - 1, 0, len(z) - 1)
    t_nodes = solver.state.temperature[ix, iy, iz]
    a_nodes = solver.state.vof[ix, iy, iz]
    fluid_tets = tet.elements[tet.fluid_element_mask]
    vtk_t = output_dir / "channel_gmsh_fluid.vtk"
    write_vtk_unstructured_tets(vtk_t, tet.nodes, fluid_tets, {"temperature": t_nodes, "vof": a_nodes})
    t_png, a_png = write_midplane_contours(x, y, z, solver.state.temperature, solver.state.vof, output_dir)
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
        vtk_structured=vtk_s,
        vtk_tet=vtk_t,
        temperature_contour=t_png,
        vof_contour=a_png,
        connectivity_ok=diag.ok,
        interface_triangles=int(tet.interface_triangles.shape[0]),
    )
