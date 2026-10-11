"""Pre-solve workflow gates: extraction, connectivity, interfaces (spec B5)."""

from __future__ import annotations

from dataclasses import dataclass, field

from coldplate_studio.connectivity.diagnostics import ConnectivityDiagnostics, run_connectivity_diagnostics
from coldplate_studio.connectivity.graph import ConnectivityGraph, build_connectivity_from_extraction
from coldplate_studio.connectivity.interface_maps import InterfaceConnectivity, build_interface_connectivity
from coldplate_studio.geometry.extraction import FluidExtractionResult
from coldplate_studio.mesh.fluid_connectivity import FluidConnectivityReport, analyze_fluid_connectivity
from coldplate_studio.mesh.integral_model import IntegralModel
from coldplate_studio.mesh.interface_geometry import analyze_interface_geometry
from coldplate_studio.mesh.quality_gmsh import check_mesh_validity


@dataclass
class PreflightItem:
    code: str
    message: str
    severity: str


@dataclass
class PreflightReport:
    ok: bool
    items: list[PreflightItem] = field(default_factory=list)

    def add(self, code: str, message: str, severity: str) -> None:
        self.items.append(PreflightItem(code, message, severity))
        if severity == "error":
            self.ok = False


def run_integral_preflight(
    model: IntegralModel,
    require_single_fluid_component: bool = True,
    require_positive_interface_area: bool = True,
) -> PreflightReport:
    report = PreflightReport(ok=True)
    ext = model.extraction
    if not ext.ok:
        report.add("extraction_failed", "Fluid extraction validation failed.", "error")
    graph = build_connectivity_from_extraction(ext)
    conn: ConnectivityDiagnostics = run_connectivity_diagnostics(graph)
    for item in conn.items:
        report.add(item.code, item.message, item.severity)
    iface: InterfaceConnectivity = build_interface_connectivity(model.mesh)
    if require_positive_interface_area and iface.total_area_m2 <= 0:
        report.add("zero_interface_area", "Fluid-solid interface area is zero.", "error")
    fluid: FluidConnectivityReport = analyze_fluid_connectivity(model.mesh)
    if fluid.n_components == 0:
        report.add("no_fluid_tets", "No fluid tetrahedra in integral mesh.", "error")
    elif require_single_fluid_component and fluid.n_components != 1:
        report.add(
            "disconnected_fluid",
            f"Found {fluid.n_components} disconnected fluid components (expected 1 for channel).",
            "error",
        )
    igeom = analyze_interface_geometry(model.mesh)
    if igeom.duplicate_triangle_keys:
        report.add(
            "duplicate_interface_triangles",
            f"Found {len(igeom.duplicate_triangle_keys)} duplicate interface triangles.",
            "warning",
        )
    if not igeom.orientation_ok:
        report.add("interface_orientation", "Interface normal orientation check failed.", "warning")
    validity = check_mesh_validity(model.mesh)
    if not validity.ok:
        report.add(
            "mesh_invalid",
            f"Mesh validity failed: {validity.negative_volume_count} non-positive tets.",
            "error",
        )
    return report
