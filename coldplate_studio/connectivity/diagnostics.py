"""Pre-solve connectivity and boundary diagnostics (spec B5)."""

from __future__ import annotations

from dataclasses import dataclass, field

from coldplate_studio.connectivity.graph import BoundaryClass, ConnectivityGraph


@dataclass
class DiagnosticItem:
    code: str
    message: str
    severity: str


@dataclass
class ConnectivityDiagnostics:
    ok: bool
    items: list[DiagnosticItem] = field(default_factory=list)
    unclassified_faces: int = 0

    def add(self, code: str, message: str, severity: str) -> None:
        self.items.append(DiagnosticItem(code, message, severity))
        if severity == "error":
            self.ok = False


def run_connectivity_diagnostics(
    graph: ConnectivityGraph,
    required_interface: bool = True,
) -> ConnectivityDiagnostics:
    diag = ConnectivityDiagnostics(ok=True)
    fluid_nodes = [n for n, d in graph.graph.nodes(data=True) if d.get("kind") == "fluid"]
    solid_nodes = [n for n, d in graph.graph.nodes(data=True) if d.get("kind") == "solid"]
    if not fluid_nodes:
        diag.add("no_fluid_component", "No fluid volume nodes in connectivity graph.", "error")
    if not solid_nodes:
        diag.add("no_solid_component", "No solid volume nodes in connectivity graph.", "warning")
    iface_nodes = [n for n, d in graph.graph.nodes(data=True) if d.get("kind") == "interface_patch"]
    if required_interface and not iface_nodes:
        diag.add("missing_interface", "Fluid-solid interface patch not present.", "error")
    # Inlet/outlet must be confirmed by user for imported CAD; mark as unclassified here.
    diag.unclassified_faces = 0
    diag.add(
        "inlet_outlet_unset",
        "Inlet/outlet classifications require user confirmation (not inferred from position alone).",
        "warning",
    )
    return diag
