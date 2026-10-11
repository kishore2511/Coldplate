"""User-confirmed port definitions — no BC inference from position alone (spec B5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from coldplate_studio.mesh.boundary_faces import BoundaryFace, classify_boundary_faces
from coldplate_studio.mesh.integral_model import IntegralMesh


class PortKind(str, Enum):
    INLET = "inlet"
    OUTLET = "outlet"
    WALL = "wall"
    SYMMETRY = "symmetry"
    UNCLASSIFIED = "unclassified"


@dataclass
class UserPortDefinition:
    """User-assigned semantic label for a set of boundary faces."""

    name: str
    kind: PortKind
    face_node_keys: list[tuple[int, int, int]]  # sorted node triples


@dataclass
class SemanticBoundaryReport:
    labeled_faces: dict[tuple[int, int, int], PortKind]
    unclassified_exterior_faces: list[tuple[int, int, int]]
    duplicate_labels: list[tuple[int, int, int]]
    ok_for_flow_solve: bool
    notes: list[str] = field(default_factory=list)


def apply_user_port_definitions(
    mesh: IntegralMesh,
    ports: list[UserPortDefinition],
) -> SemanticBoundaryReport:
    boundary = classify_boundary_faces(mesh)
    exterior_keys = {f.nodes for f in boundary.faces if f.face_type == "exterior_fluid"}
    labeled: dict[tuple[int, int, int], PortKind] = {}
    duplicates: list[tuple[int, int, int]] = []
    for port in ports:
        for key in port.face_node_keys:
            if key not in exterior_keys:
                continue
            if key in labeled:
                duplicates.append(key)
            labeled[key] = port.kind
    unclassified = sorted(exterior_keys - set(labeled.keys()))
    has_inlet = any(k == PortKind.INLET for k in labeled.values())
    has_outlet = any(k == PortKind.OUTLET for k in labeled.values())
    ok = has_inlet and has_outlet and not duplicates
    notes = []
    if not has_inlet:
        notes.append("No user-confirmed inlet faces assigned.")
    if not has_outlet:
        notes.append("No user-confirmed outlet faces assigned.")
    if duplicates:
        notes.append("Duplicate port labels on the same face.")
    return SemanticBoundaryReport(
        labeled_faces=labeled,
        unclassified_exterior_faces=unclassified,
        duplicate_labels=duplicates,
        ok_for_flow_solve=ok,
        notes=notes,
    )


def channel_ports_from_user_confirmation(
    mesh: IntegralMesh,
    inlet_face_keys: list[tuple[int, int, int]],
    outlet_face_keys: list[tuple[int, int, int]],
) -> SemanticBoundaryReport:
    """Explicit user confirmation for channel inlet/outlet face lists."""
    ports = [
        UserPortDefinition("inlet", PortKind.INLET, inlet_face_keys),
        UserPortDefinition("outlet", PortKind.OUTLET, outlet_face_keys),
    ]
    return apply_user_port_definitions(mesh, ports)
