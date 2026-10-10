"""Exterior fluid boundary faces vs FSI (spec B5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.mesh.integral_model import IntegralMesh


@dataclass
class BoundaryFace:
    nodes: tuple[int, int, int]
    face_type: str  # exterior_fluid | fluid_solid_interface
    area_m2: float
    centroid: np.ndarray


@dataclass
class BoundaryFaceReport:
    faces: list[BoundaryFace]
    exterior_count: int
    fsi_count: int
    unclassified_count: int


def _iface_edge_set(mesh: IntegralMesh) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for tri in mesh.interface_triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for e in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((a, c)))):
            edges.add(e)
    return edges


def classify_boundary_faces(mesh: IntegralMesh) -> BoundaryFaceReport:
    tets = mesh.fluid_tets
    nodes = mesh.nodes
    iface_edges = _iface_edge_set(mesh)
    local = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    face_count: dict[tuple[int, int, int], int] = {}
    for tet in tets:
        for a, b, c in local:
            key = tuple(sorted((int(tet[a]), int(tet[b]), int(tet[c]))))
            face_count[key] = face_count.get(key, 0) + 1
    faces: list[BoundaryFace] = []
    ext = fsi = uncl = 0
    for key, cnt in face_count.items():
        if cnt != 1:
            continue
        p0, p1, p2 = nodes[key[0]], nodes[key[1]], nodes[key[2]]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        centroid = (p0 + p1 + p2) / 3.0
        edges = (
            tuple(sorted((key[0], key[1]))),
            tuple(sorted((key[1], key[2]))),
            tuple(sorted((key[0], key[2]))),
        )
        on_iface = any(e in iface_edges for e in edges)
        if on_iface:
            ftype = "fluid_solid_interface"
            fsi += 1
        else:
            ftype = "exterior_fluid"
            ext += 1
        faces.append(BoundaryFace(nodes=key, face_type=ftype, area_m2=float(area), centroid=centroid))
    return BoundaryFaceReport(faces=faces, exterior_count=ext, fsi_count=fsi, unclassified_count=uncl)
