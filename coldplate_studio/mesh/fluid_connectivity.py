"""Connected fluid-volume analysis on integral tet meshes (spec B2–B3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.mesh.integral_model import IntegralMesh


@dataclass
class FluidConnectivityReport:
    n_components: int
    component_ids: np.ndarray  # per fluid tet
    component_volumes: list[float]
    largest_component_fraction: float


def _face_key(tet: np.ndarray, local_face: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(sorted(int(tet[i]) for i in local_face))


def analyze_fluid_connectivity(mesh: IntegralMesh) -> FluidConnectivityReport:
    """Label connected fluid regions via shared tet faces."""
    tets = mesh.fluid_tets
    if tets.size == 0:
        return FluidConnectivityReport(0, np.zeros(0, dtype=int), [], 0.0)
    local_faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    face_owner: dict[tuple[int, int, int], list[int]] = {}
    for e, tet in enumerate(tets):
        for lf in local_faces:
            key = _face_key(tet, lf)
            face_owner.setdefault(key, []).append(e)
    parent = list(range(len(tets)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for owners in face_owner.values():
        if len(owners) == 2:
            union(owners[0], owners[1])
    comp_id = np.array([find(i) for i in range(len(tets))], dtype=int)
    unique = sorted(set(comp_id.tolist()))
    remap = {old: i for i, old in enumerate(unique)}
    component_ids = np.array([remap[c] for c in comp_id], dtype=int)
    nodes = mesh.nodes
    volumes = []
    for cid in range(len(unique)):
        mask = component_ids == cid
        vol = 0.0
        for tet in tets[mask]:
            a, b, c, d = nodes[tet[0]], nodes[tet[1]], nodes[tet[2]], nodes[tet[3]]
            vol += abs(np.linalg.det(np.column_stack((b - a, c - a, d - a)))) / 6.0
        volumes.append(vol)
    total = sum(volumes) or 1.0
    largest = max(volumes) / total if volumes else 0.0
    return FluidConnectivityReport(
        n_components=len(unique),
        component_ids=component_ids,
        component_volumes=volumes,
        largest_component_fraction=largest,
    )
