"""Discover conforming fluid–solid interface triangles from region connectivity."""

from __future__ import annotations

import numpy as np

from coldplate_studio.mesh.integral_model import IntegralMesh


def discover_interface_triangles(mesh: IntegralMesh) -> np.ndarray:
    """Return (k, 3) node indices for interior faces shared by one fluid and one solid tet."""
    local = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))

    def face_map(tets: np.ndarray) -> dict[tuple[int, int, int], int]:
        counts: dict[tuple[int, int, int], int] = {}
        for tet in tets:
            for a, b, c in local:
                key = tuple(sorted((int(tet[a]), int(tet[b]), int(tet[c]))))
                counts[key] = counts.get(key, 0) + 1
        return counts

    fluid_interior = face_map(mesh.fluid_tets)
    solid_interior = face_map(mesh.solid_tets)
    shared: list[tuple[int, int, int]] = []
    for key, cf in fluid_interior.items():
        if cf != 1:
            continue
        if solid_interior.get(key, 0) != 1:
            continue
        shared.append(key)
    if not shared:
        return np.zeros((0, 3), dtype=int)
    return np.asarray([list(k) for k in sorted(shared)], dtype=int)
