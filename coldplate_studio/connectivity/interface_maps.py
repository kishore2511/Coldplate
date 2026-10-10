"""Solver-facing fluid-solid interface maps (spec B4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.mesh.integral_model import IntegralMesh


@dataclass
class InterfacePatch:
    patch_id: str
    triangle_node_indices: np.ndarray
    fluid_volume_centroid: np.ndarray | None
    solid_volume_centroid: np.ndarray | None
    area_m2: float


@dataclass
class InterfaceConnectivity:
    patches: list[InterfacePatch]
    total_area_m2: float


def build_interface_connectivity(mesh: IntegralMesh) -> InterfaceConnectivity:
    """Construct interface patches from exported Gmsh triangles."""
    nodes = mesh.nodes
    patches: list[InterfacePatch] = []
    total = 0.0
    for i, tri in enumerate(mesh.interface_triangles):
        p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        total += area
        patches.append(
            InterfacePatch(
                patch_id=f"fsi_{i:04d}",
                triangle_node_indices=tri,
                fluid_volume_centroid=None,
                solid_volume_centroid=None,
                area_m2=float(area),
            )
        )
    return InterfaceConnectivity(patches=patches, total_area_m2=total)
