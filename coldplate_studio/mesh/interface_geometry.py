"""Fluid-solid interface normals and orientation (spec B4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.mesh.integral_model import IntegralMesh


@dataclass
class OrientedInterfaceTriangle:
    nodes: np.ndarray  # (3,) indices
    area_m2: float
    normal: np.ndarray  # unit, fluid → solid
    fluid_centroid: np.ndarray
    solid_centroid: np.ndarray


@dataclass
class InterfaceGeometryReport:
    triangles: list[OrientedInterfaceTriangle]
    total_area_m2: float
    duplicate_triangle_keys: list[tuple[int, int, int]]
    orientation_ok: bool


def _tri_key(nodes: np.ndarray) -> tuple[int, int, int]:
    return tuple(sorted(int(n) for n in nodes))


def _fluid_solid_centroids(mesh: IntegralMesh) -> tuple[np.ndarray, np.ndarray]:
    fluid_pts = mesh.nodes[mesh.fluid_tets].mean(axis=1).mean(axis=0)
    if mesh.solid_tets.size:
        solid_pts = mesh.nodes[mesh.solid_tets].mean(axis=1).mean(axis=0)
    else:
        solid_pts = fluid_pts + np.array([0.0, 0.0, 1.0])
    return fluid_pts, solid_pts


def analyze_interface_geometry(mesh: IntegralMesh) -> InterfaceGeometryReport:
    nodes = mesh.nodes
    fluid_c, solid_c = _fluid_solid_centroids(mesh)
    ref_dir = solid_c - fluid_c
    if np.linalg.norm(ref_dir) < 1e-12:
        ref_dir = np.array([0.0, 0.0, 1.0])
    ref_dir = ref_dir / np.linalg.norm(ref_dir)
    seen: dict[tuple[int, int, int], int] = {}
    duplicates: list[tuple[int, int, int]] = []
    tris: list[OrientedInterfaceTriangle] = []
    total = 0.0
    orientation_ok = True
    for tri in mesh.interface_triangles:
        key = _tri_key(tri)
        if key in seen:
            duplicates.append(key)
        seen[key] = seen.get(key, 0) + 1
        p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        n_vec = np.cross(p1 - p0, p2 - p0)
        area = 0.5 * np.linalg.norm(n_vec)
        total += area
        if area > 1e-14:
            n_hat = n_vec / (2.0 * area)
            if np.dot(n_hat, ref_dir) < 0:
                n_hat = -n_hat
            if np.dot(n_hat, ref_dir) < 0.5:
                orientation_ok = False
        else:
            n_hat = ref_dir
        tris.append(
            OrientedInterfaceTriangle(
                nodes=tri,
                area_m2=float(area),
                normal=n_hat,
                fluid_centroid=fluid_c,
                solid_centroid=solid_c,
            )
        )
    return InterfaceGeometryReport(
        triangles=tris,
        total_area_m2=total,
        duplicate_triangle_keys=duplicates,
        orientation_ok=orientation_ok,
    )
