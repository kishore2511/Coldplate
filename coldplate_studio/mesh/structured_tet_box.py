"""Structured tet mesh on a box (for verification benchmarks)."""

from __future__ import annotations

import numpy as np

from coldplate_studio.geometry.extraction import (
    ExtractedRegion,
    FluidExtractionResult,
    RegionKind,
)
from coldplate_studio.geometry.validation import validate_gmsh_model_stats
from coldplate_studio.mesh.integral_model import IntegralMesh, IntegralModel


def _node_index(i: int, j: int, k: int, nx: int, ny: int) -> int:
    return i + j * (nx + 1) + k * (nx + 1) * (ny + 1)


def build_structured_fluid_box(
    length: float = 1.0,
    height: float = 0.1,
    width: float = 0.2,
    nx: int = 10,
    ny: int = 4,
    nz: int = 2,
) -> IntegralModel:
    """Hexahedral grid split into 5 tets per cell (consistent orientation)."""
    x = np.linspace(0.0, length, nx + 1)
    y = np.linspace(0.0, height, ny + 1)
    z = np.linspace(0.0, width, nz + 1)
    nodes = np.array([[xi, yj, zk] for zk in z for yj in y for xi in x], dtype=float)
    tets: list[list[int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n000 = _node_index(i, j, k, nx, ny)
                n100 = _node_index(i + 1, j, k, nx, ny)
                n010 = _node_index(i, j + 1, k, nx, ny)
                n110 = _node_index(i + 1, j + 1, k, nx, ny)
                n001 = _node_index(i, j, k + 1, nx, ny)
                n101 = _node_index(i + 1, j, k + 1, nx, ny)
                n011 = _node_index(i, j + 1, k + 1, nx, ny)
                n111 = _node_index(i + 1, j + 1, k + 1, nx, ny)
                # Kuhn triangulation of hex
                tets.extend(
                    [
                        [n000, n100, n010, n001],
                        [n100, n110, n010, n111],
                        [n100, n010, n001, n111],
                        [n010, n001, n011, n111],
                        [n100, n001, n101, n111],
                    ]
                )
    fluid_arr = np.asarray(tets, dtype=int)
    boundary: set[int] = set()
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                if i == 0 or i == nx or j == 0 or j == ny or k == 0 or k == nz:
                    boundary.add(_node_index(i, j, k, nx, ny))
    validation = validate_gmsh_model_stats(
        {"volumes": 1, "surfaces": 6, "bbox_diagonal": float((length**2 + height**2 + width**2) ** 0.5)},
        min_feature_length=height / ny * 0.25,
    )
    extraction = FluidExtractionResult(
        ok=validation.ok,
        fluid_tags=[1],
        solid_tags=[],
        regions=[ExtractedRegion(tag=1, kind="fluid", volume_m3=length * height * width, component_id=1)],
        validation=validation,
        interface_surface_tags=[],
        notes=["Structured tet box for verification."],
    )
    mesh = IntegralMesh(
        nodes=nodes,
        fluid_tets=fluid_arr,
        solid_tets=np.zeros((0, 4), dtype=int),
        interface_triangles=np.zeros((0, 3), dtype=int),
        boundary_fluid_nodes=np.asarray(sorted(boundary), dtype=int),
        physical_groups={"coolant": [1]},
    )
    return IntegralModel(extraction=extraction, mesh=mesh, fin_x0=0.0, fin_thickness=0.0, fin_height=0.0)
