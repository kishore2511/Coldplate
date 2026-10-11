"""Structured two-material bar with conforming FSI interface (verification)."""

from __future__ import annotations

import numpy as np

from coldplate_studio.geometry.extraction import ExtractedRegion, FluidExtractionResult
from coldplate_studio.geometry.validation import validate_gmsh_model_stats
from coldplate_studio.mesh.integral_model import IntegralMesh, IntegralModel
from coldplate_studio.mesh.structured_tet_box import _node_index


def build_two_material_tet_bar(
    length: float = 1.0,
    height: float = 0.1,
    width: float = 0.1,
    nx: int = 12,
    ny: int = 2,
    nz: int = 2,
    *,
    split_x: float = 0.5,
) -> IntegralModel:
    """
    Box split at x=split_x: x < split_x fluid tets, x >= split_x solid tets.
    Interface triangles lie on the plane x=split_x.
    """
    x = np.linspace(0.0, length, nx + 1)
    y = np.linspace(0.0, height, ny + 1)
    z = np.linspace(0.0, width, nz + 1)
    nodes = np.array([[xi, yj, zk] for zk in z for yj in y for xi in x], dtype=float)
    fluid_tets: list[list[int]] = []
    solid_tets: list[list[int]] = []
    split_i = int(round(split_x / length * nx))
    split_i = max(1, min(nx - 1, split_i))
    x_split = x[split_i]

    def add_hex_tets(i, j, k, target: list[list[int]]) -> None:
        n000 = _node_index(i, j, k, nx, ny)
        n100 = _node_index(i + 1, j, k, nx, ny)
        n010 = _node_index(i, j + 1, k, nx, ny)
        n110 = _node_index(i + 1, j + 1, k, nx, ny)
        n001 = _node_index(i, j, k + 1, nx, ny)
        n101 = _node_index(i + 1, j, k + 1, nx, ny)
        n011 = _node_index(i, j + 1, k + 1, nx, ny)
        n111 = _node_index(i + 1, j + 1, k + 1, nx, ny)
        target.extend(
            [
                [n000, n100, n010, n001],
                [n100, n110, n010, n111],
                [n100, n010, n001, n111],
                [n010, n001, n011, n111],
                [n100, n001, n101, n111],
            ]
        )

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if i < split_i:
                    add_hex_tets(i, j, k, fluid_tets)
                else:
                    add_hex_tets(i, j, k, solid_tets)

    interface_tris: list[list[int]] = []
    for k in range(nz):
        for j in range(ny):
            i = split_i
            n0 = _node_index(i, j, k, nx, ny)
            n1 = _node_index(i, j + 1, k, nx, ny)
            n2 = _node_index(i, j, k + 1, nx, ny)
            n3 = _node_index(i, j + 1, k + 1, nx, ny)
            interface_tris.append([n0, n1, n2])
            interface_tris.append([n1, n3, n2])

    boundary: set[int] = set()
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                if i == 0 or i == nx or j == 0 or j == ny or k == 0 or k == nz:
                    boundary.add(_node_index(i, j, k, nx, ny))

    validation = validate_gmsh_model_stats(
        {"volumes": 2, "surfaces": 7, "bbox_diagonal": float((length**2 + height**2 + width**2) ** 0.5)},
        min_feature_length=height / ny * 0.25,
    )
    extraction = FluidExtractionResult(
        ok=validation.ok,
        fluid_tags=[1],
        solid_tags=[2],
        regions=[
            ExtractedRegion(tag=1, kind="fluid", volume_m3=split_x * height * width, component_id=1),
            ExtractedRegion(tag=2, kind="solid", volume_m3=(length - split_x) * height * width, component_id=2),
        ],
        validation=validation,
        interface_surface_tags=[99],
        notes=["Two-material structured bar for FSI conduction verification."],
    )
    mesh = IntegralMesh(
        nodes=nodes,
        fluid_tets=np.asarray(fluid_tets, dtype=int),
        solid_tets=np.asarray(solid_tets, dtype=int),
        interface_triangles=np.asarray(interface_tris, dtype=int),
        boundary_fluid_nodes=np.asarray(sorted(boundary), dtype=int),
    )
    return IntegralModel(
        extraction=extraction,
        mesh=mesh,
        fin_x0=0.0,
        fin_thickness=0.0,
        fin_height=0.0,
    )
