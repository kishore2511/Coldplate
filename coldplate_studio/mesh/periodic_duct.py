"""Structured periodic x duct (Gmsh) for developed-channel flow benchmarks."""

from __future__ import annotations

import gmsh
import numpy as np

from coldplate_studio.geometry.extraction import (
    ExtractedRegion,
    FluidExtractionResult,
    RegionKind,
)
from coldplate_studio.geometry.validation import validate_gmsh_model_stats
from coldplate_studio.mesh.integral_model import IntegralMesh, IntegralModel


def build_integral_periodic_duct(
    length: float = 1.0,
    height: float = 0.1,
    width: float = 0.05,
    nx: int = 8,
    ny: int = 4,
    nz: int = 2,
) -> IntegralModel:
    """
    Fluid box with x-periodic mesh (paired inlet/outlet faces).
    Transfinite mesh gives matching node coordinates on x=0 and x=L.
    """
    gmsh.initialize()
    gmsh.model.add("periodic_duct")
    try:
        gmsh.model.occ.addBox(0, 0, 0, length, height, width)
        gmsh.model.occ.synchronize()
        vol_tag = gmsh.model.getEntities(3)[0][1]
        gmsh.model.addPhysicalGroup(3, [vol_tag], name="coolant")
        # Surfaces: Gmsh box ordering — identify xmin/xmax by bounding box.
        xmin_tag = xmax_tag = None
        for dim, stag in gmsh.model.getEntities(2):
            if dim != 2:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, stag)
            cx = 0.5 * (xmin + xmax)
            if abs(cx) < 1e-6 * max(length, 1.0):
                xmin_tag = stag
            if abs(cx - length) < 1e-6 * max(length, 1.0):
                xmax_tag = stag
        if xmin_tag is None or xmax_tag is None:
            raise RuntimeError("Could not identify xmin/xmax surfaces for periodic duct.")
        # Identity transform for translation periodicity in x.
        affine = [1, 0, 0, length, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        gmsh.model.mesh.setPeriodic(2, [xmax_tag], [xmin_tag], affine)
        # Transfinite sizing
        curves = [tag for dim, tag in gmsh.model.getEntities(1) if dim == 1]
        for c in curves:
            gmsh.model.mesh.setTransfiniteCurve(c, max(2, nx if _curve_along_x(c, length, height, width) else ny))
        for dim, stag in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(stag)
        gmsh.model.mesh.setTransfiniteVolume(vol_tag)
        gmsh.model.mesh.generate(3)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = np.asarray(coords, dtype=float).reshape(-1, 3)
        tag_to_index = {int(t): i for i, t in enumerate(node_tags)}
        fluid_tets: list[list[int]] = []
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(3)
        for etype, nodes_flat in zip(elem_types, elem_node_tags):
            if int(etype) != 4:
                continue
            for row in np.asarray(nodes_flat, dtype=int).reshape(-1, 4):
                fluid_tets.append([tag_to_index[int(n)] for n in row])
        fluid_arr = np.asarray(fluid_tets, dtype=int)
        from coldplate_studio.mesh.integral_pipeline import _exterior_fluid_boundary_nodes

        boundary_nodes = _exterior_fluid_boundary_nodes(fluid_arr, nodes, np.zeros((0, 3), dtype=int))
        validation = validate_gmsh_model_stats(
            {
                "volumes": 1,
                "surfaces": len(gmsh.model.getEntities(2)),
                "bbox_diagonal": float((length**2 + height**2 + width**2) ** 0.5),
            },
            min_feature_length=height / ny * 0.25,
        )
        extraction = FluidExtractionResult(
            ok=validation.ok,
            fluid_tags=[vol_tag],
            solid_tags=[],
            regions=[
                ExtractedRegion(tag=vol_tag, kind="fluid", volume_m3=length * height * width, component_id=vol_tag)
            ],
            validation=validation,
            interface_surface_tags=[],
            notes=["Periodic x duct for developed-flow benchmark."],
        )
        mesh = IntegralMesh(
            nodes=nodes,
            fluid_tets=fluid_arr,
            solid_tets=np.zeros((0, 4), dtype=int),
            interface_triangles=np.zeros((0, 3), dtype=int),
            boundary_fluid_nodes=boundary_nodes,
            physical_groups={"coolant": [vol_tag], "periodic_x": [xmin_tag, xmax_tag]},
        )
        return IntegralModel(
            extraction=extraction,
            mesh=mesh,
            fin_x0=0.0,
            fin_thickness=0.0,
            fin_height=0.0,
        )
    finally:
        gmsh.finalize()


def _curve_along_x(curve_tag: int, length: float, height: float, width: float) -> bool:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, curve_tag)
    return abs(xmax - xmin) > max(abs(ymax - ymin), abs(zmax - zmin)) * 0.5
