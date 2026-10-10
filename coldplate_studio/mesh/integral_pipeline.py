"""Integral workflow: CAD Boolean → labeled mesh → solver-ready IntegralModel (one Gmsh build)."""

from __future__ import annotations

import gmsh
import numpy as np

from coldplate_studio.geometry.extraction import (
    ExtractedRegion,
    FluidExtractionResult,
    RegionKind,
)
from coldplate_studio.geometry.validation import GeometryValidationReport, validate_gmsh_model_stats
from coldplate_studio.mesh.integral_model import IntegralMesh, IntegralModel


def build_integral_channel_model(
    length: float = 1.0,
    width: float = 0.2,
    height: float = 0.1,
    fin_thickness: float = 0.02,
    fin_height: float = 0.05,
    mesh_size: float = 0.06,
) -> IntegralModel:
    """
    Single-pass procedural cold plate:
    OCC fragment → physical groups → volume mesh → fluid/solid split → boundary tags.
    """
    fin_x0 = 0.5 * (length - fin_thickness)
    gmsh.initialize()
    gmsh.model.add("integral_channel")
    try:
        env = gmsh.model.occ.addBox(0, 0, 0, length, width, height)
        fin = gmsh.model.occ.addBox(fin_x0, 0, 0, fin_thickness, width, fin_height)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.fragment([(3, env)], [(3, fin)])
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        fluid_tags: list[int] = []
        solid_tags: list[int] = []
        regions: list[ExtractedRegion] = []
        for dim, tag in volumes:
            if dim != 3:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            vol = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
            is_fin = (
                abs((xmax + xmin) - (2 * fin_x0 + fin_thickness)) < 1e-6
                and abs(zmax - fin_height) < 1e-6
                and abs(zmin) < 1e-6
            )
            if is_fin:
                solid_tags.append(tag)
                kind: RegionKind = "solid"
            else:
                fluid_tags.append(tag)
                kind = "fluid"
            regions.append(ExtractedRegion(tag=tag, kind=kind, volume_m3=vol, component_id=tag))
        gmsh.model.addPhysicalGroup(3, fluid_tags, name="coolant")
        gmsh.model.addPhysicalGroup(3, solid_tags, name="fin_solid")
        fluid_set = set(fluid_tags)
        solid_set = set(solid_tags)
        interface_tags: list[int] = []
        for _, stag in gmsh.model.getEntities(2):
            vol_adj, _ = gmsh.model.getAdjacencies(2, stag)
            if set(vol_adj) & fluid_set and set(vol_adj) & solid_set:
                interface_tags.append(stag)
        gmsh.model.addPhysicalGroup(2, interface_tags, name="fluid_solid_interface")
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.mesh.generate(3)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = np.asarray(coords, dtype=float).reshape(-1, 3)
        tag_to_index = {int(t): i for i, t in enumerate(node_tags)}
        fluid_tets: list[list[int]] = []
        solid_tets: list[list[int]] = []
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(3)
        for etype, nodes_flat in zip(elem_types, elem_node_tags):
            if int(etype) != 4:
                continue
            for row in np.asarray(nodes_flat, dtype=int).reshape(-1, 4):
                tet = [tag_to_index[int(n)] for n in row]
                c = nodes[tet].mean(axis=0)
                in_fin = (
                    fin_x0 <= c[0] <= fin_x0 + fin_thickness
                    and c[2] <= fin_height + 1e-9
                )
                (solid_tets if in_fin else fluid_tets).append(tet)
        interface_tris: list[list[int]] = []
        for _, stag in gmsh.model.getEntities(2):
            vol_adj, _ = gmsh.model.getAdjacencies(2, stag)
            if set(vol_adj) & fluid_set and set(vol_adj) & solid_set:
                _, _, face_nodes = gmsh.model.mesh.getElements(2, stag)
                if not face_nodes:
                    continue
                for row in np.asarray(face_nodes[0], dtype=int).reshape(-1, 3):
                    interface_tris.append([tag_to_index[int(n)] for n in row])
        fluid_arr = np.asarray(fluid_tets, dtype=int)
        solid_arr = np.asarray(solid_tets, dtype=int) if solid_tets else np.zeros((0, 4), dtype=int)
        iface_arr = np.asarray(interface_tris, dtype=int) if interface_tris else np.zeros((0, 3), dtype=int)
        boundary_nodes = _exterior_fluid_boundary_nodes(fluid_arr, nodes, iface_arr)
        stats = {
            "volumes": len(volumes),
            "surfaces": len(gmsh.model.getEntities(2)),
            "bbox_diagonal": float((length**2 + width**2 + height**2) ** 0.5),
        }
        validation = validate_gmsh_model_stats(stats, min_feature_length=mesh_size * 0.25)
        if not fluid_tags:
            validation.add("no_fluid", "Boolean extraction produced no fluid volumes.", "error")
        if not interface_tags:
            validation.add("no_interface", "No shared fluid-solid interface surfaces detected.", "error")
        extraction = FluidExtractionResult(
            ok=validation.ok,
            fluid_tags=fluid_tags,
            solid_tags=solid_tags,
            regions=regions,
            validation=validation,
            interface_surface_tags=interface_tags,
            notes=["Integral pipeline: single Gmsh build for mesh + labels."],
        )
        mesh = IntegralMesh(
            nodes=nodes,
            fluid_tets=fluid_arr,
            solid_tets=solid_arr,
            interface_triangles=iface_arr,
            boundary_fluid_nodes=boundary_nodes,
            physical_groups={
                "coolant": fluid_tags,
                "fin_solid": solid_tags,
                "fluid_solid_interface": interface_tags,
            },
        )
        return IntegralModel(
            extraction=extraction,
            mesh=mesh,
            fin_x0=fin_x0,
            fin_thickness=fin_thickness,
            fin_height=fin_height,
        )
    finally:
        gmsh.finalize()


def _exterior_fluid_boundary_nodes(
    fluid_tets: np.ndarray,
    nodes: np.ndarray,
    interface_tris: np.ndarray,
) -> np.ndarray:
    """Nodes on coolant exterior faces (exclude fluid–solid interface)."""
    if fluid_tets.size == 0:
        return np.zeros(0, dtype=int)
    iface_edges = set()
    for tri in interface_tris:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            iface_edges.add(tuple(sorted((int(a), int(b)))))
    face_count: dict[tuple[int, int, int], int] = {}
    local_faces = (
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    )
    for tet in fluid_tets:
        for a, b, c in local_faces:
            key = tuple(sorted((int(tet[a]), int(tet[b]), int(tet[c]))))
            face_count[key] = face_count.get(key, 0) + 1
    boundary_nodes: set[int] = set()
    for face, count in face_count.items():
        if count != 1:
            continue
        edges = (tuple(sorted((face[0], face[1]))), tuple(sorted((face[1], face[2]))), tuple(sorted((face[0], face[2]))))
        if any(e in iface_edges for e in edges):
            continue
        boundary_nodes.update(face)
    return np.asarray(sorted(boundary_nodes), dtype=int)
