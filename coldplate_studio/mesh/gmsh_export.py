"""Export Gmsh fluid/solid meshes for visualization and connectivity checks."""

from __future__ import annotations

from dataclasses import dataclass

import gmsh
import numpy as np


@dataclass
class TetMesh:
    nodes: np.ndarray  # (n, 3)
    elements: np.ndarray  # (m, 4) zero-based node indices
    fluid_element_mask: np.ndarray  # (m,) bool
    interface_triangles: np.ndarray  # (k, 3) node indices on fluid-solid interface


def export_channel_tet_mesh(
    length: float = 1.0,
    width: float = 0.2,
    height: float = 0.1,
    fin_thickness: float = 0.02,
    fin_height: float = 0.05,
    mesh_size: float = 0.05,
) -> TetMesh:
    """Build channel geometry and return coolant-region tetrahedra (fin excluded)."""
    gmsh.initialize()
    try:
        gmsh.model.add("export")
        env = gmsh.model.occ.addBox(0, 0, 0, length, width, height)
        fin_x0 = 0.5 * (length - fin_thickness)
        fin = gmsh.model.occ.addBox(fin_x0, 0, 0, fin_thickness, width, fin_height)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.fragment([(3, env)], [(3, fin)])
        gmsh.model.occ.synchronize()
        fluid_tags: list[int] = []
        solid_tags: list[int] = []
        for dim, tag in gmsh.model.getEntities(3):
            xmin, _, zmin, xmax, _, zmax = gmsh.model.getBoundingBox(dim, tag)
            is_fin = (
                abs((xmax + xmin) - (2 * fin_x0 + fin_thickness)) < 1e-6
                and abs(zmax - fin_height) < 1e-6
                and abs(zmin) < 1e-6
            )
            (solid_tags if is_fin else fluid_tags).append(tag)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.mesh.generate(3)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        tag_to_index = {int(t): i for i, t in enumerate(node_tags)}
        tets: list[list[int]] = []
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(3)
        for etype, nodes_flat in zip(elem_types, elem_node_tags):
            if int(etype) != 4:
                continue
            nodes = np.asarray(nodes_flat, dtype=int).reshape(-1, 4)
            for row in nodes:
                tets.append([tag_to_index[int(n)] for n in row])
        tets_arr = np.asarray(tets, dtype=int)
        centroids = coords[tets_arr].mean(axis=1)
        fin_mask = (
            (centroids[:, 0] >= fin_x0)
            & (centroids[:, 0] <= fin_x0 + fin_thickness)
            & (centroids[:, 2] <= fin_height + 1e-9)
        )
        fluid_mask = ~fin_mask
        fluid_set = set(fluid_tags)
        solid_set = set(solid_tags)
        interface_tris: list[list[int]] = []
        for _, stag in gmsh.model.getEntities(2):
            vol_adj, _ = gmsh.model.getAdjacencies(2, stag)
            if set(vol_adj) & fluid_set and set(vol_adj) & solid_set:
                etypes, _, node_tags_face = gmsh.model.mesh.getElements(2, stag)
                if not etypes:
                    continue
                for nodes_flat in node_tags_face:
                    tris = np.asarray(nodes_flat, dtype=int).reshape(-1, 3)
                    for row in tris:
                        interface_tris.append([tag_to_index[int(n)] for n in row])
        return TetMesh(
            nodes=coords,
            elements=tets_arr,
            fluid_element_mask=fluid_mask,
            interface_triangles=np.asarray(interface_tris, dtype=int) if interface_tris else np.zeros((0, 3), dtype=int),
        )
    finally:
        gmsh.finalize()
