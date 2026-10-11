"""Global DOF numbering for P2 velocity / P1 pressure on fluid tet subdomains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.fr.tet_p2_basis import TET_P2_EDGE_PAIRS
from coldplate_studio.mesh.integral_model import IntegralMesh


@dataclass(frozen=True)
class P2VelocityDofMap:
    n_mesh_nodes: int
    n_velocity_dofs: int
    vertex_dof: np.ndarray  # (n_mesh_nodes,) -1 if not a velocity dof
    edge_dof: dict[tuple[int, int], int]  # sorted node pair -> dof index
    fluid_vertex_mask: np.ndarray  # (n_mesh_nodes,) bool

    def local_dof_indices(self, tet: np.ndarray) -> np.ndarray:
        """10 local velocity DOF indices for tet node indices tet[0..3]."""
        out = np.zeros(10, dtype=int)
        for i in range(4):
            out[i] = int(self.vertex_dof[int(tet[i])])
        for e, (a, b) in enumerate(TET_P2_EDGE_PAIRS):
            key = (int(tet[a]), int(tet[b]))
            if key[0] > key[1]:
                key = (key[1], key[0])
            out[4 + e] = int(self.edge_dof[key])
        return out


def build_p2_velocity_dof_map(mesh: IntegralMesh, fluid_vertex_mask: np.ndarray) -> P2VelocityDofMap:
    """
    Velocity unknowns on fluid-region vertices and edges (edges with both endpoints fluid).
    """
    n = mesh.nodes.shape[0]
    vertex_dof = np.full(n, -1, dtype=int)
    edge_dof: dict[tuple[int, int], int] = {}
    nv = 0
    for i in range(n):
        if fluid_vertex_mask[i]:
            vertex_dof[i] = nv
            nv += 1
    ne = 0
    for tet in mesh.fluid_tets:
        for a, b in TET_P2_EDGE_PAIRS:
            i, j = int(tet[a]), int(tet[b])
            if not (fluid_vertex_mask[i] and fluid_vertex_mask[j]):
                continue
            key = (i, j) if i < j else (j, i)
            if key not in edge_dof:
                edge_dof[key] = nv + ne
                ne += 1
    return P2VelocityDofMap(
        n_mesh_nodes=n,
        n_velocity_dofs=nv + ne,
        vertex_dof=vertex_dof,
        edge_dof=edge_dof,
        fluid_vertex_mask=fluid_vertex_mask,
    )


def pressure_dofs_on_fluid_vertices(fluid_vertex_mask: np.ndarray) -> np.ndarray:
    """Indices of fluid vertices used as P1 pressure nodes (same order as mesh node id)."""
    return np.array(sorted(np.where(fluid_vertex_mask)[0]), dtype=int)
