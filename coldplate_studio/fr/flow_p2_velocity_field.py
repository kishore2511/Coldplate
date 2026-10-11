"""P2 velocity coefficient field for flow → CHT coupling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.fr.flow_p2_dof_map import P2VelocityDofMap
from coldplate_studio.fr.tet_p2_basis import p2_shape_values


@dataclass(frozen=True)
class P2VelocityField:
    dof_map: P2VelocityDofMap
    coefficients: np.ndarray  # length 3 * n_velocity_dofs

    def evaluate_on_tet(self, local_dof_indices: np.ndarray, barycentric: np.ndarray) -> np.ndarray:
        phi = p2_shape_values(barycentric)
        u = np.zeros(3)
        for a in range(10):
            ia = int(local_dof_indices[a])
            for comp in range(3):
                u[comp] += self.coefficients[3 * ia + comp] * phi[a]
        return u

    def vertex_values(self, n_nodes: int) -> np.ndarray:
        vel = np.zeros((n_nodes, 3))
        for i in range(n_nodes):
            vd = int(self.dof_map.vertex_dof[i])
            if vd < 0:
                continue
            for comp in range(3):
                vel[i, comp] = self.coefficients[3 * vd + comp]
        return vel
