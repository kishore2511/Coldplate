"""P1 simplex operators (lowest-order nodal FR / DG branch on tetrahedra)."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from coldplate_studio.mesh.integral_model import IntegralMesh


def _tet_gradients(nodes: np.ndarray) -> tuple[float, np.ndarray]:
    """Return volume and (4,3) gradient rows for P1 basis on a tet."""
    x = nodes
    mat = np.column_stack((np.ones(4), x))
    det = np.linalg.det(mat)
    vol = abs(det) / 6.0
    inv = np.linalg.inv(mat)
    grads = inv[1:, :].T  # each row is grad phi_i
    return vol, grads


def assemble_p1_operators(
    mesh: IntegralMesh,
    thermal_diffusivity: float,
    velocity: tuple[float, float, float],
    rho_cp: float,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """
    Build lumped mass M, diffusion stiffness K, advection C, and cell volumes per tet.

    Governing semi-discrete form: M dT/dt = -K T - C T.
    """
    n_nodes = mesh.nodes.shape[0]
    vx, vy, vz = velocity
    v = np.array([vx, vy, vz])
    rows_k: list[int] = []
    cols_k: list[int] = []
    data_k: list[float] = []
    rows_c: list[int] = []
    cols_c: list[int] = []
    data_c: list[float] = []
    lumped = np.zeros(n_nodes)
    tet_volumes = np.zeros(mesh.fluid_tets.shape[0])
    for e, tet in enumerate(mesh.fluid_tets):
        vol, grads = _tet_gradients(mesh.nodes[tet])
        tet_volumes[e] = vol
        for i in range(4):
            lumped[tet[i]] += rho_cp * vol / 4.0
        for i in range(4):
            for j in range(4):
                kij = thermal_diffusivity * vol * float(np.dot(grads[i], grads[j]))
                rows_k.append(tet[i])
                cols_k.append(tet[j])
                data_k.append(kij)
                cij = vol * float(np.dot(v, grads[j]))
                if abs(cij) > 0:
                    rows_c.append(tet[i])
                    cols_c.append(tet[j])
                    data_c.append(cij)
    m = sparse.diags(lumped, format="csr")
    k = sparse.csr_matrix((data_k, (rows_k, cols_k)), shape=(n_nodes, n_nodes))
    c = sparse.csr_matrix((data_c, (rows_c, cols_c)), shape=(n_nodes, n_nodes))
    return m, k, c, tet_volumes
