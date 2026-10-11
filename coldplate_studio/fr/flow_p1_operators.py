"""Shared P1 tetrahedral operators for incompressible flow (not high-order FR)."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from coldplate_studio.fr.tet_p1 import _tet_gradients
from coldplate_studio.mesh.integral_model import IntegralMesh


def fluid_node_mask(mesh: IntegralMesh) -> np.ndarray:
    n = mesh.nodes.shape[0]
    mask = np.zeros(n, dtype=bool)
    for tet in mesh.fluid_tets:
        mask[tet] = True
    return mask


def assemble_lumped_mass(mesh: IntegralMesh, rho: float = 1.0) -> sparse.csr_matrix:
    """Vector mass: rho per velocity component (block-diagonal)."""
    n = mesh.nodes.shape[0]
    lumped = np.zeros(n)
    for tet in mesh.fluid_tets:
        vol, _ = _tet_gradients(mesh.nodes[tet])
        for i in range(4):
            lumped[tet[i]] += rho * vol / 4.0
    m_scalar = sparse.diags(lumped, format="csr")
    blocks = [m_scalar, m_scalar, m_scalar]
    return sparse.block_diag(blocks, format="csr")


def assemble_scalar_lumped_mass(mesh: IntegralMesh) -> np.ndarray:
    lumped = np.zeros(mesh.nodes.shape[0])
    for tet in mesh.fluid_tets:
        vol, _ = _tet_gradients(mesh.nodes[tet])
        for i in range(4):
            lumped[tet[i]] += vol / 4.0
    return lumped


def assemble_viscous_stiffness(mesh: IntegralMesh, nu: float) -> sparse.csr_matrix:
    n = mesh.nodes.shape[0]
    dim = 3 * n
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for tet in mesh.fluid_tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        for comp in range(3):
            for i in range(4):
                for j in range(4):
                    val = nu * vol * float(np.dot(grads[i], grads[j]))
                    rows.append(comp * n + tet[i])
                    cols.append(comp * n + tet[j])
                    data.append(val)
    return sparse.csr_matrix((data, (rows, cols)), shape=(dim, dim))


def assemble_divergence_penalty(mesh: IntegralMesh, beta: float) -> sparse.csr_matrix:
    n = mesh.nodes.shape[0]
    dim = 3 * n
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for tet in mesh.fluid_tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    for ell in range(3):
                        rows.append(k * n + tet[i])
                        cols.append(ell * n + tet[j])
                        data.append(beta * vol * grads[i, k] * grads[j, ell])
    return sparse.csr_matrix((data, (rows, cols)), shape=(dim, dim))


def assemble_weak_divergence(mesh: IntegralMesh) -> sparse.csr_matrix:
    """
    D: (3n) -> n with (D @ u)_i = Σ_{j,k} u_{j,k} ∫_Ω φ_i ∂φ_j/∂x_k dV (P1 lumped quadrature).
    """
    n = mesh.nodes.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for tet in mesh.fluid_tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    val = vol / 4.0 * grads[j, k]
                    rows.append(tet[i])
                    cols.append(k * n + tet[j])
                    data.append(val)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, 3 * n))


def lumped_divergence_at_nodes(mesh: IntegralMesh, velocity: np.ndarray) -> np.ndarray:
    """Nodal lumped div_h(u): volume-weighted average of element-wise P1 div."""
    n = mesh.nodes.shape[0]
    div_sum = np.zeros(n)
    weight = np.zeros(n)
    for tet in mesh.fluid_tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        div_e = 0.0
        for k in range(3):
            div_e += float(np.dot(grads[:, k], velocity[tet, k]))
        for i in range(4):
            idx = tet[i]
            div_sum[idx] += vol / 4.0 * div_e
            weight[idx] += vol / 4.0
    out = np.zeros(n)
    mask = weight > 0
    out[mask] = div_sum[mask] / weight[mask]
    return out


def assemble_pressure_laplacian(mesh: IntegralMesh) -> sparse.csr_matrix:
    n = mesh.nodes.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for tet in mesh.fluid_tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        for i in range(4):
            for j in range(4):
                val = vol * float(np.dot(grads[i], grads[j]))
                rows.append(tet[i])
                cols.append(tet[j])
                data.append(val)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def assemble_velocity_gradient(mesh: IntegralMesh) -> sparse.csr_matrix:
    """
    G: n -> 3n mapping pressure nodal values to nodal velocity correction grad p.
    (G p)_ at node i component k = Σ_j p_j ∫ φ_i ∂φ_j/∂x_k dV (lumped row sum).
    """
    n = mesh.nodes.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for tet in mesh.fluid_tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    rows.append(k * n + tet[i])
                    cols.append(tet[j])
                    data.append(vol / 4.0 * grads[j, k])
    return sparse.csr_matrix((data, (rows, cols)), shape=(3 * n, n))


def apply_dirichlet_block(
    a: sparse.lil_matrix,
    b: np.ndarray,
    pinned_dofs: set[int],
) -> None:
    for row in sorted(pinned_dofs):
        a[row, :] = 0
        a[:, row] = 0
        a[row, row] = 1.0
        b[row] = 0.0


def wall_pinned_dofs(n: int, pinned_nodes: set[int]) -> set[int]:
    dofs: set[int] = set()
    for idx in pinned_nodes:
        for comp in range(3):
            dofs.add(comp * n + int(idx))
    return dofs


def flat_velocity(velocity: np.ndarray) -> np.ndarray:
    n = velocity.shape[0]
    out = np.zeros(3 * n)
    for k in range(3):
        out[k * n : (k + 1) * n] = velocity[:, k]
    return out


def unflat_velocity(u_flat: np.ndarray, n: int) -> np.ndarray:
    velocity = np.zeros((n, 3))
    for k in range(3):
        velocity[:, k] = u_flat[k * n : (k + 1) * n]
    return velocity
