"""Taylor–Hood P2–P1 Stokes assembly on fluid tets."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from coldplate_studio.fr.flow_p2_dof_map import P2VelocityDofMap, build_p2_velocity_dof_map
from coldplate_studio.fr.flow_p1_operators import fluid_node_mask
from coldplate_studio.fr.tet_p2_basis import (
    linear_tet_volume_and_grads,
    p1_pressure_gradients,
    p1_pressure_values,
    p2_shape_gradients,
    p2_shape_values,
    tet_quadrature_4pt,
)
from coldplate_studio.mesh.integral_model import IntegralMesh


def assemble_p2p1_stokes(
    mesh: IntegralMesh,
    nu: float,
    dof_map: P2VelocityDofMap,
    pressure_node_indices: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """
    Return (K_uu, B, C) with sizes (n_u, n_u), (n_p, n_u), (n_p, n_p unused — zero block).
    Mixed system: [[K, B^T], [B, 0]].
    """
    n_u = dof_map.n_velocity_dofs
    n_p = pressure_node_indices.size
    p_local = {int(n): i for i, n in enumerate(pressure_node_indices)}
    rows_k, cols_k, data_k = [], [], []
    rows_b, cols_b, data_b = [], [], []
    q_pts, q_w = tet_quadrature_4pt()
    for tet in mesh.fluid_tets:
        nodes = mesh.nodes[tet]
        vol, vgrads = linear_tet_volume_and_grads(nodes)
        loc_u = dof_map.local_dof_indices(tet)
        loc_p = [p_local[int(tet[i])] for i in range(4) if int(tet[i]) in p_local]
        if len(loc_p) < 4:
            continue
        for q, wq in zip(q_pts, q_w):
            dvol = vol * wq
            L = q
            grad_p2 = p2_shape_gradients(L, vgrads)
            grad_p1 = p1_pressure_gradients(vgrads)
            # Viscous: nu * grad u · grad v (vector Laplacian, 3 components)
            for a in range(10):
                ia = int(loc_u[a])
                for b in range(10):
                    ib = int(loc_u[b])
                    val = nu * dvol * float(np.dot(grad_p2[a], grad_p2[b]))
                    for comp in range(3):
                        rows_k.append(3 * ia + comp)
                        cols_k.append(3 * ib + comp)
                        data_k.append(val)
            # Weak divergence: ∫ q div u
            div_u = np.zeros(10)
            for b in range(10):
                for comp in range(3):
                    div_u[b] += grad_p2[b, comp]
            for pi in range(4):
                node = int(tet[pi])
                if node not in p_local:
                    continue
                row_p = p_local[node]
                q_val = p1_pressure_values(L)[pi]
                for b in range(10):
                    ib = int(loc_u[b])
                    for comp in range(3):
                        val = dvol * q_val * grad_p2[b, comp]
                        rows_b.append(row_p)
                        cols_b.append(3 * ib + comp)
                        data_b.append(val)
    k_uu = sparse.csr_matrix((data_k, (rows_k, cols_k)), shape=(3 * n_u, 3 * n_u))
    b_mat = sparse.csr_matrix((data_b, (rows_b, cols_b)), shape=(n_p, 3 * n_u))
    c_zero = sparse.csr_matrix((n_p, n_p))
    return k_uu, b_mat, c_zero


def assemble_p2_body_force_rhs(
    mesh: IntegralMesh,
    dof_map: P2VelocityDofMap,
    force_xyz: tuple[float, float, float],
) -> np.ndarray:
    fx, fy, fz = force_xyz
    n_u = dof_map.n_velocity_dofs
    rhs = np.zeros(3 * n_u)
    q_pts, q_w = tet_quadrature_4pt()
    for tet in mesh.fluid_tets:
        nodes = mesh.nodes[tet]
        vol, vgrads = linear_tet_volume_and_grads(nodes)
        loc_u = dof_map.local_dof_indices(tet)
        for q, wq in zip(q_pts, q_w):
            dvol = vol * wq
            phi = p2_shape_values(q)
            for a in range(10):
                ia = int(loc_u[a])
                rhs[3 * ia + 0] += fx * phi[a] * dvol
                rhs[3 * ia + 1] += fy * phi[a] * dvol
                rhs[3 * ia + 2] += fz * phi[a] * dvol
    return rhs


def assemble_p2_mms_body_force_rhs(
    mesh: IntegralMesh,
    dof_map: P2VelocityDofMap,
    force_at_barycentric: callable,
) -> np.ndarray:
    """force_at_barycentric(nodes4, L) -> (3,) force vector in physical space."""
    n_u = dof_map.n_velocity_dofs
    rhs = np.zeros(3 * n_u)
    q_pts, q_w = tet_quadrature_4pt()
    for tet in mesh.fluid_tets:
        nodes = mesh.nodes[tet]
        vol, _ = linear_tet_volume_and_grads(nodes)
        loc_u = dof_map.local_dof_indices(tet)
        for q, wq in zip(q_pts, q_w):
            dvol = vol * wq
            # Physical point from barycentric coordinates on linear tet
            x = np.zeros(3)
            for k in range(4):
                x += q[k] * nodes[k]
            f = force_at_barycentric(x)
            phi = p2_shape_values(q)
            for a in range(10):
                ia = int(loc_u[a])
                for comp in range(3):
                    rhs[3 * ia + comp] += f[comp] * phi[a] * dvol
    return rhs
