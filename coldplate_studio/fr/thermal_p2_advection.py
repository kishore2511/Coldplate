"""Galerkin advection ρ c_p u·∇T with full P2 velocity on P1 temperature."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from coldplate_studio.fr.flow_p1_operators import fluid_node_mask
from coldplate_studio.fr.flow_p2_dof_map import build_p2_velocity_dof_map
from coldplate_studio.fr.flow_p2_velocity_field import P2VelocityField
from coldplate_studio.fr.tet_p2_basis import p1_pressure_values, p2_shape_gradients, tet_quadrature_4pt
from coldplate_studio.fr.tet_p1 import _tet_gradients
from coldplate_studio.mesh.integral_model import IntegralMesh


def assemble_p2_velocity_advection(
    mesh: IntegralMesh,
    p2_field: P2VelocityField,
    rho_cp: float,
) -> sparse.csr_matrix:
    """
    C_ij = ∫_Ω ρ c_p (u_h · ∇ψ_j) ψ_i dV on fluid tets (ψ: P1 temperature).
    """
    n = mesh.nodes.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    q_pts, q_w = tet_quadrature_4pt()
    dof_map = p2_field.dof_map
    for tet in mesh.fluid_tets:
        vol, vgrads = _tet_gradients(mesh.nodes[tet])
        loc_u = dof_map.local_dof_indices(tet)
        for q, wq in zip(q_pts, q_w):
            dvol = vol * wq
            u = p2_field.evaluate_on_tet(loc_u, q)
            grad_p1 = vgrads  # rows are ∇L_i
            phi_p1 = p1_pressure_values(q)
            for i in range(4):
                for j in range(4):
                    val = rho_cp * dvol * float(np.dot(u, grad_p1[j])) * phi_p1[i]
                    if abs(val) > 0:
                        rows.append(int(tet[i]))
                        cols.append(int(tet[j]))
                        data.append(val)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def build_p2_velocity_field_from_vertex_array(
    mesh: IntegralMesh,
    velocity_vertices: np.ndarray,
) -> P2VelocityField:
    """Promote vertex-only velocity to P2 field (edge mids interpolated from endpoints)."""
    fluid = fluid_node_mask(mesh)
    dof_map = build_p2_velocity_dof_map(mesh, fluid)
    n_u = dof_map.n_velocity_dofs
    coeffs = np.zeros(3 * n_u)
    for i in range(mesh.nodes.shape[0]):
        vd = int(dof_map.vertex_dof[i])
        if vd >= 0:
            for comp in range(3):
                coeffs[3 * vd + comp] = velocity_vertices[i, comp]
    for (a, b), ed in dof_map.edge_dof.items():
        for comp in range(3):
            coeffs[3 * ed + comp] = 0.5 * (velocity_vertices[a, comp] + velocity_vertices[b, comp])
    return P2VelocityField(dof_map=dof_map, coefficients=coeffs)
