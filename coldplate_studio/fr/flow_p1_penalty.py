"""
Steady incompressible Stokes flow (P1 tet) via viscous + divergence penalty.

This is a **vector finite-element** solve — not the constant-velocity scalar model.

Governing steady approximation:
  -ν ∇²u + β ∇(∇·u) = f
with uniform body force f = (dp/dx, 0, 0) driving channel flow.
No-slip on exterior fluid boundaries and FSI faces.

Pressure is not a DOF yet; β penalizes divergence (precursor to artificial compressibility FR path).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from coldplate_studio.fr.tet_p1 import _tet_gradients
from coldplate_studio.mesh.integral_model import IntegralMesh


@dataclass
class FlowSolveResult:
    velocity: np.ndarray  # (n_nodes, 3)
    divergence_l2: float
    inlet_flux_m3_s: float
    outlet_flux_m3_s: float
    mean_velocity_x: float


def _assemble_vector_laplacian(mesh: IntegralMesh, nu: float, fluid_mask: np.ndarray) -> sparse.csr_matrix:
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


def _assemble_divergence_penalty(mesh: IntegralMesh, beta: float) -> sparse.csr_matrix:
    n = mesh.nodes.shape[0]
    dim = 3 * n
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for tet in mesh.fluid_tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        for i in range(4):
            for j in range(4):
                # ∫ β (∇·u)(∇·v) with P1: Σ_{k,ℓ} β vol ∂φ_i/∂x_k ∂φ_j/∂x_ℓ
                for k in range(3):
                    for ell in range(3):
                        rows.append(k * n + tet[i])
                        cols.append(ell * n + tet[j])
                        data.append(beta * vol * grads[i, k] * grads[j, ell])
    return sparse.csr_matrix((data, (rows, cols)), shape=(dim, dim))


def _fluid_boundary_outward_area_vectors(mesh: IntegralMesh) -> dict[tuple[int, int, int], np.ndarray]:
    """Area vector (2× area × unit outward normal) for each fluid boundary triangle."""
    local = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    face_count: dict[tuple[int, int, int], int] = {}
    area_vecs: dict[tuple[int, int, int], np.ndarray] = {}
    for tet in mesh.fluid_tets:
        pts = mesh.nodes[tet]
        for a, b, c in local:
            key = tuple(sorted((int(tet[a]), int(tet[b]), int(tet[c]))))
            face_count[key] = face_count.get(key, 0) + 1
            opp = next(i for i in range(4) if i not in (a, b, c))
            v0, v1, v2 = pts[a], pts[b], pts[c]
            n_vec = np.cross(v1 - v0, v2 - v0)
            if float(np.dot(n_vec, pts[opp] - v0)) > 0.0:
                n_vec = -n_vec
            area_vecs[key] = n_vec
    return {k: v for k, v in area_vecs.items() if face_count.get(k, 0) == 1}


def _oriented_face_flux(
    mesh: IntegralMesh,
    velocity: np.ndarray,
    face_keys: list[tuple[int, int, int]],
    area_vecs: dict[tuple[int, int, int], np.ndarray],
) -> float:
    """∫_Γ u·n dA with n outward from the fluid domain."""
    flux = 0.0
    for key in face_keys:
        n_vec = area_vecs[key]
        i0, i1, i2 = key
        u_face = (velocity[i0] + velocity[i1] + velocity[i2]) / 3.0
        flux += float(np.dot(u_face, n_vec) / 2.0)
    return flux


def solve_steady_channel_flow(
    mesh: IntegralMesh,
    *,
    nu: float = 1e-3,
    beta: float = 1e2,
    dp_dx: float = -1.0,
    no_slip_nodes: np.ndarray,
    inlet_faces: list[tuple[int, int, int]],
    outlet_faces: list[tuple[int, int, int]],
    zero_velocity_nodes: np.ndarray | None = None,
) -> FlowSolveResult:
    n = mesh.nodes.shape[0]
    fluid_mask = np.zeros(n, dtype=bool)
    for tet in mesh.fluid_tets:
        fluid_mask[tet] = True
    pinned = set(int(i) for i in no_slip_nodes)
    if zero_velocity_nodes is not None:
        pinned.update(int(i) for i in zero_velocity_nodes)
    else:
        pinned.update(int(i) for i in np.where(~fluid_mask)[0])
    a = (_assemble_vector_laplacian(mesh, nu, fluid_mask) + _assemble_divergence_penalty(mesh, beta)).tolil()
    b = np.zeros(3 * n)
    for tet in mesh.fluid_tets:
        vol, _ = _tet_gradients(mesh.nodes[tet])
        for i in range(4):
            b[0 * n + tet[i]] += dp_dx * vol / 4.0
    # Dirichlet u=0 on walls and non-fluid nodes (integral mesh includes solid DOFs).
    for idx in sorted(pinned):
        for comp in range(3):
            row = comp * n + int(idx)
            a[row, :] = 0
            a[:, row] = 0
            a[row, row] = 1.0
            b[row] = 0.0
    u_flat = spsolve(a.tocsr(), b)
    velocity = np.zeros((n, 3))
    for comp in range(3):
        velocity[:, comp] = u_flat[comp * n : (comp + 1) * n]
    div_sq = 0.0
    vol_sum = 0.0
    for tet in mesh.fluid_tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        div = 0.0
        for k in range(3):
            div += float(np.dot(grads[:, k], velocity[tet, k]))
        div_sq += vol * div**2
        vol_sum += vol
    div_l2 = float(np.sqrt(div_sq / max(vol_sum, 1e-30)))
    area_vecs = _fluid_boundary_outward_area_vectors(mesh)
    phi_in = _oriented_face_flux(mesh, velocity, inlet_faces, area_vecs)
    phi_out = _oriented_face_flux(mesh, velocity, outlet_faces, area_vecs)
    mean_ux = float(np.mean(velocity[fluid_mask, 0]))
    return FlowSolveResult(
        velocity=velocity,
        divergence_l2=div_l2,
        inlet_flux_m3_s=phi_in,
        outlet_flux_m3_s=phi_out,
        mean_velocity_x=mean_ux,
    )
