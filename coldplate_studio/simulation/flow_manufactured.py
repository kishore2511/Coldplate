"""Manufactured steady Stokes solution on the integral fluid duct (verification)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.mesh.integral_model import IntegralModel


def manufactured_velocity_field(
    nodes: np.ndarray,
    *,
    length: float,
    height: float,
    width: float,
    grad_p: float,
    nu: float,
) -> np.ndarray:
    """
    u_x = U0 y(H-y) z(W-z) sin(πx/L), u_y=u_z=0 — zero on all six box faces.
    Divergence-free. For **P1 penalty** benchmarks use f = -ν∇²u (no pressure DOF).
    For **P2–P1 mixed** MMS use f = -ν∇²u + ∇p with p = grad_p * x (`flow_p2_mms.py`).
    """
    g = -grad_p
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    u0 = g / (2.0 * nu)
    ux = u0 * y * (height - y) * z * (width - z) * np.sin(np.pi * x / length)
    return np.column_stack((ux, np.zeros_like(ux), np.zeros_like(ux)))


def manufactured_body_force_rhs(
    mesh,
    n: int,
    length: float,
    height: float,
    width: float,
    grad_p: float,
    nu: float,
) -> np.ndarray:
    """Galerkin RHS ∫ f · φ_i with f = -ν∇²u_exact (x-component only)."""
    from coldplate_studio.fr.tet_p1 import _tet_gradients

    g = -grad_p
    b = np.zeros(3 * n)
    u0 = g / (2.0 * nu)
    for tet in mesh.fluid_tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        pts = mesh.nodes[tet]
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        sinx = np.sin(np.pi * x / length)
        lap = u0 * sinx * (
            -2.0 * z * (width - z)
            - 2.0 * y * (height - y)
            - (np.pi**2 / length**2) * y * (height - y) * z * (width - z)
        )
        fx = -nu * lap
        for i in range(4):
            b[0 * n + tet[i]] += fx[i] * vol / 4.0
    return b


@dataclass
class ManufacturedFlowResult:
    mesh_size: float
    ux_l2_error: float
    ux_linf_error: float
    divergence_l2: float
    solver_residual_norm: float


def run_manufactured_stokes_benchmark(
    model: IntegralModel,
    *,
    length: float = 1.0,
    height: float = 0.1,
    width: float = 0.2,
    nu: float = 1e-3,
    grad_p: float = -1.0,
    mesh_size: float,
    beta: float = 0.0,
) -> ManufacturedFlowResult:
    mesh = model.mesh
    n = mesh.nodes.shape[0]
    u_exact = manufactured_velocity_field(
        mesh.nodes, length=length, height=height, width=width, grad_p=grad_p, nu=nu
    )
    wall_nodes = np.array(list(model.mesh.boundary_fluid_nodes), dtype=int)
    # Custom solve with manufactured RHS
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from coldplate_studio.fr.flow_p1_operators import (
        assemble_divergence_penalty,
        assemble_viscous_stiffness,
        wall_pinned_dofs,
    )

    a = (assemble_viscous_stiffness(mesh, nu) + assemble_divergence_penalty(mesh, beta)).tolil()
    b = manufactured_body_force_rhs(mesh, n, length, height, width, grad_p, nu)
    pinned = wall_pinned_dofs(n, set(int(i) for i in wall_nodes))
    for dof in sorted(pinned):
        a[dof, :] = 0
        a[:, dof] = 0
        a[dof, dof] = 1.0
        b[dof] = 0.0
    u_flat = spsolve(a.tocsr(), b)
    velocity = np.zeros((n, 3))
    for k in range(3):
        velocity[:, k] = u_flat[k * n : (k + 1) * n]
    diff = velocity[:, 0] - u_exact[:, 0]
    fluid = np.zeros(n, bool)
    for tet in mesh.fluid_tets:
        fluid[tet] = True
    diff_f = diff[fluid]
    l2 = float(np.sqrt(np.mean(diff_f**2)))
    linf = float(np.max(np.abs(diff_f)))
    from coldplate_studio.fr.tet_p1 import _tet_gradients

    div_sq = vol_sum = 0.0
    for tet in mesh.fluid_tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        div = sum(float(np.dot(grads[:, k], velocity[tet, k])) for k in range(3))
        div_sq += vol * div**2
        vol_sum += vol
    div_l2 = float(np.sqrt(div_sq / max(vol_sum, 1e-30)))
    res = float(np.linalg.norm(a @ u_flat - b) / max(np.linalg.norm(b), 1e-12))
    return ManufacturedFlowResult(
        mesh_size=mesh_size,
        ux_l2_error=l2,
        ux_linf_error=linf,
        divergence_l2=div_l2,
        solver_residual_norm=res,
    )
