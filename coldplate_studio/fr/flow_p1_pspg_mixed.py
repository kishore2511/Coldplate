"""
Steady Stokes — P1 velocity / P1 pressure with PSPG stabilization (not Taylor–Hood, not FR).

Weak form:
  ν ∫ ∇u·∇v dΩ − ∫ p ∇·v dΩ = ∫ f·v dΩ
  ∫ q ∇·u dΩ + τ ∫ ∇q·∇p dΩ = 0   (PSPG, τ = h²/(4ν))

Pressure reference: pin one pressure DOF to 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from coldplate_studio.fr.flow_p1_operators import (
    assemble_pressure_laplacian,
    assemble_viscous_stiffness,
    assemble_weak_divergence,
    unflat_velocity,
    wall_pinned_dofs,
)
from coldplate_studio.fr.flow_periodic_stokes import (
    _add_periodic_velocity_penalty,
    _divergence_l2,
    pair_x_periodic_nodes,
)
from coldplate_studio.mesh.integral_model import IntegralMesh


@dataclass
class PSPGStokesResult:
    velocity: np.ndarray
    pressure: np.ndarray
    divergence_l2: float
    pressure_residual_norm: float
    solver_residual_norm: float


def _mean_mesh_length(mesh: IntegralMesh) -> float:
    from coldplate_studio.fr.tet_p1 import _tet_gradients

    vols = []
    for tet in mesh.fluid_tets:
        vol, _ = _tet_gradients(mesh.nodes[tet])
        vols.append(vol)
    v = float(np.mean(vols)) ** (1.0 / 3.0)
    return max(v, 1e-6)


def _body_force_rhs(mesh: IntegralMesh, n: int, force_x: float) -> np.ndarray:
    from coldplate_studio.fr.tet_p1 import _tet_gradients

    b = np.zeros(3 * n)
    for tet in mesh.fluid_tets:
        vol, _ = _tet_gradients(mesh.nodes[tet])
        for i in range(4):
            b[0 * n + tet[i]] += force_x * vol / 4.0
    return b


def solve_stokes_pspg(
    mesh: IntegralMesh,
    *,
    nu: float = 1e-3,
    force_x: float = 1.0,
    pinned_velocity_nodes: set[int],
    pressure_pin_node: int,
    periodic_pairs: list[tuple[int, int]] | None = None,
    periodic_gamma: float = 1e4,
) -> PSPGStokesResult:
    n = mesh.nodes.shape[0]
    dim_u = 3 * n
    d = assemble_weak_divergence(mesh)
    k_u = assemble_viscous_stiffness(mesh, nu)
    lp = assemble_pressure_laplacian(mesh)
    h = _mean_mesh_length(mesh)
    tau = (h**2) / (4.0 * nu)
    a = sparse.bmat(
        [
            [k_u, d.T],
            [d, -tau * lp],
        ],
        format="lil",
    )
    rhs = np.zeros(dim_u + n)
    rhs[:dim_u] = _body_force_rhs(mesh, n, force_x)
    if periodic_pairs:
        _add_periodic_velocity_penalty(a, periodic_pairs, n, periodic_gamma)
    pinned = wall_pinned_dofs(n, pinned_velocity_nodes)
    for dof in sorted(pinned):
        a[dof, :] = 0
        a[:, dof] = 0
        a[dof, dof] = 1.0
        rhs[dof] = 0.0
    p_row = dim_u + int(pressure_pin_node)
    a[p_row, :] = 0
    a[:, p_row] = 0
    a[p_row, p_row] = 1.0
    rhs[p_row] = 0.0
    sol = spsolve(a.tocsr(), rhs)
    u_flat = sol[:dim_u]
    pressure = sol[dim_u:]
    velocity = unflat_velocity(u_flat, n)
    div_l2 = _divergence_l2(mesh, velocity)
    p_res = float(np.linalg.norm(d @ u_flat) / max(np.linalg.norm(u_flat), 1e-12))
    res = float(np.linalg.norm(a @ sol - rhs) / max(np.linalg.norm(rhs), 1e-12))
    return PSPGStokesResult(
        velocity=velocity,
        pressure=pressure,
        divergence_l2=div_l2,
        pressure_residual_norm=p_res,
        solver_residual_norm=res,
    )
