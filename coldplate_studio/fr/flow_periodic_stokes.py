"""
Developed-channel Stokes with x-periodic duct mesh and y no-slip walls.

Governing (periodic x, body force G = -dp/dx):
  -ν ∇²u + β ∇(∇·u) = (G, 0, 0)

Analytical (fully developed, parabolic in y, uniform in z, no-slip y=0,H):
  u_x(y) = (G / 2ν) y (H - y)
  Q = G W H³ / (12 ν)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.fr.flow_p1_operators import (
    assemble_divergence_penalty,
    assemble_viscous_stiffness,
    wall_pinned_dofs,
)
from coldplate_studio.mesh.integral_model import IntegralMesh
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class DevelopedChannelResult:
    velocity: np.ndarray
    divergence_l2: float
    ux_l2_error: float
    ux_linf_error: float
    flow_rate_numerical: float
    flow_rate_analytical: float
    flow_rate_relative_error: float
    mass_flux_imbalance: float


def _parabolic_ux(y: np.ndarray, height: float, g: float, nu: float) -> np.ndarray:
    return (g / (2.0 * nu)) * y * (height - y)


def _divergence_l2(mesh: IntegralMesh, velocity: np.ndarray) -> float:
    from coldplate_studio.fr.tet_p1 import _tet_gradients

    div_sq = vol_sum = 0.0
    for tet in mesh.fluid_tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        div = sum(float(np.dot(grads[:, k], velocity[tet, k])) for k in range(3))
        div_sq += vol * div**2
        vol_sum += vol
    return float(np.sqrt(div_sq / max(vol_sum, 1e-30)))


def pair_x_periodic_nodes(nodes: np.ndarray, length: float, tol: float = 5e-4) -> list[tuple[int, int]]:
    """Match x=0 and x=L nodes by (y,z) for periodic channel."""
    left: dict[tuple[float, float], int] = {}
    right: dict[tuple[float, float], int] = {}
    for i, p in enumerate(nodes):
        if abs(p[0]) <= tol:
            left[(round(float(p[1]), 8), round(float(p[2]), 8))] = i
        if abs(p[0] - length) <= tol:
            right[(round(float(p[1]), 8), round(float(p[2]), 8))] = i
    return [(left[k], right[k]) for k in left if k in right]


def _add_periodic_velocity_penalty(
    a: sparse.lil_matrix,
    pairs: list[tuple[int, int]],
    n: int,
    gamma: float,
) -> None:
    for a_node, b_node in pairs:
        for comp in range(3):
            ia = comp * n + a_node
            ib = comp * n + b_node
            a[ia, ia] += gamma
            a[ib, ib] += gamma
            a[ia, ib] -= gamma
            a[ib, ia] -= gamma


def _y_wall_nodes(mesh: IntegralMesh, height: float, tol_frac: float = 0.02) -> np.ndarray:
    ytol = tol_frac * height
    nodes = mesh.nodes
    walls: list[int] = []
    for i in mesh.boundary_fluid_nodes:
        y = nodes[int(i), 1]
        if y <= ytol or y >= height - ytol:
            walls.append(int(i))
    return np.asarray(walls, dtype=int)


def solve_developed_periodic_channel(
    mesh: IntegralMesh,
    *,
    length: float = 1.0,
    height: float = 0.1,
    width: float = 0.05,
    nu: float = 1e-3,
    grad_p: float = -1.0,
    beta: float = 1e3,
    periodic_gamma: float = 1e4,
) -> DevelopedChannelResult:
    n = mesh.nodes.shape[0]
    wall_nodes = _y_wall_nodes(mesh, height)
    pinned = wall_pinned_dofs(n, set(int(i) for i in wall_nodes))
    pairs = pair_x_periodic_nodes(mesh.nodes, length)
    a = (assemble_viscous_stiffness(mesh, nu) + assemble_divergence_penalty(mesh, beta)).tolil()
    _add_periodic_velocity_penalty(a, pairs, n, periodic_gamma)
    g = -grad_p
    b = np.zeros(3 * n)
    for tet in mesh.fluid_tets:
        from coldplate_studio.fr.tet_p1 import _tet_gradients

        vol, _ = _tet_gradients(mesh.nodes[tet])
        for i in range(4):
            b[0 * n + tet[i]] += g * vol / 4.0
    for dof in sorted(pinned):
        a[dof, :] = 0
        a[:, dof] = 0
        a[dof, dof] = 1.0
        b[dof] = 0.0
    u_flat = spsolve(a.tocsr(), b)
    velocity = np.zeros((n, 3))
    for k in range(3):
        velocity[:, k] = u_flat[k * n : (k + 1) * n]
    y = mesh.nodes[:, 1]
    u_exact = _parabolic_ux(y, height, g, nu)
    fluid = np.zeros(n, bool)
    for tet in mesh.fluid_tets:
        fluid[tet] = True
    interior = fluid & (y > 0.05 * height) & (y < 0.95 * height)
    diff = velocity[interior, 0] - u_exact[interior]
    ux_l2 = float(np.sqrt(np.mean(diff**2)))
    ux_linf = float(np.max(np.abs(diff))) if diff.size else 0.0
    xs = mesh.nodes[:, 0]
    mid = interior & (xs > 0.2 * length) & (xs < 0.8 * length)
    q_num = float(np.mean(velocity[mid, 0]) * width * height) if mid.any() else 0.0
    q_ana = g * width * (height**3) / (12.0 * nu)
    # Port imbalance N/A for periodic; use divergence as continuity metric.
    return DevelopedChannelResult(
        velocity=velocity,
        divergence_l2=_divergence_l2(mesh, velocity),
        ux_l2_error=ux_l2,
        ux_linf_error=ux_linf,
        flow_rate_numerical=q_num,
        flow_rate_analytical=q_ana,
        flow_rate_relative_error=abs(q_num - q_ana) / max(abs(q_ana), 1e-12),
        mass_flux_imbalance=0.0,
    )
