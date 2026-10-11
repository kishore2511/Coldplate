"""
Transient incompressible Stokes flow (P1 tet) — fractional-step pressure projection.

Algorithm (Chorin–Temam style, equal-order P1):
  1. Momentum: (M/dt + νK + γP) u* = M/dt u^n + f^{n+1}
  2. Pressure Poisson: Lp φ = div_h(u*) / dt  (φ is pressure correction increment)
  3. Correction: u^{n+1} = u* − dt G φ, enforce Dirichlet again

γP is the same grad-div penalty as steady flow for stabilization (not high-order FR).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from coldplate_studio.fr.flow_p1_operators import (
    apply_dirichlet_block,
    assemble_divergence_penalty,
    assemble_lumped_mass,
    assemble_pressure_laplacian,
    assemble_scalar_lumped_mass,
    assemble_velocity_gradient,
    assemble_viscous_stiffness,
    flat_velocity,
    lumped_divergence_at_nodes,
    unflat_velocity,
    wall_pinned_dofs,
)
from coldplate_studio.fr.flow_p1_penalty import (
    _fluid_boundary_outward_area_vectors,
    _oriented_face_flux,
)
from coldplate_studio.mesh.integral_model import IntegralMesh


@dataclass
class TransientFlowState:
    velocity: np.ndarray
    pressure: np.ndarray
    time: float


@dataclass
class TransientFlowStepDiagnostics:
    time: float
    divergence_l2: float
    div_before_projection: float
    div_after_projection: float
    inlet_flux: float
    outlet_flux: float
    momentum_residual_norm: float


@dataclass
class TransientFlowResult:
    final: TransientFlowState
    history: list[TransientFlowStepDiagnostics] = field(default_factory=list)
    converged: bool = True
    failure_message: str = ""


def _divergence_l2(mesh: IntegralMesh, velocity: np.ndarray) -> float:
    from coldplate_studio.fr.tet_p1 import _tet_gradients

    div_sq = 0.0
    vol_sum = 0.0
    for tet in mesh.fluid_tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        div = 0.0
        for k in range(3):
            div += float(np.dot(grads[:, k], velocity[tet, k]))
        div_sq += vol * div**2
        vol_sum += vol
    return float(np.sqrt(div_sq / max(vol_sum, 1e-30)))


def _body_force_rhs(mesh: IntegralMesh, n: int, force: np.ndarray) -> np.ndarray:
    b = np.zeros(3 * n)
    fx, fy, fz = force
    for tet in mesh.fluid_tets:
        from coldplate_studio.fr.tet_p1 import _tet_gradients

        vol, _ = _tet_gradients(mesh.nodes[tet])
        for i in range(4):
            b[0 * n + tet[i]] += fx * vol / 4.0
            b[1 * n + tet[i]] += fy * vol / 4.0
            b[2 * n + tet[i]] += fz * vol / 4.0
    return b


def advance_transient_stokes(
    mesh: IntegralMesh,
    state: TransientFlowState,
    *,
    dt: float,
    nu: float = 1e-3,
    gamma_penalty: float = 1e2,
    force: np.ndarray | None = None,
    pinned_nodes: set[int],
    pressure_pin_node: int,
    inlet_faces: list[tuple[int, int, int]],
    outlet_faces: list[tuple[int, int, int]],
) -> tuple[TransientFlowState, TransientFlowStepDiagnostics]:
    n = mesh.nodes.shape[0]
    if force is None:
        force = np.array([-1.0, 0.0, 0.0])
    m = assemble_lumped_mass(mesh)
    k = assemble_viscous_stiffness(mesh, nu)
    p_pen = assemble_divergence_penalty(mesh, gamma_penalty)
    lp = assemble_pressure_laplacian(mesh)
    g_grad = assemble_velocity_gradient(mesh)
    pinned_dofs = wall_pinned_dofs(n, pinned_nodes)
    a_mom = (m / dt + k + p_pen).tolil()
    u_flat = flat_velocity(state.velocity)
    rhs = m @ (u_flat / dt) + _body_force_rhs(mesh, n, force)
    apply_dirichlet_block(a_mom, rhs, pinned_dofs)
    u_star = spsolve(a_mom.tocsr(), rhs)
    vel_star = unflat_velocity(u_star, n)
    div_before = _divergence_l2(mesh, vel_star)
    div_h = lumped_divergence_at_nodes(mesh, vel_star)
    lp_sys = lp.tolil()
    p_rhs = div_h / dt
    pin = int(pressure_pin_node)
    lp_sys[pin, :] = 0
    lp_sys[:, pin] = 0
    lp_sys[pin, pin] = 1.0
    p_rhs[pin] = 0.0
    phi = spsolve(lp_sys.tocsr(), p_rhs)
    u_corr = u_star - dt * (g_grad @ phi)
    for dof in pinned_dofs:
        u_corr[dof] = 0.0
    vel_np1 = unflat_velocity(u_corr, n)
    p_np1 = state.pressure + phi
    div_after = _divergence_l2(mesh, vel_np1)
    area_vecs = _fluid_boundary_outward_area_vectors(mesh)
    phi_in = _oriented_face_flux(mesh, vel_np1, inlet_faces, area_vecs)
    phi_out = _oriented_face_flux(mesh, vel_np1, outlet_faces, area_vecs)
    mom_res = float(np.linalg.norm(a_mom @ u_star - rhs) / max(np.linalg.norm(rhs), 1e-12))
    diag = TransientFlowStepDiagnostics(
        time=state.time + dt,
        divergence_l2=div_after,
        div_before_projection=div_before,
        div_after_projection=div_after,
        inlet_flux=phi_in,
        outlet_flux=phi_out,
        momentum_residual_norm=mom_res,
    )
    return TransientFlowState(velocity=vel_np1, pressure=p_np1, time=state.time + dt), diag


def run_transient_stokes(
    mesh: IntegralMesh,
    *,
    t_end: float,
    dt: float,
    nu: float = 1e-3,
    gamma_penalty: float = 1e2,
    force: np.ndarray | None = None,
    pinned_nodes: set[int],
    pressure_pin_node: int,
    inlet_faces: list[tuple[int, int, int]],
    outlet_faces: list[tuple[int, int, int]],
    u0: np.ndarray | None = None,
) -> TransientFlowResult:
    n = mesh.nodes.shape[0]
    if u0 is None:
        u0 = np.zeros((n, 3))
    state = TransientFlowState(velocity=u0.copy(), pressure=np.zeros(n), time=0.0)
    history: list[TransientFlowStepDiagnostics] = []
    n_steps = int(np.ceil(t_end / dt))
    for _ in range(n_steps):
        try:
            state, diag = advance_transient_stokes(
                mesh,
                state,
                dt=dt,
                nu=nu,
                gamma_penalty=gamma_penalty,
                force=force,
                pinned_nodes=pinned_nodes,
                pressure_pin_node=pressure_pin_node,
                inlet_faces=inlet_faces,
                outlet_faces=outlet_faces,
            )
        except Exception as exc:  # pragma: no cover - surfaced in tests
            return TransientFlowResult(
                final=state,
                history=history,
                converged=False,
                failure_message=str(exc),
            )
        history.append(diag)
    return TransientFlowResult(final=state, history=history, converged=True)
