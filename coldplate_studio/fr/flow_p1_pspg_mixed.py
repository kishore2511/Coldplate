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
    assemble_divergence_penalty,
    assemble_pressure_laplacian,
    assemble_viscous_stiffness,
    assemble_weak_divergence,
    fluid_node_mask,
    unflat_velocity,
    wall_pinned_dofs,
)
from coldplate_studio.fr.flow_boundary_flux import (
    discrete_divergence_l2,
    fluid_boundary_outward_area_vectors,
    normalized_mass_imbalance,
    oriented_face_flux,
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
    tau_pspg: float
    mean_mesh_length: float
    inlet_flux_m3_s: float | None = None
    outlet_flux_m3_s: float | None = None
    mass_flux_relative_imbalance: float | None = None
    pressure_drop: float | None = None
    max_velocity_magnitude: float | None = None


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
    pressure_pin_nodes_extra: set[int] | None = None,
    momentum_grad_div_beta: float = 0.0,
    periodic_pairs: list[tuple[int, int]] | None = None,
    periodic_gamma: float = 1e4,
    inlet_faces: list[tuple[int, int, int]] | None = None,
    outlet_faces: list[tuple[int, int, int]] | None = None,
) -> PSPGStokesResult:
    n = mesh.nodes.shape[0]
    dim_u = 3 * n
    d = assemble_weak_divergence(mesh)
    k_u = assemble_viscous_stiffness(mesh, nu)
    if momentum_grad_div_beta > 0.0:
        k_u = k_u + assemble_divergence_penalty(mesh, momentum_grad_div_beta)
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
    pin_p = {int(pressure_pin_node)}
    if pressure_pin_nodes_extra:
        pin_p |= {int(i) for i in pressure_pin_nodes_extra}
    for pnode in sorted(pin_p):
        p_row = dim_u + pnode
        a[p_row, :] = 0
        a[:, p_row] = 0
        a[p_row, p_row] = 1.0
        rhs[p_row] = 0.0
    sol = spsolve(a.tocsr(), rhs)
    u_flat = sol[:dim_u]
    pressure = sol[dim_u:]
    velocity = unflat_velocity(u_flat, n)
    div_l2 = discrete_divergence_l2(mesh, velocity)
    p_res = float(np.linalg.norm(d @ u_flat) / max(np.linalg.norm(u_flat), 1e-12))
    res = float(np.linalg.norm(a @ sol - rhs) / max(np.linalg.norm(rhs), 1e-12))
    fluid = fluid_node_mask(mesh)
    umax = float(np.max(np.linalg.norm(velocity[fluid], axis=1))) if fluid.any() else 0.0
    phi_in = phi_out = None
    imb = None
    dp = None
    if inlet_faces is not None and outlet_faces is not None:
        area = fluid_boundary_outward_area_vectors(mesh)
        in_keys = [tuple(sorted(f)) for f in inlet_faces]
        out_keys = [tuple(sorted(f)) for f in outlet_faces]
        phi_in = oriented_face_flux(mesh, velocity, in_keys, area)
        phi_out = oriented_face_flux(mesh, velocity, out_keys, area)
        imb = normalized_mass_imbalance(phi_in, phi_out)
        pin = np.mean(pressure[[i for f in inlet_faces for i in f]])
        pout = np.mean(pressure[[i for f in outlet_faces for i in f]])
        dp = float(pin - pout)
    return PSPGStokesResult(
        velocity=velocity,
        pressure=pressure,
        divergence_l2=div_l2,
        pressure_residual_norm=p_res,
        solver_residual_norm=res,
        tau_pspg=tau,
        mean_mesh_length=h,
        inlet_flux_m3_s=phi_in,
        outlet_flux_m3_s=phi_out,
        mass_flux_relative_imbalance=imb,
        pressure_drop=dp,
        max_velocity_magnitude=umax,
    )


def solve_integral_channel_pspg(
    mesh: IntegralMesh,
    boundaries,
    *,
    nu: float = 1e-3,
    dp_dx: float = -1.0,
    momentum_grad_div_beta: float = 100.0,
) -> PSPGStokesResult:
    """
    Fin-channel PSPG with open inlet/outlet (natural traction) and body-force drive.

    On conforming fluid–solid meshes, pressure DOFs on solid-only nodes are pinned to zero.
    ``momentum_grad_div_beta`` augments the momentum block for open-duct mass control (documented;
    periodic/MMS benchmarks use beta=0).
    """
    solid_pins = set(int(i) for i in boundaries.solid_only_nodes)
    return solve_stokes_pspg(
        mesh,
        nu=nu,
        force_x=dp_dx,
        pinned_velocity_nodes=boundaries.pinned_velocity_nodes,
        pressure_pin_node=boundaries.fluid_pressure_reference_node,
        pressure_pin_nodes_extra=solid_pins,
        momentum_grad_div_beta=momentum_grad_div_beta,
        inlet_faces=boundaries.inlet_faces,
        outlet_faces=boundaries.outlet_faces,
    )
