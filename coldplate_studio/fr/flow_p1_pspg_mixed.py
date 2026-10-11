"""
Steady Stokes — P1 velocity / P1 pressure with PSPG stabilization (not Taylor–Hood, not FR).

Weak form:
  ν ∫ ∇u·∇v dΩ − ∫ p ∇·v dΩ = ∫ f·v dΩ
  ∫ q ∇·u dΩ + τ ∫ ∇q·∇p dΩ = 0   (PSPG, τ = h²/(4ν))

Pressure reference: pin one fluid pressure DOF to 0 when ``fluid_pressure_only=True``
(pressure unknowns live only on fluid nodes; solid-only nodes are not in the pressure block).
Legacy full-node pressure + solid p=0 pins: ``fluid_pressure_only=False``.
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
    add_pressure_traction_rhs,
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
    inlet_pressure: float | None = None,
    outlet_pressure: float | None = None,
    inlet_velocity_dirichlet: dict[int, tuple[float, float, float]] | None = None,
    fluid_pressure_only: bool = True,
) -> PSPGStokesResult:
    n = mesh.nodes.shape[0]
    dim_u = 3 * n
    d_full = assemble_weak_divergence(mesh)
    k_u = assemble_viscous_stiffness(mesh, nu)
    if momentum_grad_div_beta > 0.0:
        k_u = k_u + assemble_divergence_penalty(mesh, momentum_grad_div_beta)
    lp_full = assemble_pressure_laplacian(mesh)
    fluid_idx = np.array(sorted(np.where(fluid_node_mask(mesh))[0]), dtype=int)
    if fluid_pressure_only:
        d = d_full[fluid_idx, :]
        lp = lp_full[fluid_idx, :][:, fluid_idx]
        n_p = fluid_idx.size
        pin_p_nodes = {int(pressure_pin_node)}
    else:
        d = d_full
        lp = lp_full
        n_p = n
        pin_p_nodes = {int(pressure_pin_node)}
        if pressure_pin_nodes_extra:
            pin_p_nodes |= {int(i) for i in pressure_pin_nodes_extra}
    h = _mean_mesh_length(mesh)
    tau = (h**2) / (4.0 * nu)
    a = sparse.bmat(
        [
            [k_u, d.T],
            [d, -tau * lp],
        ],
        format="lil",
    )
    rhs = np.zeros(dim_u + n_p)
    rhs[:dim_u] = _body_force_rhs(mesh, n, force_x)
    area_vecs = fluid_boundary_outward_area_vectors(mesh)
    if inlet_faces is not None and inlet_pressure is not None:
        in_keys = [tuple(sorted(f)) for f in inlet_faces]
        add_pressure_traction_rhs(rhs[:dim_u], n, in_keys, inlet_pressure, area_vecs)
    if outlet_faces is not None and outlet_pressure is not None:
        out_keys = [tuple(sorted(f)) for f in outlet_faces]
        add_pressure_traction_rhs(rhs[:dim_u], n, out_keys, outlet_pressure, area_vecs)
    if periodic_pairs:
        _add_periodic_velocity_penalty(a, periodic_pairs, n, periodic_gamma)
    pinned = wall_pinned_dofs(n, pinned_velocity_nodes)
    inlet_nodes_set = set(inlet_velocity_dirichlet or {})
    for dof in sorted(pinned):
        node = dof % n
        comp = dof // n
        if node in inlet_nodes_set and comp < 3:
            continue
        a[dof, :] = 0
        a[:, dof] = 0
        a[dof, dof] = 1.0
        rhs[dof] = 0.0
    if inlet_velocity_dirichlet:
        for node, (ux, uy, uz) in inlet_velocity_dirichlet.items():
            for comp, val in enumerate((ux, uy, uz)):
                dof = comp * n + int(node)
                a[dof, :] = 0
                a[:, dof] = 0
                a[dof, dof] = 1.0
                rhs[dof] = val
    if fluid_pressure_only:
        pin_local = int(np.where(fluid_idx == int(pressure_pin_node))[0][0])
        pin_p_nodes = {pin_local}
    for pnode in sorted(pin_p_nodes):
        p_row = dim_u + pnode
        a[p_row, :] = 0
        a[:, p_row] = 0
        a[p_row, p_row] = 1.0
        rhs[p_row] = 0.0
    sol = spsolve(a.tocsr(), rhs)
    u_flat = sol[:dim_u]
    p_red = sol[dim_u:]
    pressure = np.zeros(n)
    if fluid_pressure_only:
        pressure[fluid_idx] = p_red
    else:
        pressure = p_red
    velocity = unflat_velocity(u_flat, n)
    div_l2 = discrete_divergence_l2(mesh, velocity)
    if fluid_pressure_only:
        p_res = float(np.linalg.norm(d_full[fluid_idx, :] @ u_flat) / max(np.linalg.norm(u_flat), 1e-12))
    else:
        p_res = float(np.linalg.norm(d @ u_flat) / max(np.linalg.norm(u_flat), 1e-12))
    res = float(np.linalg.norm(a @ sol - rhs) / max(np.linalg.norm(rhs), 1e-12))
    fluid = fluid_node_mask(mesh)
    umax = float(np.max(np.linalg.norm(velocity[fluid], axis=1))) if fluid.any() else 0.0
    phi_in = phi_out = None
    imb = None
    dp = None
    if inlet_faces is not None and outlet_faces is not None:
        in_keys = [tuple(sorted(f)) for f in inlet_faces]
        out_keys = [tuple(sorted(f)) for f in outlet_faces]
        phi_in = oriented_face_flux(mesh, velocity, in_keys, area_vecs)
        phi_out = oriented_face_flux(mesh, velocity, out_keys, area_vecs)
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
    drive: str = "body_force_legacy",
    inlet_pressure_pa: float = 1.0,
    outlet_pressure_pa: float = 0.0,
    dp_dx: float = -1.0,
    momentum_grad_div_beta: float | None = None,
    scale_grad_div_with_mesh: bool = True,
    inlet_u_max: float = 0.5,
    channel_width_y: float = 0.2,
) -> PSPGStokesResult:
    """
    Fin-channel steady Stokes (PSPG).

    **body_force_legacy (default):** uniform f_x=dp_dx, open inlet/outlet (natural traction),
    no-slip walls, PSPG continuity (τ=h²/(4ν)), momentum grad-div β (default 100, same scale as penalty).

    **inlet_parabolic:** parabolic u_x on inlet face nodes, β=0, no body force (experimental on fin mesh).

    **pressure_traction:** weak ∫ p n·v dΓ with prescribed inlet/outlet pressure; β=0.
    """
    from coldplate_studio.fr.flow_stabilization import scaled_grad_div_beta
    from coldplate_studio.simulation.flow_inlet_profile import parabolic_inlet_ux

    pin_extra: set[int] | None = set(int(i) for i in boundaries.solid_only_nodes)
    use_fluid_p = drive != "body_force_legacy" or scale_grad_div_with_mesh
    inlet_vel: dict[int, tuple[float, float, float]] | None = None
    force_x = 0.0
    beta = 0.0
    p_in = p_out = None
    if drive == "body_force_legacy":
        force_x = dp_dx
        if momentum_grad_div_beta is None:
            beta = scaled_grad_div_beta(mesh) if scale_grad_div_with_mesh else 100.0
        else:
            beta = momentum_grad_div_beta
    elif drive == "inlet_parabolic":
        inlet_nodes = np.array(sorted({i for f in boundaries.inlet_faces for i in f}), dtype=int)
        inlet_vel = parabolic_inlet_ux(mesh, inlet_nodes, width_y=channel_width_y, u_max=inlet_u_max)
    elif drive == "pressure_traction":
        force_x = 0.0
        p_in = inlet_pressure_pa
        p_out = outlet_pressure_pa
    return solve_stokes_pspg(
        mesh,
        nu=nu,
        force_x=force_x,
        pinned_velocity_nodes=boundaries.pinned_velocity_nodes,
        pressure_pin_node=boundaries.fluid_pressure_reference_node,
        pressure_pin_nodes_extra=pin_extra if not use_fluid_p else None,
        fluid_pressure_only=use_fluid_p,
        momentum_grad_div_beta=beta,
        inlet_faces=boundaries.inlet_faces,
        outlet_faces=boundaries.outlet_faces,
        inlet_pressure=p_in,
        outlet_pressure=p_out,
        inlet_velocity_dirichlet=inlet_vel,
    )
