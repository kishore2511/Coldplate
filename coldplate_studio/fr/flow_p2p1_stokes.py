"""Steady Stokes — Taylor–Hood P2 velocity / P1 pressure (inf-sup stable on tets)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from coldplate_studio.fr.flow_p2_saddle_solve import solve_stokes_saddle
from coldplate_studio.fr.flow_p2_velocity_field import P2VelocityField

from coldplate_studio.fr.flow_boundary_flux import (
    discrete_divergence_l2,
    fluid_boundary_outward_area_vectors,
    normalized_mass_imbalance,
    oriented_face_flux,
)
from coldplate_studio.fr.flow_p1_operators import fluid_node_mask
from coldplate_studio.fr.flow_p2_dof_map import P2VelocityDofMap, build_p2_velocity_dof_map, pressure_dofs_on_fluid_vertices
from coldplate_studio.fr.flow_p2p1_operators import (
    assemble_p2_body_force_rhs,
    assemble_p2_mms_body_force_rhs,
    assemble_p2p1_stokes,
)
from coldplate_studio.fr.tet_p2_basis import TET_P2_EDGE_PAIRS, p2_shape_values, tet_quadrature_4pt
from coldplate_studio.fr.tet_p1 import _tet_gradients
from coldplate_studio.mesh.integral_model import IntegralMesh


@dataclass
class P2P1StokesResult:
    velocity: np.ndarray  # (n_nodes, 3) vertex values only
    pressure: np.ndarray  # (n_nodes,) zero away from fluid pressure nodes
    divergence_l2: float
    pressure_residual_norm: float
    solver_residual_norm: float
    solver_method: str = "lsqr"
    p2_velocity_field: P2VelocityField | None = None
    inlet_flux_m3_s: float | None = None
    outlet_flux_m3_s: float | None = None
    mass_flux_relative_imbalance: float | None = None
    pressure_drop: float | None = None
    max_velocity_magnitude: float | None = None
    velocity_dofs: int = 0
    pressure_dofs: int = 0
    formulation: str = "p2p1_taylor_hood_stokes"


def _pin_p2_velocity_dofs(
    dof_map: P2VelocityDofMap,
    pinned_nodes: set[int],
    no_slip_nodes: set[int] | None = None,
) -> set[int]:
    """
    Dirichlet u=0 on pinned vertices and on edge-midpoint DOFs that lie on no-slip
    segments (any edge with at least one no-slip endpoint; FSI/solid uses pinned_nodes).
    """
    slip = no_slip_nodes if no_slip_nodes is not None else pinned_nodes
    pinned: set[int] = set()
    for node in pinned_nodes:
        vd = int(dof_map.vertex_dof[int(node)])
        if vd >= 0:
            for comp in range(3):
                pinned.add(3 * vd + comp)
    for (i, j), ed in dof_map.edge_dof.items():
        on_wall = (i in slip and j in slip) or (i in slip or j in slip)
        on_solid = i in pinned_nodes and j in pinned_nodes
        if on_wall or on_solid:
            for comp in range(3):
                pinned.add(3 * ed + comp)
    return pinned


def _expand_vertex_velocity(dof_map: P2VelocityDofMap, u_flat: np.ndarray, n_nodes: int) -> np.ndarray:
    vel = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        vd = int(dof_map.vertex_dof[i])
        if vd < 0:
            continue
        for comp in range(3):
            vel[i, comp] = u_flat[3 * vd + comp]
    return vel


def _p2_face_flux(
    mesh: IntegralMesh,
    dof_map: P2VelocityDofMap,
    u_flat: np.ndarray,
    face_keys: list[tuple[int, int, int]],
    area_vecs: dict,
) -> float:
    """Port flux using P2 velocity on boundary triangles."""
    flux = 0.0
    for key in face_keys:
        n_vec = area_vecs[key]
        i0, i1, i2 = key
        # Find fluid tet containing this face
        face_set = {i0, i1, i2}
        for tet in mesh.fluid_tets:
            nodes = {int(tet[k]) for k in range(4)}
            if not face_set.issubset(nodes):
                continue
            opp = next(int(tet[k]) for k in range(4) if int(tet[k]) not in face_set)
            # Barycentric coords on face: use 3-point rule on triangle
            pts = mesh.nodes[[i0, i1, i2]]
            loc_u = dof_map.local_dof_indices(tet)
            for bary in ((1.0 / 3, 1.0 / 3, 1.0 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)):
                w_tri = 1.0 / 12.0
                # Map face bary to tet bary with opp weight 0
                L = np.zeros(4)
                for k, node in enumerate((i0, i1, i2)):
                    for j in range(4):
                        if int(tet[j]) == node:
                            L[j] = bary[k]
                L_sum = L.sum()
                L[ [j for j in range(4) if int(tet[j]) == opp][0] ] = 1.0 - L_sum
                phi = p2_shape_values(L)
                u = np.zeros(3)
                for a in range(10):
                    ia = int(loc_u[a])
                    for comp in range(3):
                        u[comp] += u_flat[3 * ia + comp] * phi[a]
                flux += w_tri * float(np.dot(u, n_vec) / 2.0)
            break
    return flux


def solve_p2p1_stokes(
    mesh: IntegralMesh,
    *,
    nu: float = 1e-3,
    force_xyz: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    mms_force_at_point: callable | None = None,
    pinned_velocity_nodes: set[int],
    no_slip_nodes: set[int] | None = None,
    pressure_pin_node: int,
    inlet_faces: list[tuple[int, int, int]] | None = None,
    outlet_faces: list[tuple[int, int, int]] | None = None,
) -> P2P1StokesResult:
    n = mesh.nodes.shape[0]
    fluid_mask = fluid_node_mask(mesh)
    dof_map = build_p2_velocity_dof_map(mesh, fluid_mask)
    p_nodes = pressure_dofs_on_fluid_vertices(fluid_mask)
    k_uu, b_mat, _ = assemble_p2p1_stokes(mesh, nu, dof_map, p_nodes)
    if mms_force_at_point is not None:
        rhs_u = assemble_p2_mms_body_force_rhs(mesh, dof_map, mms_force_at_point)
    else:
        fx, fy, fz = force_xyz or (0.0, 0.0, 0.0)
        rhs_u = assemble_p2_body_force_rhs(mesh, dof_map, (fx, fy, fz))
    n_u = dof_map.n_velocity_dofs
    n_p = p_nodes.size
    dim_u = 3 * n_u
    z_p = sparse.csr_matrix((n_p, n_p))
    a = sparse.bmat(
        [[k_uu, b_mat.T], [b_mat, z_p]],
        format="lil",
    )
    rhs = np.zeros(dim_u + n_p)
    rhs[:dim_u] = rhs_u
    pinned = _pin_p2_velocity_dofs(dof_map, pinned_velocity_nodes, no_slip_nodes)
    for dof in sorted(pinned):
        a[dof, :] = 0
        a[:, dof] = 0
        a[dof, dof] = 1.0
        rhs[dof] = 0.0
    pin_local = int(np.where(p_nodes == int(pressure_pin_node))[0][0])
    prow = dim_u + pin_local
    a[prow, :] = 0
    a[:, prow] = 0
    a[prow, prow] = 1.0
    rhs[prow] = 0.0
    sol, method, _ = solve_stokes_saddle(a.tocsr(), rhs, pressure_pin_row=prow)
    u_flat = sol[:dim_u]
    p2_field = P2VelocityField(dof_map=dof_map, coefficients=u_flat.copy())
    p_coeff = sol[dim_u:]
    velocity = _expand_vertex_velocity(dof_map, u_flat, n)
    pressure = np.zeros(n)
    for i, node in enumerate(p_nodes):
        pressure[int(node)] = p_coeff[i]
    vol_sum = 0.0
    for tet in mesh.fluid_tets:
        vol, _ = _tet_gradients(mesh.nodes[tet])
        vol_sum += vol
    bu = b_mat @ u_flat
    div_l2 = float(np.linalg.norm(bu) / max(np.sqrt(vol_sum), 1e-30))
    pres_res = float(np.linalg.norm(bu) / max(np.linalg.norm(u_flat), 1e-12))
    # Vertex P1 diagnostic (legacy reporting only)
    _ = discrete_divergence_l2(mesh, velocity)
    res = float(np.linalg.norm(a @ sol - rhs) / max(np.linalg.norm(rhs), 1e-12))
    area_vecs = fluid_boundary_outward_area_vectors(mesh)
    phi_in = phi_out = imb = dp = None
    if inlet_faces is not None and outlet_faces is not None:
        in_keys = [tuple(sorted(f)) for f in inlet_faces]
        out_keys = [tuple(sorted(f)) for f in outlet_faces]
        phi_in = _p2_face_flux(mesh, dof_map, u_flat, in_keys, area_vecs)
        phi_out = _p2_face_flux(mesh, dof_map, u_flat, out_keys, area_vecs)
        imb = normalized_mass_imbalance(phi_in, phi_out)
        pin = np.mean(pressure[[i for f in inlet_faces for i in f]])
        pout = np.mean(pressure[[i for f in outlet_faces for i in f]])
        dp = float(pin - pout)
    umax = float(np.max(np.linalg.norm(velocity[fluid_mask], axis=1))) if fluid_mask.any() else 0.0
    return P2P1StokesResult(
        velocity=velocity,
        pressure=pressure,
        divergence_l2=div_l2,
        pressure_residual_norm=pres_res,
        solver_residual_norm=res,
        solver_method=method,
        p2_velocity_field=p2_field,
        inlet_flux_m3_s=phi_in,
        outlet_flux_m3_s=phi_out,
        mass_flux_relative_imbalance=imb,
        pressure_drop=dp,
        max_velocity_magnitude=umax,
        velocity_dofs=n_u,
        pressure_dofs=n_p,
    )


def solve_integral_channel_p2p1(
    mesh: IntegralMesh,
    boundaries,
    *,
    nu: float = 1e-3,
    dp_dx: float = -1.0,
) -> P2P1StokesResult:
    return solve_p2p1_stokes(
        mesh,
        nu=nu,
        force_xyz=(dp_dx, 0.0, 0.0),
        pinned_velocity_nodes=boundaries.pinned_velocity_nodes,
        no_slip_nodes=set(int(i) for i in boundaries.no_slip_nodes),
        pressure_pin_node=boundaries.fluid_pressure_reference_node,
        inlet_faces=boundaries.inlet_faces,
        outlet_faces=boundaries.outlet_faces,
    )
