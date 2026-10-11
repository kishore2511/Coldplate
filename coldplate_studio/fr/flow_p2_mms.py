"""Manufactured Stokes MMS for Taylor–Hood P2–P1 verification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.simulation.flow_manufactured import manufactured_velocity_field

from coldplate_studio.fr.flow_p2_mms_eval import volume_l2_velocity_error


def manufactured_pressure_field(
    nodes: np.ndarray,
    *,
    grad_p: float,
) -> np.ndarray:
    """p(x) = grad_p * x (reference; mean removed in error norms)."""
    return grad_p * nodes[:, 0]


def stokes_mms_force_vector(
    nodes: np.ndarray,
    *,
    length: float,
    height: float,
    width: float,
    grad_p: float,
    nu: float,
) -> np.ndarray:
    """
    Body force for steady Stokes weak form ν∫∇u·∇v − ∫p div v = ∫f·v:

    f = −νΔu + ∇p  (x-component: f_x = −νΔu_x + ∂p/∂x).
    """
    g = -grad_p
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    u0 = g / (2.0 * nu)
    sinx = np.sin(np.pi * x / length)
    lap = u0 * sinx * (
        -2.0 * z * (width - z)
        - 2.0 * y * (height - y)
        - (np.pi**2 / length**2) * y * (height - y) * z * (width - z)
    )
    fx = -nu * lap + grad_p
    return np.column_stack((fx, np.zeros_like(fx), np.zeros_like(fx)))


@dataclass
class P2MMSReport:
    mesh_level: int
    velocity_dofs: int
    pressure_dofs: int
    ux_l2: float
    ux_l2_volume_p2: float
    ux_h1_semi: float
    p_l2_mean_zero: float
    p_l2_relative: float
    divergence_l2: float
    solver_residual: float
    solver_method: str
    h_eff: float


def parabolic_channel_ux(y: np.ndarray, height: float, u_max: float) -> np.ndarray:
    """Divergence-free channel profile u_x = u_max * 4 y (H−y) / H², zero on y=0,H."""
    return u_max * 4.0 * y * (height - y) / max(height**2, 1e-30)


def stokes_parabolic_channel_force_y(
    y: float,
    *,
    height: float,
    u_max: float,
    nu: float,
    grad_p: float,
) -> np.ndarray:
    """f_x = −ν ∂²u_x/∂y² + grad_p with u_x parabolic in y only."""
    d2ux_dy2 = -8.0 * u_max / max(height**2, 1e-30)
    fx = -nu * d2ux_dy2 + grad_p
    return np.array([fx, 0.0, 0.0])


def run_p2p1_mms_study(
    *,
    length: float = 1.0,
    height: float = 0.1,
    width: float = 0.2,
    nu: float = 1e-3,
    grad_p: float = -1.0,
    nx_levels: tuple[int, ...] = (4, 6, 8),
    uniform_refinement: bool = True,
) -> list[P2MMSReport]:
    from coldplate_studio.fr.flow_p1_operators import fluid_node_mask
    from coldplate_studio.fr.flow_p2_dof_map import build_p2_velocity_dof_map, pressure_dofs_on_fluid_vertices
    from coldplate_studio.fr.flow_p2p1_stokes import solve_p2p1_stokes
    from coldplate_studio.mesh.structured_tet_box import build_structured_fluid_box

    reports: list[P2MMSReport] = []
    for nx in nx_levels:
        if uniform_refinement:
            # nx is cells along x; choose ny,nz so cell size ~ length/nx in every direction.
            h = length / nx
            ny = max(2, int(round(height / h)))
            nz = max(2, int(round(width / h)))
        else:
            ny = max(3, nx // 2)
            nz = 2
        model = build_structured_fluid_box(
            nx=nx, ny=ny, nz=nz, length=length, width=width, height=height
        )
        mesh = model.mesh
        walls = set(int(i) for i in mesh.boundary_fluid_nodes)
        pin = int(mesh.fluid_tets[0, 0])

        def force_at(x: np.ndarray) -> np.ndarray:
            return stokes_mms_force_vector(
                x.reshape(1, 3),
                length=length,
                height=height,
                width=width,
                grad_p=grad_p,
                nu=nu,
            )[0]

        r = solve_p2p1_stokes(
            mesh,
            nu=nu,
            mms_force_at_point=force_at,
            pinned_velocity_nodes=walls,
            no_slip_nodes=walls,
            pressure_pin_node=pin,
        )
        u_ex = manufactured_velocity_field(
            mesh.nodes, length=length, height=height, width=width, grad_p=grad_p, nu=nu
        )
        p_ex = manufactured_pressure_field(mesh.nodes, grad_p=grad_p)
        fluid = fluid_node_mask(mesh)
        ux_err = r.velocity[fluid, 0] - u_ex[fluid, 0]
        ux_l2 = float(np.sqrt(np.mean(ux_err**2)))
        dof_map = build_p2_velocity_dof_map(mesh, fluid)

        def u_ex_at(x: np.ndarray) -> np.ndarray:
            return manufactured_velocity_field(
                x.reshape(1, 3),
                length=length,
                height=height,
                width=width,
                grad_p=grad_p,
                nu=nu,
            )[0]

        ux_vol = volume_l2_velocity_error(
            mesh,
            dof_map,
            r.p2_velocity_field.coefficients,
            u_ex,
            u_exact_at_point=u_ex_at,
        )
        # Semi-H1: L2 of grad(ux) error on P1 tet-wise constant grad
        from coldplate_studio.fr.tet_p1 import _tet_gradients

        h1_sq = 0.0
        vol_sum = 0.0
        for tet in mesh.fluid_tets:
            vol, grads = _tet_gradients(mesh.nodes[tet])
            g_num = sum(grads[k, 0] * r.velocity[tet[k], 0] for k in range(4))
            g_ex = sum(grads[k, 0] * u_ex[tet[k], 0] for k in range(4))
            for d in range(3):
                g_num_d = sum(grads[k, d] * r.velocity[tet[k], 0] for k in range(4))
                g_ex_d = sum(grads[k, d] * u_ex[tet[k], 0] for k in range(4))
            g_num_x = sum(grads[k, 0] * r.velocity[tet[k], 0] for k in range(4))
            g_ex_x = sum(grads[k, 0] * u_ex[tet[k], 0] for k in range(4))
            h1_sq += vol * (g_num_x - g_ex_x) ** 2
            vol_sum += vol
        ux_h1 = float(np.sqrt(h1_sq / max(vol_sum, 1e-30)))
        p_nodes = pressure_dofs_on_fluid_vertices(fluid)
        p_h = r.pressure[p_nodes]
        p_ref = p_ex[p_nodes] - np.mean(p_ex[p_nodes])
        p_l2 = float(np.sqrt(np.mean((p_h - np.mean(p_h) - p_ref) ** 2)))
        p_norm = float(np.sqrt(np.mean(p_ref**2)))
        p_rel = p_l2 / max(p_norm, 1e-30)
        h_eff = length / nx  # reference cell size along streamwise direction
        reports.append(
            P2MMSReport(
                mesh_level=nx,
                velocity_dofs=r.velocity_dofs,
                pressure_dofs=r.pressure_dofs,
                ux_l2=ux_l2,
                ux_l2_volume_p2=ux_vol,
                ux_h1_semi=ux_h1,
                p_l2_mean_zero=p_l2,
                p_l2_relative=p_rel,
                divergence_l2=r.divergence_l2,
                solver_residual=r.solver_residual_norm,
                solver_method=r.solver_method,
                h_eff=h_eff,
            )
        )
    return reports


def format_p2_mms_table(reports: list[P2MMSReport]) -> str:
    lines = [
        "| nx | h | ‖ux‖_L2 vert | ‖u‖_L2 vol P2 | ‖ux‖_H1 semi | ‖p‖_L2 rel | ‖div‖ | residual |",
        "|----|---|--------------|---------------|--------------|------------|-------|----------|",
    ]
    for r in reports:
        lines.append(
            f"| {r.mesh_level} | {r.h_eff:.4f} | {r.ux_l2:.4e} | {r.ux_l2_volume_p2:.4e} | "
            f"{r.ux_h1_semi:.4e} | {r.p_l2_relative:.4e} | {r.divergence_l2:.2e} | {r.solver_residual:.2e} |"
        )
    return "\n".join(lines)


def run_p2p1_parabolic_channel_mms_study(
    *,
    height: float = 0.1,
    width: float = 0.2,
    length: float = 1.0,
    nu: float = 1e-3,
    grad_p: float = -1.0,
    u_max: float | None = None,
    ny_levels: tuple[int, ...] = (4, 8, 16),
) -> list[P2MMSReport]:
    """
    **Not a valid MMS for this solver:** parabolic u_x(y) is incompatible with all-face
    no-slip on a 3D box (u_x ≠ 0 on x/z walls). Retained for debugging only — do not use
    for convergence claims (see D-008).
    """
    from coldplate_studio.fr.flow_p1_operators import fluid_node_mask
    from coldplate_studio.fr.flow_p2_dof_map import build_p2_velocity_dof_map, pressure_dofs_on_fluid_vertices
    from coldplate_studio.fr.flow_p2p1_stokes import solve_p2p1_stokes
    from coldplate_studio.mesh.structured_tet_box import build_structured_fluid_box

    if u_max is None:
        u_max = (-grad_p) / (2.0 * nu) * height**2 / 12.0 * 4.0  # scale with Poiseuille magnitude order

    reports: list[P2MMSReport] = []
    for ny in ny_levels:
        model = build_structured_fluid_box(nx=4, ny=ny, nz=2, width=width, height=height, length=length)
        mesh = model.mesh
        walls = set(int(i) for i in mesh.boundary_fluid_nodes)
        pin = int(mesh.fluid_tets[0, 0])
        y_nodes = mesh.nodes[:, 1]

        def force_at(x: np.ndarray) -> np.ndarray:
            return stokes_parabolic_channel_force_y(
                float(x[1]), height=height, u_max=u_max, nu=nu, grad_p=grad_p
            )

        r = solve_p2p1_stokes(
            mesh,
            nu=nu,
            mms_force_at_point=force_at,
            pinned_velocity_nodes=walls,
            no_slip_nodes=walls,
            pressure_pin_node=pin,
        )
        u_ex = np.zeros_like(mesh.nodes)
        u_ex[:, 0] = parabolic_channel_ux(y_nodes, height, u_max)
        fluid = fluid_node_mask(mesh)
        ux_l2 = float(np.sqrt(np.mean((r.velocity[fluid, 0] - u_ex[fluid, 0]) ** 2)))
        dof_map = build_p2_velocity_dof_map(mesh, fluid)
        def u_ex_at(x: np.ndarray) -> np.ndarray:
            out = np.zeros(3)
            out[0] = parabolic_channel_ux(float(x[1]), height, u_max)
            return out

        ux_vol = volume_l2_velocity_error(
            mesh, dof_map, r.p2_velocity_field.coefficients, u_ex, u_exact_at_point=u_ex_at
        )
        reports.append(
            P2MMSReport(
                mesh_level=ny,
                velocity_dofs=r.velocity_dofs,
                pressure_dofs=r.pressure_dofs,
                ux_l2=ux_l2,
                ux_l2_volume_p2=ux_vol,
                ux_h1_semi=0.0,
                p_l2_mean_zero=0.0,
                p_l2_relative=0.0,
                divergence_l2=r.divergence_l2,
                h_eff=height / ny,
                solver_residual=r.solver_residual_norm,
                solver_method=r.solver_method,
            )
        )
    return reports


def run_p2p1_polynomial_channel_mms(
    *,
    height: float = 0.1,
    width: float = 0.2,
    length: float = 1.0,
    nu: float = 1e-3,
    grad_p: float = -1.0,
    ny_levels: tuple[int, ...] = (2, 4, 8),
) -> list[P2MMSReport]:
    """
    u_x = y (H−y) is in P2, div-free, p = grad_p * x is in P1.

    Dirichlet data is the exact trace (nonzero on x/z faces). This is the
    acceptance MMS for pressure: the sin-π box field is not in the FE space.
    """
    from coldplate_studio.fr.flow_p1_operators import fluid_node_mask
    from coldplate_studio.fr.flow_p2_dof_map import build_p2_velocity_dof_map, pressure_dofs_on_fluid_vertices
    from coldplate_studio.fr.flow_p2p1_stokes import solve_p2p1_stokes
    from coldplate_studio.mesh.structured_tet_box import build_structured_fluid_box

    reports: list[P2MMSReport] = []
    for ny in ny_levels:
        h = height / ny
        model = build_structured_fluid_box(nx=4, ny=ny, nz=2, length=length, width=width, height=height)
        mesh = model.mesh
        walls = set(int(i) for i in mesh.boundary_fluid_nodes)
        pin = int(mesh.fluid_tets[0, 0])

        def u_exact_vec(x: np.ndarray) -> np.ndarray:
            out = np.zeros(3)
            out[0] = float(x[1]) * (height - float(x[1]))
            return out

        def force_at(x: np.ndarray) -> np.ndarray:
            # ∂²u_x/∂y² = -2, f = −ν Δu + ∇p
            return np.array([-nu * (-2.0) + grad_p, 0.0, 0.0])

        r = solve_p2p1_stokes(
            mesh,
            nu=nu,
            mms_force_at_point=force_at,
            dirichlet_velocity_at_point=u_exact_vec,
            pinned_velocity_nodes=walls,
            no_slip_nodes=walls,
            pressure_pin_node=pin,
        )
        u_ex = np.zeros_like(mesh.nodes)
        u_ex[:, 0] = mesh.nodes[:, 1] * (height - mesh.nodes[:, 1])
        fluid = fluid_node_mask(mesh)
        ux_l2 = float(np.sqrt(np.mean((r.velocity[fluid, 0] - u_ex[fluid, 0]) ** 2)))
        dof_map = build_p2_velocity_dof_map(mesh, fluid)
        ux_vol = volume_l2_velocity_error(
            mesh, dof_map, r.p2_velocity_field.coefficients, u_ex, u_exact_at_point=u_exact_vec
        )
        p_ex = manufactured_pressure_field(mesh.nodes, grad_p=grad_p)
        p_nodes = pressure_dofs_on_fluid_vertices(fluid)
        # Boundary pressure DOFs whose velocity support is fully Dirichlet are not
        # controlled by the saddle (LSQR sets them to 0). Score interior vertices.
        interior = [int(i) for i in p_nodes if int(i) not in walls]
        p_shift = float(p_ex[pin])
        p_h = r.pressure[interior]
        p_ref = p_ex[interior] - p_shift
        p_l2 = float(np.sqrt(np.mean((p_h - p_ref) ** 2)))
        p_norm = float(np.sqrt(np.mean(p_ref**2)))
        reports.append(
            P2MMSReport(
                mesh_level=ny,
                velocity_dofs=r.velocity_dofs,
                pressure_dofs=r.pressure_dofs,
                ux_l2=ux_l2,
                ux_l2_volume_p2=ux_vol,
                ux_h1_semi=0.0,
                p_l2_mean_zero=p_l2,
                p_l2_relative=p_l2 / max(p_norm, 1e-30),
                divergence_l2=r.divergence_l2,
                solver_residual=r.solver_residual_norm,
                solver_method=r.solver_method,
                h_eff=h,
            )
        )
    return reports


def estimate_ux_volume_convergence_order(reports: list[P2MMSReport]) -> float | None:
    """Observed order from last two mesh levels using ``ux_l2_volume_p2``."""
    if len(reports) < 2:
        return None
    r0, r1 = reports[-2], reports[-1]
    if r0.ux_l2_volume_p2 <= 0 or r1.ux_l2_volume_p2 <= 0:
        return None
    h0 = 1.0 / r0.mesh_level
    h1 = 1.0 / r1.mesh_level
    return float(np.log(r0.ux_l2_volume_p2 / r1.ux_l2_volume_p2) / np.log(h0 / h1))
