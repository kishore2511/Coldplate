import numpy as np

from coldplate_studio.fr.flow_p1_pspg_mixed import solve_stokes_pspg
from coldplate_studio.fr.flow_periodic_stokes import _y_wall_nodes, pair_x_periodic_nodes
from coldplate_studio.mesh.structured_tet_box import build_structured_fluid_box
from coldplate_studio.simulation.flow_manufactured import (
    manufactured_body_force_rhs,
    manufactured_velocity_field,
)


def test_pspg_periodic_channel_profile_and_divergence():
    model = build_structured_fluid_box(nx=10, ny=5, nz=2, width=0.2, height=0.1)
    mesh = model.mesh
    n = mesh.nodes.shape[0]
    walls = set(int(i) for i in _y_wall_nodes(mesh, 0.1))
    pairs = pair_x_periodic_nodes(mesh.nodes, 1.0)
    pin = int(mesh.fluid_tets[0, 0])
    r = solve_stokes_pspg(
        mesh,
        force_x=1.0,
        pinned_velocity_nodes=walls,
        pressure_pin_node=pin,
        periodic_pairs=pairs,
    )
    g = 1.0
    u_exact = manufactured_velocity_field(mesh.nodes, length=1.0, height=0.1, width=0.2, grad_p=-1.0, nu=1e-3)[:, 0]
    # parabolic reference for developed channel
    y = mesh.nodes[:, 1]
    u_par = (g / (2e-3)) * y * (0.1 - y)
    interior = (y > 0.01) & (y < 0.09)
    err = float(np.sqrt(np.mean((r.velocity[interior, 0] - u_par[interior]) ** 2)))
    assert err < 5e-3
    assert r.divergence_l2 < 0.05
    assert r.solver_residual_norm < 1e-6


def test_pspg_manufactured_rhs_residual():
    """PSPG with MMS forcing (β=0 equivalent forcing) — solver residual gate."""
    model = build_structured_fluid_box(nx=6, ny=4, nz=2, width=0.2, height=0.1)
    mesh = model.mesh
    n = mesh.nodes.shape[0]
    walls = set(int(i) for i in mesh.boundary_fluid_nodes)
    pin = int(mesh.fluid_tets[0, 0])
    # Use manufactured body force as external force via custom solve — reuse operators
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from coldplate_studio.fr.flow_p1_operators import (
        assemble_pressure_laplacian,
        assemble_viscous_stiffness,
        assemble_weak_divergence,
    )
    from coldplate_studio.fr.flow_p1_pspg_mixed import _mean_mesh_length

    d = assemble_weak_divergence(mesh)
    k_u = assemble_viscous_stiffness(mesh, 1e-3)
    lp = assemble_pressure_laplacian(mesh)
    h = _mean_mesh_length(mesh)
    tau = (h**2) / (4.0 * 1e-3)
    dim_u = 3 * n
    a = sparse.bmat([[k_u, d.T], [d, -tau * lp]], format="lil")
    rhs = np.zeros(dim_u + n)
    rhs[:dim_u] = manufactured_body_force_rhs(mesh, n, 1.0, 0.1, 0.2, -1.0, 1e-3)
    from coldplate_studio.fr.flow_p1_operators import wall_pinned_dofs

    pinned = wall_pinned_dofs(n, walls)
    for dof in sorted(pinned):
        a[dof, :] = 0
        a[:, dof] = 0
        a[dof, dof] = 1.0
        rhs[dof] = 0.0
    prow = dim_u + pin
    a[prow, :] = 0
    a[:, prow] = 0
    a[prow, prow] = 1.0
    rhs[prow] = 0.0
    sol = spsolve(a.tocsr(), rhs)
    u = sol[:dim_u]
    u_exact = manufactured_velocity_field(mesh.nodes, length=1.0, height=0.1, width=0.2, grad_p=-1.0, nu=1e-3)
    err = float(np.sqrt(np.mean((sol[:n] - u_exact[:, 0]) ** 2)))
    assert err < 0.05
    assert float(np.linalg.norm(a @ sol - rhs)) < 1e-5
