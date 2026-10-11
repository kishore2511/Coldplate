import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from coldplate_studio.fr.thermal_p1 import assemble_region_stiffness
from coldplate_studio.mesh.structured_tet_box import build_structured_fluid_box


def test_1d_steady_conduction_linear_profile():
    """T(x) linear between Dirichlet ends on structured bar (ny=nz=1)."""
    model = build_structured_fluid_box(length=1.0, height=0.05, width=0.05, nx=20, ny=1, nz=1)
    mesh = model.mesh
    k = assemble_region_stiffness(mesh, mesh.fluid_tets, conductivity=1.0)
    n = mesh.nodes.shape[0]
    b = np.zeros(n)
    x = mesh.nodes[:, 0]
    boundary = model.mesh.boundary_fluid_nodes
    a = k.tolil()
    for i in boundary:
        i = int(i)
        a[i, :] = 0
        a[:, i] = 0
        a[i, i] = 1.0
        b[i] = 300.0 + 100.0 * x[i]
    t = spsolve(a.tocsr(), b)
    x = mesh.nodes[:, 0]
    t_exact = 300.0 + 100.0 * x
    err = float(np.sqrt(np.mean((t - t_exact) ** 2)))
    assert err < 0.5


def test_transient_diffusion_fourier_mode_decay():
    """∂T/∂t = α ∂²T/∂x², T=sin(πx) on [0,1] with T=0 boundaries, analytic decay exp(-απ²t)."""
    from coldplate_studio.fr.thermal_p1 import assemble_region_lumped_mass

    model = build_structured_fluid_box(length=1.0, height=0.05, width=0.05, nx=24, ny=1, nz=1)
    mesh = model.mesh
    alpha = 0.1
    k = assemble_region_stiffness(mesh, mesh.fluid_tets, conductivity=alpha)
    m_lump = assemble_region_lumped_mass(mesh, mesh.fluid_tets, rho_cp=1.0)
    m = sparse.diags(m_lump, format="csr")
    x = mesh.nodes[:, 0]
    t0 = np.sin(np.pi * x)
    ends = (x < 1e-12) | (x > 1.0 - 1e-12)
    t0[ends] = 0.0
    dt = 0.01
    steps = 5
    t = t0.copy()
    for _ in range(steps):
        a = (m / dt + k).tolil()
        b = m.dot(t) / dt
        for i in np.where(ends)[0]:
            a[i, :] = 0
            a[:, i] = 0
            a[i, i] = 1.0
            b[i] = 0.0
        t = spsolve(a.tocsr(), b)
    t_analytic = np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * dt * steps)
    t_analytic[ends] = 0.0
    interior = ~ends
    err = float(np.sqrt(np.mean((t[interior] - t_analytic[interior]) ** 2)))
    assert err < 0.08
