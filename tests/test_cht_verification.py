import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from coldplate_studio.fr.cht_energy import transient_energy_balance
from coldplate_studio.fr.cht_transient import ConjugateHeatTransferSolver, ThermalBoundaryConditions, ThermalMaterialProperties
from coldplate_studio.fr.thermal_p1 import assemble_region_stiffness, interface_heat_flux
from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.mesh.structured_tet_box import build_structured_fluid_box
from coldplate_studio.simulation.flow_channel import run_channel_flow
from coldplate_studio.simulation.flow_channel_boundaries import (
    build_integral_channel_flow_boundaries,
    channel_port_fluid_nodes,
)


def _apply_dirichlet_rows(a: sparse.spmatrix, b: np.ndarray, node_values: list[tuple[int, float]]):
    """Eliminate Dirichlet nodes into the RHS (required when zeroing stiffness columns)."""
    a = a.tolil()
    b = b.copy()
    for i, val in node_values:
        b -= a[:, i].toarray().ravel() * val
        a[:, i] = 0
        a[i, :] = 0
        a[i, i] = 1.0
        b[i] = val
    return a.tocsr(), b


def test_two_material_steady_conduction_interface():
    """k1/k2 composite bar; interface T matches 1D series solution."""
    model = build_structured_fluid_box(length=1.0, height=0.05, width=0.05, nx=20, ny=1, nz=1)
    mesh = model.mesh
    n = mesh.nodes.shape[0]
    k1, k2 = 1.0, 5.0
    x = mesh.nodes[:, 0]
    rows, cols, data = [], [], []
    for tet in mesh.fluid_tets:
        from coldplate_studio.fr.tet_p1 import _tet_gradients

        vol, grads = _tet_gradients(mesh.nodes[tet])
        cx = mesh.nodes[tet, 0].mean()
        k = k1 if cx < 0.5 else k2
        for i in range(4):
            for j in range(4):
                rows.append(tet[i])
                cols.append(tet[j])
                data.append(k * vol * float(np.dot(grads[i], grads[j])))
    k_mat = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    b = np.zeros(n)
    left = np.where(x < 1e-12)[0]
    right = np.where(x > 1.0 - 1e-12)[0]
    pairs = [(int(i), 300.0) for i in left] + [(int(i), 400.0) for i in right]
    a, b = _apply_dirichlet_rows(k_mat, b, pairs)
    t = spsolve(a, b)
    l1, l2 = 0.5, 0.5
    heat_flux = (400.0 - 300.0) / (l1 / k1 + l2 / k2)
    t_interface = 300.0 + heat_flux * l1 / k1
    mid = np.where(np.abs(x - 0.5) < 1e-2)[0]
    err = abs(float(np.mean(t[mid])) - t_interface)
    assert err < 1.0


@pytest.mark.gmsh
def test_integral_channel_interface_flux_continuity_linear_z():
    """Same k on both sides; T linear in z → flux continuity on horizontal fin interfaces."""
    model = build_integral_channel_model(mesh_size=0.07)
    z = model.mesh.nodes[:, 2]
    t = 300.0 + 100.0 * z
    qf, qs = interface_heat_flux(model.mesh, t, 1.0, 1.0)
    assert abs(qf + qs) / max(abs(qf), abs(qs), 1e-12) < 1e-6


@pytest.mark.gmsh
def test_cht_transient_energy_balance_relative_residual():
    model = build_integral_channel_model(mesh_size=0.08)
    run = run_channel_flow(model, formulation="pspg")
    boundaries = build_integral_channel_flow_boundaries(model)
    inlet, outlet = channel_port_fluid_nodes(boundaries)
    solver = ConjugateHeatTransferSolver(
        model.mesh,
        run.flow.velocity,
        ThermalMaterialProperties(),
        ThermalBoundaryConditions(fluid_inlet_nodes=inlet, fluid_inlet_temperature_k=360.0),
    )
    t_prev = solver.state.temperature.copy()
    dt = 0.02
    solver.step(dt)
    bal = transient_energy_balance(
        model.mesh,
        t_prev,
        solver.state.temperature,
        dt,
        rho_cp_fluid=solver.props.rho_cp_fluid,
        rho_cp_solid=solver.props.rho_cp_solid,
        k_fluid=solver.props.k_fluid,
        k_solid=solver.props.k_solid,
        inlet_nodes=inlet,
        inlet_temperature=360.0,
        outlet_nodes=outlet,
        velocity=run.flow.velocity,
    )
    assert bal.relative_residual < 0.35
