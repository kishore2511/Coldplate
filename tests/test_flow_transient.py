import numpy as np
import pytest

from coldplate_studio.fr.flow_p1_transient import run_transient_stokes
from coldplate_studio.mesh.integral_pipeline import build_integral_fluid_duct
from coldplate_studio.simulation.flow_poiseuille import _duct_port_faces


@pytest.mark.gmsh
def test_transient_projection_reduces_divergence():
    model = build_integral_fluid_duct(mesh_size=0.06)
    inlet, outlet = _duct_port_faces(model, 1.0)
    port_nodes = {i for f in inlet + outlet for i in f}
    pinned = {int(i) for i in model.mesh.boundary_fluid_nodes if int(i) not in port_nodes}
    pin_node = int(model.mesh.fluid_tets[0, 0])
    result = run_transient_stokes(
        model.mesh,
        t_end=0.05,
        dt=0.01,
        nu=1e-3,
        gamma_penalty=1e3,
        pinned_nodes=pinned,
        pressure_pin_node=pin_node,
        inlet_faces=inlet,
        outlet_faces=outlet,
    )
    assert result.converged
    assert len(result.history) >= 5
    div_before = np.mean([h.div_before_projection for h in result.history])
    div_after = np.mean([h.div_after_projection for h in result.history])
    assert div_after <= div_before * 1.05
    last = result.history[-1]
    assert last.inlet_flux > 0
    assert last.outlet_flux < 0


@pytest.mark.gmsh
def test_transient_reproducible():
    model = build_integral_fluid_duct(mesh_size=0.07)
    inlet, outlet = _duct_port_faces(model, 1.0)
    port_nodes = {i for f in inlet + outlet for i in f}
    pinned = {int(i) for i in model.mesh.boundary_fluid_nodes if int(i) not in port_nodes}
    pin_node = int(model.mesh.fluid_tets[0, 0])
    kw = dict(
        t_end=0.02,
        dt=0.005,
        pinned_nodes=pinned,
        pressure_pin_node=pin_node,
        inlet_faces=inlet,
        outlet_faces=outlet,
    )
    r1 = run_transient_stokes(model.mesh, **kw)
    r2 = run_transient_stokes(model.mesh, **kw)
    np.testing.assert_allclose(r1.final.velocity, r2.final.velocity, rtol=0, atol=1e-12)


@pytest.mark.gmsh
def test_transient_dt_refinement_stable():
    model = build_integral_fluid_duct(mesh_size=0.07)
    inlet, outlet = _duct_port_faces(model, 1.0)
    port_nodes = {i for f in inlet + outlet for i in f}
    pinned = {int(i) for i in model.mesh.boundary_fluid_nodes if int(i) not in port_nodes}
    pin_node = int(model.mesh.fluid_tets[0, 0])
    coarse = run_transient_stokes(
        model.mesh,
        t_end=0.04,
        dt=0.02,
        pinned_nodes=pinned,
        pressure_pin_node=pin_node,
        inlet_faces=inlet,
        outlet_faces=outlet,
    )
    fine = run_transient_stokes(
        model.mesh,
        t_end=0.04,
        dt=0.01,
        pinned_nodes=pinned,
        pressure_pin_node=pin_node,
        inlet_faces=inlet,
        outlet_faces=outlet,
    )
    assert coarse.converged and fine.converged
    assert np.linalg.norm(fine.final.velocity) <= np.linalg.norm(coarse.final.velocity) * 2.5
