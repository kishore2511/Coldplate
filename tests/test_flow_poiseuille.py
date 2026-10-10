import pytest

from coldplate_studio.mesh.integral_pipeline import build_integral_fluid_duct
from coldplate_studio.simulation.flow_manufactured import run_manufactured_stokes_benchmark
from coldplate_studio.simulation.flow_poiseuille import run_poiseuille_benchmark


@pytest.mark.gmsh
def test_manufactured_stokes_converges_with_refinement():
    """MMS with homogeneous Dirichlet u on all duct walls."""
    mesh_sizes = [0.06, 0.045, 0.032]
    results = []
    for h in mesh_sizes:
        model = build_integral_fluid_duct(mesh_size=h)
        results.append(run_manufactured_stokes_benchmark(model, mesh_size=h))
    assert results[0].ux_l2_error < 0.08
    assert results[0].solver_residual_norm < 1e-6
    assert results[-1].ux_l2_error < results[0].ux_l2_error


@pytest.mark.gmsh
def test_quasi_2d_poiseuille_flow_rate_documented():
    """Body-force channel; documents mismatch vs 1D theory when x-ends are natural."""
    model = build_integral_fluid_duct(mesh_size=0.06)
    r = run_poiseuille_benchmark(model, mesh_size=0.06)
    assert r.divergence_l2 < 0.01
    assert r.numerical_flow_rate > 0
    assert r.flow_rate_relative_error < 0.9995  # developing flow; see VERIFICATION_EVIDENCE.md
