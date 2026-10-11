import numpy as np

from coldplate_studio.fr.scalar_hex_advdiff import (
    HexAdvDiffConfig,
    ScalarHexAdvDiff3D,
    estimate_l2_convergence_order,
    manufactured_scalar_field,
    manufactured_scalar_source,
    run_hex_advdiff_mms_study,
)


def test_hex_advdiff_manufactured_source_balances_rhs_at_steady():
    cfg = HexAdvDiffConfig(ne_x=3, ne_y=2, ne_z=2, degree=2, velocity_m_s=(0.1, 0.0, 0.0))
    solver = ScalarHexAdvDiff3D(cfg)
    lx, ly, lz = 1.0, 0.2, 0.1
    exact = manufactured_scalar_field(solver.state.x, solver.state.y, solver.state.z, lx=lx, ly=ly, lz=lz)
    source = manufactured_scalar_source(solver, exact)
    solver.set_scalar(exact)
    rhs = solver.rhs(source=source)
    assert float(np.max(np.abs(rhs))) < 1e-12


def test_hex_advdiff_manufactured_field_invariant_under_rk4_step():
    cfg = HexAdvDiffConfig(ne_x=4, ne_y=3, ne_z=2, degree=2, velocity_m_s=(0.08, 0.0, 0.0), diffusivity=0.05)
    solver = ScalarHexAdvDiff3D(cfg)
    exact = manufactured_scalar_field(solver.state.x, solver.state.y, solver.state.z, lx=1.0, ly=0.2, lz=0.1)
    source = manufactured_scalar_source(solver, exact)
    solver.set_scalar(exact)
    solver.step_rk4(0.01, source=source)
    assert solver.volume_mean_l2_error(exact) < 1e-10


def test_hex_fr_laplacian_refinement_reduces_discretization_error():
    reports = run_hex_advdiff_mms_study(ne_levels=(3, 6))
    assert reports[0].l2_error > reports[1].l2_error
    order = estimate_l2_convergence_order(reports)
    assert order is not None
    assert order > 0.1
    ratio = reports[0].l2_error / reports[1].l2_error
    assert ratio > 1.05
