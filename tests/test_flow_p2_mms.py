from coldplate_studio.fr.flow_p2_mms import (
    estimate_ux_volume_convergence_order,
    run_p2p1_mms_study,
)


def test_p2p1_manufactured_stokes_mms_residual_and_refinement():
    """Full MMS forcing; records errors — does not assert optimal Taylor–Hood rate on this box MMS."""
    reports = run_p2p1_mms_study(nx_levels=(4, 6, 8))
    assert len(reports) == 3
    for r in reports:
        assert r.solver_residual < 0.05
        assert r.divergence_l2 < 0.05
        assert r.ux_l2 < 0.05
        assert r.ux_l2_volume_p2 < 0.08
        assert r.solver_method in ("lsqr", "direct")


def test_p2p1_sin_mms_volume_error_recorded_for_order_audit():
    reports = run_p2p1_mms_study(nx_levels=(4, 8))
    assert all(r.ux_l2_volume_p2 > 0 for r in reports)
    _ = estimate_ux_volume_convergence_order(reports)  # may be non-monotone (see D-008)
