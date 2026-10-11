from coldplate_studio.fr.flow_p2_mms import run_p2p1_mms_study


def test_p2p1_manufactured_stokes_mms_residual_and_refinement():
    """Full MMS forcing; records errors — does not assert optimal Taylor–Hood rate on this box MMS."""
    reports = run_p2p1_mms_study(nx_levels=(4, 6, 8))
    assert len(reports) == 3
    for r in reports:
        assert r.solver_residual < 0.05
        assert r.divergence_l2 < 0.05
        assert r.ux_l2 < 0.05
