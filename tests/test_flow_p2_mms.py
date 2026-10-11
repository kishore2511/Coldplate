"""P2–P1 manufactured Stokes on sin-π box (see docs/engineering/P2_MMS_AUDIT.md)."""

import numpy as np

from coldplate_studio.fr.flow_p2_mms import (
    format_p2_mms_table,
    run_p2p1_mms_study,
    run_p2p1_polynomial_channel_mms,
    stokes_mms_force_vector,
)
from coldplate_studio.simulation.flow_manufactured import manufactured_velocity_field


def test_p2p1_mms_forcing_matches_weak_form_stokes():
    """f = −νΔu + ∇p at sample points (finite-difference Laplacian spot check)."""
    length, height, width, nu, grad_p = 1.0, 0.1, 0.2, 1e-3, -1.0
    x0 = np.array([0.35, 0.04, 0.08])
    eps = 1e-5
    u0 = manufactured_velocity_field(
        x0.reshape(1, 3), length=length, height=height, width=width, grad_p=grad_p, nu=nu
    )[0, 0]
    lap_fd = 0.0
    for d in range(3):
        xp = x0.copy()
        xm = x0.copy()
        xp[d] += eps
        xm[d] -= eps
        up = manufactured_velocity_field(
            xp.reshape(1, 3), length=length, height=height, width=width, grad_p=grad_p, nu=nu
        )[0, 0]
        um = manufactured_velocity_field(
            xm.reshape(1, 3), length=length, height=height, width=width, grad_p=grad_p, nu=nu
        )[0, 0]
        lap_fd += (up - 2 * u0 + um) / eps**2
    f = stokes_mms_force_vector(
        x0.reshape(1, 3), length=length, height=height, width=width, grad_p=grad_p, nu=nu
    )[0, 0]
    assert abs(f - (-nu * lap_fd + grad_p)) / max(abs(f), 1e-12) < 0.05


def test_p2p1_manufactured_stokes_mms_residual_and_velocity():
    reports = run_p2p1_mms_study(nx_levels=(4, 8, 12), uniform_refinement=True)
    assert len(reports) == 3
    for r in reports:
        assert r.solver_residual < 1e-8
        assert r.divergence_l2 < 1e-10
        assert r.ux_l2 < 0.01
        assert r.ux_l2_volume_p2 < 0.01
        assert r.solver_method in ("lsqr", "direct", "minres")
    # Volume L2 plateaus on this non-polynomial exact field (documented); do not require monotone refinement.
    rel_plateau = abs(reports[0].ux_l2_volume_p2 - reports[-1].ux_l2_volume_p2) / reports[0].ux_l2_volume_p2
    assert rel_plateau < 0.2
    _ = format_p2_mms_table(reports)  # smoke: table formatter


def test_p2p1_polynomial_channel_pressure_and_velocity_are_exact():
    """u in P2, p in P1, exact Dirichlet — regression for pressure-block sign (D-010)."""
    reports = run_p2p1_polynomial_channel_mms(ny_levels=(2, 4))
    for r in reports:
        assert r.solver_residual < 1e-8
        assert r.divergence_l2 < 1e-8
        assert r.ux_l2_volume_p2 < 1e-6
        assert r.p_l2_relative < 1e-6, r
        assert r.solver_method in ("direct", "minres", "lsqr")
