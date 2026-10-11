import pytest

from coldplate_studio.diagnostics.cht_refinement import run_fin_channel_cht_refinement


@pytest.mark.gmsh
def test_fin_channel_cht_refinement_flow_gates_and_energy():
    """
    Mesh study for the production path. Local gradient flux may be noisy (D-004);
    this test does not loosen flow or energy gates.
    """
    rows = run_fin_channel_cht_refinement((0.12, 0.08, 0.06), thermal_steps=6, thermal_dt=0.05)
    assert [r.mesh_size for r in rows] == [0.12, 0.08, 0.06]
    for r in rows:
        assert r.flow_gates_ok
        assert r.formulation.startswith("p2p1")
        assert r.solver_method in ("direct", "lsqr", "minres")
        assert abs(r.energy_relative_residual) <= 0.35
        assert r.solid_temperature_max_k > 300.0
        # Both interface statements are recorded. Opposite signs are not required:
        # the gradient-based fluid flux can be orders of magnitude below the solid
        # side on these meshes (D-004). Energy residual remains the global gate.
