import pytest

from coldplate_studio.simulation.fin_channel_e2e import run_fin_channel_e2e


@pytest.mark.gmsh
@pytest.mark.parametrize("mesh_size", (0.12, 0.08, 0.06))
def test_fin_channel_e2e_flow_gates_pass_fin_meshes(tmp_path, mesh_size):
    result = run_fin_channel_e2e(
        tmp_path / str(mesh_size),
        mesh_size=mesh_size,
        thermal_steps=3,
        run_refinement_study=False,
    )
    assert result.flow_gates["solver_residual_ok"]
    assert result.flow_gates["divergence_ok"]
    assert result.flow_gates["mass_imbalance_ok"]
    assert result.flow_gates["ok"]
    assert result.flow.get("solver_method")
    summary = tmp_path / str(mesh_size) / "fin_channel_e2e_summary.json"
    assert summary.is_file()
    if not result.cht_skipped:
        assert result.cht is not None
        assert "galerkin_flux_fluid_w" in result.cht
        assert "thermal_resistance_k_per_w" in result.cht
        assert "solid_temperature_max_k" in result.cht
        assert "energy_relative_residual" in result.cht


def test_failed_flow_gate_cannot_pass_e2e(monkeypatch, tmp_path):
    """A required flow gate failure must not be reported as an overall success."""
    from coldplate_studio.simulation import fin_channel_e2e as e2e_mod

    real = e2e_mod.run_channel_flow

    def _bad_divergence(*args, **kwargs):
        run = real(*args, **kwargs)
        run.flow.divergence_l2 = 1.0
        return run

    monkeypatch.setattr(e2e_mod, "run_channel_flow", _bad_divergence)
    result = run_fin_channel_e2e(
        tmp_path, mesh_size=0.12, thermal_steps=1, run_refinement_study=False
    )
    assert result.flow_gates["ok"] is False
    assert result.ok is False
    assert result.cht_skipped is True
