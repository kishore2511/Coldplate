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
