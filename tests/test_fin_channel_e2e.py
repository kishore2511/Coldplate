import pytest

from coldplate_studio.simulation.fin_channel_e2e import run_fin_channel_e2e


@pytest.mark.gmsh
def test_fin_channel_e2e_flow_gates_pass_reference_mesh(tmp_path):
    result = run_fin_channel_e2e(tmp_path, mesh_size=0.08, thermal_steps=3)
    assert result.flow_gates["solver_residual_ok"]
    assert result.flow_gates["divergence_ok"]
    assert result.flow_gates["mass_imbalance_ok"]
    assert result.flow_gates["ok"]
    assert (tmp_path / "fin_channel_e2e_summary.json").is_file()
    assert result.flow.get("solver_method")
    assert result.cht is not None
    assert "galerkin_flux_fluid_w" in result.cht
