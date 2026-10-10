import pytest

from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.simulation.flow_channel import run_channel_flow


@pytest.mark.gmsh
def test_channel_stokes_mass_conservation():
    model = build_integral_channel_model(mesh_size=0.08)
    result = run_channel_flow(model)
    assert result.ports_ok
    assert result.flow.divergence_l2 < 5.0  # coarse penalty Stokes
    assert result.mass_flux_relative_imbalance < 0.35
    assert abs(result.flow.mean_velocity_x) > 1e-8
