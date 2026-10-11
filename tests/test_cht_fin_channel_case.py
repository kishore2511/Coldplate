import pytest

from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.simulation.cht_fin_channel_case import run_fin_channel_cht_case
from coldplate_studio.simulation.flow_channel import run_channel_flow
from coldplate_studio.simulation.flow_channel_boundaries import build_integral_channel_flow_boundaries


@pytest.mark.gmsh
def test_fin_channel_cht_nonzero_interface_flux_with_volumetric_source():
    model = build_integral_channel_model(mesh_size=0.08)
    flow = run_channel_flow(model, formulation="pspg")
    assert flow.mass_flux_relative_imbalance < 0.12
    boundaries = build_integral_channel_flow_boundaries(model)
    _, case = run_fin_channel_cht_case(model, flow, boundaries, thermal_steps=25, thermal_dt=0.05)
    assert abs(case.interface_flux_fluid_w) > 1e-4
    assert abs(case.interface_flux_solid_w) > 1e-4
    assert case.energy_relative_residual < 0.35
    assert case.solid_temperature_max_k > 300.01
