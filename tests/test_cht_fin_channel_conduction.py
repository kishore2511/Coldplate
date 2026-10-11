"""Integral fin-channel CHT with u=0 (conduction-dominated interface flux)."""

import pytest

from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.simulation.cht_fin_channel_conduction import run_fin_channel_conduction_case
from coldplate_studio.simulation.flow_channel_boundaries import build_integral_channel_flow_boundaries


@pytest.mark.gmsh
def test_fin_channel_conduction_interface_flux_and_energy_budget():
    model = build_integral_channel_model(mesh_size=0.08)
    boundaries = build_integral_channel_flow_boundaries(model)
    out = run_fin_channel_conduction_case(model, boundaries, thermal_steps=80, thermal_dt=0.05)
    assert abs(out["interface_flux_solid_w"]) > 1e-4
    assert abs(out["interface_flux_fluid_w"]) > 1e-5
    assert out["energy_relative_residual"] < 0.85
    assert out["max_solid_temperature_k"] > 300.005
