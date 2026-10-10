import pytest

from coldplate_studio.fr.flow_diagnostics import analyze_mass_continuity
from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.simulation.flow_channel import _select_channel_inlet_outlet_faces
from coldplate_studio.simulation.flow_channel import run_channel_flow


@pytest.mark.gmsh
def test_channel_mass_continuity_diagnosis_reproduces_baseline():
    model = build_integral_channel_model(mesh_size=0.08)
    result = run_channel_flow(model)
    inlet, outlet = _select_channel_inlet_outlet_faces(model, 0.0, 1.0)
    report = analyze_mass_continuity(model.mesh, result.flow.velocity, inlet, outlet)
    assert 0.10 < report.mass_flux_relative_imbalance < 0.30
    assert report.divergence_l2 < 0.01
    assert report.inlet_flux > 0
    assert report.outlet_flux < 0
    assert abs(report.wall_flux) < 2e-4  # P1 slip on no-slip walls at coarse h
    assert abs(report.inlet_plus_outlet) < abs(report.inlet_flux) * 0.35
