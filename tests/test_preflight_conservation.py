import pytest

from coldplate_studio.diagnostics.conservation import audit_conservation_report
from coldplate_studio.diagnostics.preflight import run_integral_preflight
from coldplate_studio.fr.solver3d import ConservationReport
from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.mesh.fluid_connectivity import analyze_fluid_connectivity


@pytest.mark.gmsh
def test_channel_has_single_connected_fluid_component():
    model = build_integral_channel_model(mesh_size=0.06)
    fluid = analyze_fluid_connectivity(model.mesh)
    assert fluid.n_components == 1
    assert fluid.largest_component_fraction > 0.99


@pytest.mark.gmsh
def test_integral_preflight_passes_for_channel():
    model = build_integral_channel_model(mesh_size=0.06)
    report = run_integral_preflight(model)
    assert report.ok


def test_conservation_audit_rejects_invalid_alpha():
    bad = ConservationReport(
        liquid_mass_kg=1.0,
        total_thermal_energy_j=1.0,
        phase_change_power_w=0.0,
        alpha_min=-0.01,
        alpha_max=0.5,
    )
    assert not audit_conservation_report(bad).ok
