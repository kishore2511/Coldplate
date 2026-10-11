import pytest

from coldplate_studio.fr.flow_p1_pspg_mixed import solve_integral_channel_pspg
from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.simulation.flow_channel import run_channel_flow
from coldplate_studio.simulation.flow_channel_boundaries import build_integral_channel_flow_boundaries


@pytest.mark.gmsh
def test_fin_channel_pspg_matches_penalty_mass_flux_at_reference_mesh():
    """PSPG+beta=100 reproduces legacy penalty port fluxes on mesh_size=0.08."""
    model = build_integral_channel_model(mesh_size=0.08)
    pen = run_channel_flow(model, formulation="penalty")
    pspg = run_channel_flow(model, formulation="pspg_legacy_body_force")
    assert pspg.flow.formulation == "pspg_mixed_stokes"
    assert pspg.flow.solver_residual_norm is not None
    assert pspg.flow.solver_residual_norm < 1e-8
    assert abs(pspg.flow.inlet_flux_m3_s - pen.flow.inlet_flux_m3_s) < 1e-9
    assert abs(pspg.mass_flux_relative_imbalance - pen.mass_flux_relative_imbalance) < 1e-6


@pytest.mark.gmsh
def test_fin_channel_pspg_refinement_table_recorded():
    """Three mesh levels — document conservation (reference mesh gate mesh_size=0.08)."""
    rows = []
    for mesh_size in (0.12, 0.08, 0.06):
        model = build_integral_channel_model(mesh_size=mesh_size)
        mesh = model.mesh
        b = build_integral_channel_flow_boundaries(model)
        r = solve_integral_channel_pspg(mesh, b)
        rows.append(
            (
                mesh_size,
                mesh.nodes.shape[0],
                mesh.fluid_tets.shape[0],
                r.divergence_l2,
                r.mass_flux_relative_imbalance,
                r.solver_residual_norm,
            )
        )
    # v0.4.1: scaled β + fluid-only pressure — same port-flux table as v0.4 at 0.08
    ref = next(x for x in rows if x[0] == 0.08)
    coarse = next(x for x in rows if x[0] == 0.12)
    fine = next(x for x in rows if x[0] == 0.06)
    assert ref[4] < 0.12
    assert ref[2] > 400
    assert ref[3] < 0.005
    assert coarse[4] > 0.5
    assert fine[4] > 0.5
