"""Regression: fin-channel flow gates on three mesh sizes (honest pass/fail)."""

import pytest

from coldplate_studio.diagnostics.flow_gates import evaluate_flow_gates
from coldplate_studio.diagnostics.flow_refinement import run_fin_channel_flow_refinement
from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.simulation.flow_channel import run_channel_flow


@pytest.mark.gmsh
def test_fin_channel_flow_refinement_gates_documented():
    """
    Mass imbalance definition: |Φ_in + Φ_out| / mean(|Φ_in|, |Φ_out|) with outward-normal port fluxes.

    v0.4.1 (scaled β, fluid-only pressure): only mesh_size=0.08 passes the 20% gate; coarser/finer
    levels remain documented failures (not retuned β_ref).
    """
    rows = run_fin_channel_flow_refinement((0.12, 0.08, 0.06))
    by_size = {r.mesh_size: r for r in rows}
    assert by_size[0.08].flow_gates_ok
    assert not by_size[0.12].flow_gates_ok
    assert not by_size[0.06].flow_gates_ok
    assert by_size[0.08].mass_flux_relative_imbalance < 0.12
    assert by_size[0.12].mass_flux_relative_imbalance > 0.5
    assert by_size[0.06].mass_flux_relative_imbalance > 0.5
    for ms in (0.12, 0.08, 0.06):
        run = run_channel_flow(build_integral_channel_model(mesh_size=ms), formulation="pspg")
        assert evaluate_flow_gates(run).ok == by_size[ms].flow_gates_ok
