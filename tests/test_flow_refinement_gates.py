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

    v0.5 Taylor–Hood P2–P1 production path: mass imbalance ≤20% on all three fin meshes in recorded study.
    """
    rows = run_fin_channel_flow_refinement((0.12, 0.08, 0.06), formulation="p2p1")
    by_size = {r.mesh_size: r for r in rows}
    for ms in (0.12, 0.08, 0.06):
        assert by_size[ms].mass_flux_relative_imbalance < 0.20
        assert by_size[ms].flow_gates_ok
    for ms in (0.12, 0.08, 0.06):
        run = run_channel_flow(build_integral_channel_model(mesh_size=ms), formulation="p2p1")
        assert evaluate_flow_gates(run).ok == by_size[ms].flow_gates_ok
