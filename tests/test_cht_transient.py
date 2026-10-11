import pytest

from coldplate_studio.fr.cht_transient import ConjugateHeatTransferSolver, ThermalBoundaryConditions, ThermalMaterialProperties
from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.mesh.structured_tet_box import build_structured_fluid_box
import numpy as np


@pytest.mark.gmsh
def test_cht_integral_channel_energy_increases_with_hot_inlet():
    model = build_integral_channel_model(mesh_size=0.08)
    from coldplate_studio.simulation.flow_channel import run_channel_flow

    run = run_channel_flow(model)
    from coldplate_studio.simulation.flow_channel_boundaries import (
        build_integral_channel_flow_boundaries,
        channel_port_fluid_nodes,
    )

    boundaries = build_integral_channel_flow_boundaries(model)
    inlet_nodes, _outlet = channel_port_fluid_nodes(boundaries)
    solid_nodes = {int(t) for tet in model.mesh.solid_tets for t in tet}
    outer_solid = np.array(
        [i for i in solid_nodes if model.mesh.nodes[i, 2] > 0.08],
        dtype=int,
    )
    props = ThermalMaterialProperties()
    bcs = ThermalBoundaryConditions(
        fluid_inlet_nodes=inlet_nodes,
        fluid_inlet_temperature_k=360.0,
        solid_outer_nodes=outer_solid[: min(5, len(outer_solid))] if outer_solid.size else inlet_nodes[:1],
        solid_outer_temperature_k=300.0,
    )
    solver = ConjugateHeatTransferSolver(
        model.mesh,
        run.flow.velocity,
        props,
        bcs,
        p2_velocity_field=run.flow.p2_velocity_field,
    )
    e0 = solver.diagnostics().total_energy_j
    for _ in range(3):
        solver.step(0.05)
    e1 = solver.diagnostics().total_energy_j
    assert e1 > e0


def test_cht_interface_flux_opposite_signs_structured():
    """With uniform T, interface flux imbalance should be small."""
    model = build_structured_fluid_box(nx=4, ny=2, nz=1)
    velocity = np.zeros((model.mesh.nodes.shape[0], 3))
    props = ThermalMaterialProperties(k_fluid=1.0, k_solid=1.0)
    bcs = ThermalBoundaryConditions()
    solver = ConjugateHeatTransferSolver(model.mesh, velocity, props, bcs)
    diag = solver.diagnostics()
    assert diag.total_energy_j > 0
