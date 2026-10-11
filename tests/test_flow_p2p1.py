import numpy as np

from coldplate_studio.fr.flow_p2p1_stokes import solve_p2p1_stokes
from coldplate_studio.mesh.structured_tet_box import build_structured_fluid_box
from coldplate_studio.simulation.flow_manufactured import manufactured_velocity_field


def test_p2p1_stokes_manufactured_residual_and_divergence():
    model = build_structured_fluid_box(nx=6, ny=4, nz=2, width=0.2, height=0.1)
    mesh = model.mesh
    walls = set(int(i) for i in mesh.boundary_fluid_nodes)
    pin = int(mesh.fluid_tets[0, 0])
    r = solve_p2p1_stokes(
        mesh,
        nu=1e-3,
        force_xyz=(1.0, 0.0, 0.0),
        pinned_velocity_nodes=walls,
        no_slip_nodes=walls,
        pressure_pin_node=pin,
    )
    assert r.solver_residual_norm < 1e-3
    assert r.divergence_l2 < 0.05
    assert r.pressure_residual_norm < 1e-2


def test_p2p1_weak_divergence_norm_small_on_refined_box():
    model = build_structured_fluid_box(nx=8, ny=5, nz=2, width=0.2, height=0.1)
    mesh = model.mesh
    walls = set(int(i) for i in mesh.boundary_fluid_nodes)
    pin = int(mesh.fluid_tets[0, 0])
    r = solve_p2p1_stokes(
        mesh,
        nu=1e-3,
        force_xyz=(1.0, 0.0, 0.0),
        pinned_velocity_nodes=walls,
        no_slip_nodes=walls,
        pressure_pin_node=pin,
    )
    assert r.divergence_l2 < 0.02


def test_p2p1_fin_channel_mass_balance_reference_mesh():
    from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
    from coldplate_studio.simulation.flow_channel import run_channel_flow

    model = build_integral_channel_model(mesh_size=0.08)
    run = run_channel_flow(model, formulation="p2p1")
    assert run.flow.solver_residual_norm is not None
    assert run.flow.solver_residual_norm < 1e-6
    assert run.mass_flux_relative_imbalance < 0.35
