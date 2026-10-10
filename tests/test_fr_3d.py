import numpy as np

from coldplate_studio.fr.solver3d import PhaseChangeConfig, ScalarThermalVOF3D, Solver3DConfig


def test_3d_vof_stays_bounded_under_advection():
    cfg = Solver3DConfig(ne_x=3, ne_y=2, ne_z=2, degree=2, velocity_m_s=(0.2, 0.0, 0.0))
    solver = ScalarThermalVOF3D(cfg)
    solver.set_initial_vof(np.full_like(solver.state.vof, 0.6))
    solver.set_initial_temperature(np.full_like(solver.state.temperature, 300.0))
    for _ in range(5):
        report = solver.step_rk4(0.05)
    assert report.alpha_min >= 0.0
    assert report.alpha_max <= 1.0


def test_3d_phase_change_reduces_liquid_mass_when_superheated():
    cfg = Solver3DConfig(
        ne_x=3,
        ne_y=2,
        ne_z=2,
        degree=2,
        phase_change=PhaseChangeConfig(enabled=True, saturation_temperature=300.0, transfer_coefficient=0.1),
    )
    solver = ScalarThermalVOF3D(cfg)
    solver.set_initial_vof(np.full_like(solver.state.vof, 0.5))
    solver.set_initial_temperature(np.full_like(solver.state.temperature, 330.0))
    m0 = solver.conservation_report().liquid_mass_kg
    solver.step_rk4(0.1)
    m1 = solver.conservation_report().liquid_mass_kg
    assert m1 < m0


def test_3d_temperature_diffusion_reduces_peak():
    cfg = Solver3DConfig(ne_x=3, ne_y=2, ne_z=2, degree=2, thermal_diffusivity=1e-3, velocity_m_s=(0.0, 0.0, 0.0))
    solver = ScalarThermalVOF3D(cfg)
    x, y, z = solver.state.x, solver.state.y, solver.state.z
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    hot = 10.0 * np.exp(-((X - 0.5) ** 2 + (Y - 0.1) ** 2 + (Z - 0.05) ** 2) / 0.01)
    solver.set_initial_temperature(hot)
    solver.set_initial_vof(np.zeros_like(hot))
    peak0 = float(solver.state.temperature.max())
    for _ in range(4):
        solver.step_rk4(0.01)
    peak1 = float(solver.state.temperature.max())
    assert peak1 < peak0
