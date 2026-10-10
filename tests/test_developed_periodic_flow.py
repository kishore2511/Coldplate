import pytest

from coldplate_studio.fr.flow_periodic_stokes import solve_developed_periodic_channel
from coldplate_studio.mesh.structured_tet_box import build_structured_fluid_box


def test_developed_parabolic_profile_structured_duct():
    """x-periodic penalty + y no-slip; analytical u_x ∝ y(H-y)."""
    results = []
    for nx in (8, 12, 16):
        model = build_structured_fluid_box(nx=nx, ny=6, nz=2)
        results.append(solve_developed_periodic_channel(model.mesh, width=0.2, height=0.1))
    assert results[0].ux_l2_error < 1e-6
    assert results[-1].ux_l2_error <= results[0].ux_l2_error
    assert results[0].divergence_l2 < 1e-2
