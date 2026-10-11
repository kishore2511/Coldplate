import numpy as np

from coldplate_studio.fr.flow_p1_operators import fluid_node_mask
from coldplate_studio.fr.flow_p2_dof_map import build_p2_velocity_dof_map
from coldplate_studio.fr.flow_p2_velocity_field import P2VelocityField
from coldplate_studio.fr.thermal_p1 import assemble_advection
from coldplate_studio.fr.thermal_p2_advection import assemble_p2_velocity_advection, build_p2_velocity_field_from_vertex_array
from coldplate_studio.mesh.structured_tet_box import build_structured_fluid_box


def test_p2_velocity_field_reproduces_constant_on_tet_centroid():
    model = build_structured_fluid_box(nx=3, ny=2, nz=2)
    mesh = model.mesh
    n = mesh.nodes.shape[0]
    u_const = np.zeros((n, 3))
    u_const[:, 0] = 2.5
    field = build_p2_velocity_field_from_vertex_array(mesh, u_const)
    tet = mesh.fluid_tets[0]
    fluid = fluid_node_mask(mesh)
    dof_map = build_p2_velocity_dof_map(mesh, fluid)
    loc = dof_map.local_dof_indices(tet)
    L = np.array([0.25, 0.25, 0.25, 0.25])
    u = field.evaluate_on_tet(loc, L)
    assert np.linalg.norm(u - np.array([2.5, 0.0, 0.0])) < 1e-10


def test_p2_advection_active_for_uniform_flow():
    """P2 Galerkin advection is not identical to legacy cell-mean P1; check it transports on a ramp."""
    model = build_structured_fluid_box(nx=4, ny=3, nz=2)
    mesh = model.mesh
    n = mesh.nodes.shape[0]
    u_const = np.zeros((n, 3))
    u_const[:, 0] = 1.0
    field = build_p2_velocity_field_from_vertex_array(mesh, u_const)
    c_p2 = assemble_p2_velocity_advection(mesh, field, 1.0)
    t = mesh.nodes[:, 0].copy()
    ct = c_p2 @ t
    fluid = fluid_node_mask(mesh)
    assert float(np.max(np.abs(ct[fluid]))) > 1e-6
