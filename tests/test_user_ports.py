import pytest

from coldplate_studio.connectivity.ports import PortKind, apply_user_port_definitions, UserPortDefinition
from coldplate_studio.mesh.boundary_faces import classify_boundary_faces
from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.simulation.flow_channel import _select_channel_inlet_outlet_faces


@pytest.mark.gmsh
def test_ports_require_user_face_lists_not_position_guess():
    model = build_integral_channel_model(mesh_size=0.06)
    boundary = classify_boundary_faces(model.mesh)
    exterior = [f.nodes for f in boundary.faces if f.face_type == "exterior_fluid"]
    # Empty user assignment -> not ok for flow
    report = apply_user_port_definitions(model.mesh, [])
    assert not report.ok_for_flow_solve
    assert report.unclassified_exterior_faces


@pytest.mark.gmsh
def test_user_confirmed_inlet_outlet_faces():
    model = build_integral_channel_model(mesh_size=0.06)
    inlet, outlet = _select_channel_inlet_outlet_faces(model, 0.0, 1.0)
    assert inlet and outlet
    report = apply_user_port_definitions(
        model.mesh,
        [
            UserPortDefinition("inlet", PortKind.INLET, inlet),
            UserPortDefinition("outlet", PortKind.OUTLET, outlet),
        ],
    )
    assert report.ok_for_flow_solve
