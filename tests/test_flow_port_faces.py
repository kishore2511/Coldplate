import numpy as np
import pytest

from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.simulation.flow_channel_boundaries import (
    _face_outward_normal,
    build_integral_channel_flow_boundaries,
)


@pytest.mark.gmsh
def test_inlet_outlet_faces_have_streamwise_outward_normals():
    model = build_integral_channel_model(mesh_size=0.08)
    b = build_integral_channel_flow_boundaries(model)
    assert len(b.inlet_faces) >= 4
    assert len(b.outlet_faces) >= 4
    mesh = model.mesh
    for face in b.inlet_faces:
        n = _face_outward_normal(mesh, face)
        assert n[0] < -0.5
    for face in b.outlet_faces:
        n = _face_outward_normal(mesh, face)
        assert n[0] > 0.5
