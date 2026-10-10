"""CHT prerequisite: interface geometry and flux operator symmetry scaffold."""

import numpy as np
import pytest

from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.mesh.interface_geometry import analyze_interface_geometry


@pytest.mark.gmsh
def test_fsi_interface_area_positive_and_oriented():
    model = build_integral_channel_model(mesh_size=0.06)
    report = analyze_interface_geometry(model.mesh)
    assert report.total_area_m2 > 1e-6
    assert len(report.triangles) == model.mesh.interface_triangles.shape[0]
    for tri in report.triangles:
        assert tri.area_m2 > 0
        assert np.linalg.norm(tri.normal) > 0.9
