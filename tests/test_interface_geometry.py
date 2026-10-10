import pytest

from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model
from coldplate_studio.mesh.interface_geometry import analyze_interface_geometry
from coldplate_studio.connectivity.interface_maps import build_interface_connectivity


@pytest.mark.gmsh
def test_interface_area_positive_and_maps_consistent():
    model = build_integral_channel_model(mesh_size=0.06)
    geom = analyze_interface_geometry(model.mesh)
    maps = build_interface_connectivity(model.mesh)
    assert geom.total_area_m2 > 0
    assert abs(geom.total_area_m2 - maps.total_area_m2) < 1e-6
    assert geom.orientation_ok
    assert len(geom.triangles) == model.mesh.interface_triangles.shape[0]
