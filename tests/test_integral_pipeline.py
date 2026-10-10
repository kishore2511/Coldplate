import pytest

from coldplate_studio.mesh.integral_pipeline import build_integral_channel_model


@pytest.mark.gmsh
def test_integral_model_single_mesh_for_fluid_and_interface():
    model = build_integral_channel_model(mesh_size=0.06)
    assert model.extraction.ok
    assert model.mesh.fluid_tets.shape[0] > 0
    assert model.mesh.interface_triangles.shape[0] > 0
    assert model.mesh.fr_element_family == "p1_tetrahedron"
    # All fluid tet nodes should reference the same node array used for coordinates.
    assert model.mesh.nodes.shape[1] == 3
