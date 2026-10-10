import pytest

from coldplate_studio.geometry.extraction import extract_channel_fluid_domain


@pytest.mark.gmsh
def test_channel_extraction_labels_fluid_solid_and_interface():
    result = extract_channel_fluid_domain(1.0, 0.2, 0.1, 0.02, 0.05, mesh_size=0.04)
    assert result.ok
    assert result.fluid_tags
    assert result.solid_tags
    assert result.interface_surface_tags
    kinds = {r.kind for r in result.regions}
    assert "fluid" in kinds and "solid" in kinds
