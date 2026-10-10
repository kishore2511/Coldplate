import pytest

from coldplate_studio.connectivity.diagnostics import run_connectivity_diagnostics
from coldplate_studio.connectivity.graph import build_connectivity_from_extraction
from coldplate_studio.geometry.extraction import extract_channel_fluid_domain


@pytest.mark.gmsh
def test_connectivity_graph_links_fluid_solid_interface():
    result = extract_channel_fluid_domain(1.0, 0.2, 0.1, 0.02, 0.05, mesh_size=0.04)
    graph = build_connectivity_from_extraction(result)
    diag = run_connectivity_diagnostics(graph)
    assert graph.graph.number_of_nodes() >= 3
    assert any(d.get("kind") == "fluid" for _, d in graph.graph.nodes(data=True))
    assert any(d.get("kind") == "interface_patch" for _, d in graph.graph.nodes(data=True))
    assert diag.ok
