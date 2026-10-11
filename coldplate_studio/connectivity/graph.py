"""Connectivity graph for fluids, solids, ports, and interfaces (spec B3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx

from coldplate_studio.geometry.extraction import FluidExtractionResult


class BoundaryClass(str, Enum):
    INLET = "inlet"
    OUTLET = "outlet"
    WALL = "wall"
    SYMMETRY = "symmetry"
    FLUID_SOLID_INTERFACE = "fluid_solid_interface"
    UNCLASSIFIED = "unclassified"


@dataclass
class GraphNode:
    node_id: str
    kind: str
    entity_tag: int | None
    component_id: int | None
    labels: dict[str, Any] = field(default_factory=dict)
    confidence: str = "high"


@dataclass
class ConnectivityGraph:
    graph: nx.Graph
    nodes: dict[str, GraphNode]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {k: v.__dict__ for k, v in self.nodes.items()},
            "edges": list(self.graph.edges(data=True)),
        }


def build_connectivity_from_extraction(result: FluidExtractionResult) -> ConnectivityGraph:
    g = nx.Graph()
    nodes: dict[str, GraphNode] = {}
    for region in result.regions:
        nid = f"vol:{region.tag}"
        nodes[nid] = GraphNode(
            node_id=nid,
            kind=region.kind,
            entity_tag=region.tag,
            component_id=region.component_id,
            labels={"volume_m3": region.volume_m3},
        )
        g.add_node(nid, kind=region.kind)
    iface_nid = "iface:fluid_solid"
    nodes[iface_nid] = GraphNode(
        node_id=iface_nid,
        kind="interface_patch",
        entity_tag=None,
        component_id=None,
        labels={"surface_tags": result.interface_surface_tags},
        confidence="high",
    )
    g.add_node(iface_nid, kind="interface_patch")
    for ft in result.fluid_tags:
        g.add_edge(f"vol:{ft}", iface_nid, relation="fluid_side")
    for st in result.solid_tags:
        g.add_edge(f"vol:{st}", iface_nid, relation="solid_side")
    return ConnectivityGraph(graph=g, nodes=nodes)
