"""Boundary sets for integral fin-channel flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.fr.flow_p1_operators import fluid_node_mask
from coldplate_studio.mesh.boundary_faces import classify_boundary_faces
from coldplate_studio.mesh.integral_model import IntegralModel


def _select_channel_inlet_outlet_faces(model: IntegralModel, x_in: float, x_out: float, tol: float = 0.02):
    report = classify_boundary_faces(model.mesh)
    inlet: list[tuple[int, int, int]] = []
    outlet: list[tuple[int, int, int]] = []
    for f in report.faces:
        if f.face_type != "exterior_fluid":
            continue
        if abs(f.centroid[0] - x_in) <= tol:
            inlet.append(f.nodes)
        if abs(f.centroid[0] - x_out) <= tol:
            outlet.append(f.nodes)
    return inlet, outlet


@dataclass
class ChannelFlowBoundaries:
    inlet_faces: list[tuple[int, int, int]]
    outlet_faces: list[tuple[int, int, int]]
    no_slip_nodes: np.ndarray
    solid_only_nodes: np.ndarray
    pinned_velocity_nodes: set[int]
    fluid_pressure_reference_node: int


def build_integral_channel_flow_boundaries(
    model: IntegralModel,
    length: float = 1.0,
) -> ChannelFlowBoundaries:
    inlet, outlet = _select_channel_inlet_outlet_faces(model, 0.0, length)
    port_nodes = {i for face in inlet + outlet for i in face}
    wall_nodes = np.array(
        [i for i in model.mesh.boundary_fluid_nodes if int(i) not in port_nodes],
        dtype=int,
    )
    fluid_mask = fluid_node_mask(model.mesh)
    solid_only = np.array([i for i in range(model.mesh.nodes.shape[0]) if not fluid_mask[i]], dtype=int)
    pinned = set(int(i) for i in wall_nodes) | set(int(i) for i in solid_only)
    ref = int(np.min(np.where(fluid_mask)[0]))
    return ChannelFlowBoundaries(
        inlet_faces=inlet,
        outlet_faces=outlet,
        no_slip_nodes=wall_nodes,
        solid_only_nodes=solid_only,
        pinned_velocity_nodes=pinned,
        fluid_pressure_reference_node=ref,
    )
