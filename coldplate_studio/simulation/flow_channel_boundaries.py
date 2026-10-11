"""Boundary sets for integral fin-channel flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.fr.flow_p1_operators import fluid_node_mask
from coldplate_studio.mesh.boundary_faces import classify_boundary_faces
from coldplate_studio.mesh.integral_model import IntegralModel


def _face_outward_normal(mesh, nodes: tuple[int, int, int]) -> np.ndarray:
    from coldplate_studio.fr.flow_boundary_flux import fluid_boundary_outward_area_vectors

    key = tuple(sorted(nodes))
    area = fluid_boundary_outward_area_vectors(mesh)
    n_vec = area[key]
    mag = float(np.linalg.norm(n_vec))
    if mag < 1e-30:
        return np.zeros(3)
    return n_vec / mag


def _select_channel_inlet_outlet_faces(
    model: IntegralModel,
    x_in: float,
    x_out: float,
    tol: float = 0.02,
    normal_alignment: float = 0.7,
):
    """Cross-section ports: x≈x_in/x_out and outward normal aligned with ±x."""
    mesh = model.mesh
    report = classify_boundary_faces(mesh)
    inlet: list[tuple[int, int, int]] = []
    outlet: list[tuple[int, int, int]] = []
    for f in report.faces:
        if f.face_type != "exterior_fluid":
            continue
        n_hat = _face_outward_normal(mesh, f.nodes)
        cx = float(f.centroid[0])
        if abs(cx - x_in) <= tol and n_hat[0] < -normal_alignment:
            inlet.append(f.nodes)
        if abs(cx - x_out) <= tol and n_hat[0] > normal_alignment:
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


def channel_port_fluid_nodes(
    boundaries: ChannelFlowBoundaries,
) -> tuple[np.ndarray, np.ndarray]:
    """Unique fluid nodes on tagged inlet/outlet port triangles."""
    inlet = np.array(sorted({int(i) for f in boundaries.inlet_faces for i in f}), dtype=int)
    outlet = np.array(sorted({int(i) for f in boundaries.outlet_faces for i in f}), dtype=int)
    return inlet, outlet


def solid_base_dirichlet_nodes(mesh, z_tol: float = 0.005) -> np.ndarray:
    """Solid nodes on the z=0 cold-face plane (integral fin base)."""
    fluid_mask = fluid_node_mask(mesh)
    z = mesh.nodes[:, 2]
    return np.array(
        [i for i in range(mesh.nodes.shape[0]) if not fluid_mask[i] and float(z[i]) <= z_tol],
        dtype=int,
    )


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
