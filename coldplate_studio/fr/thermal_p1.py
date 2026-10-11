"""P1 thermal operators on fluid and solid tet regions (not high-order FR)."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from coldplate_studio.fr.tet_p1 import _tet_gradients
from coldplate_studio.mesh.integral_model import IntegralMesh


def region_masks(mesh: IntegralMesh) -> tuple[np.ndarray, np.ndarray]:
    n = mesh.nodes.shape[0]
    fluid = np.zeros(n, dtype=bool)
    solid = np.zeros(n, dtype=bool)
    for tet in mesh.fluid_tets:
        fluid[tet] = True
    for tet in mesh.solid_tets:
        solid[tet] = True
    return fluid, solid


def assemble_region_lumped_mass(
    mesh: IntegralMesh,
    tets: np.ndarray,
    rho_cp: float,
) -> np.ndarray:
    lumped = np.zeros(mesh.nodes.shape[0])
    for tet in tets:
        vol, _ = _tet_gradients(mesh.nodes[tet])
        for i in range(4):
            lumped[tet[i]] += rho_cp * vol / 4.0
    return lumped


def assemble_region_stiffness(
    mesh: IntegralMesh,
    tets: np.ndarray,
    conductivity: float,
) -> sparse.csr_matrix:
    n = mesh.nodes.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for tet in tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        for i in range(4):
            for j in range(4):
                val = conductivity * vol * float(np.dot(grads[i], grads[j]))
                rows.append(tet[i])
                cols.append(tet[j])
                data.append(val)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def assemble_advection(
    mesh: IntegralMesh,
    velocity: np.ndarray,
    rho_cp: float,
) -> sparse.csr_matrix:
    """Galerkin advection ρ c u·∇T on fluid tets only."""
    n = mesh.nodes.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for tet in mesh.fluid_tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        u_cell = velocity[tet].mean(axis=0)
        for i in range(4):
            for j in range(4):
                cij = rho_cp * vol * float(np.dot(u_cell, grads[j]))
                if abs(cij) > 0:
                    rows.append(tet[i])
                    cols.append(tet[j])
                    data.append(cij)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def interface_heat_flux(
    mesh: IntegralMesh,
    temperature: np.ndarray,
    conductivity_fluid: float,
    conductivity_solid: float,
) -> tuple[float, float]:
    """
    Integrated normal heat flux ∫ q·n dA on FSI triangles from fluid and solid sides.

    ``n`` points fluid → solid (``interface_geometry``). Returns
    (Q_fluid_to_solid, Q_solid_to_fluid); for a conservative field Q_f ≈ −Q_s.
    """
    from coldplate_studio.mesh.interface_geometry import analyze_interface_geometry

    geom = analyze_interface_geometry(mesh)
    q_f = q_s = 0.0
    for tri_g, tri_m in zip(geom.triangles, mesh.interface_triangles):
        tri = np.asarray(tri_m, dtype=int)
        n_vec = tri_g.normal * tri_g.area_m2
        grad_f = _interface_temperature_gradient(mesh, tri, temperature, "fluid", tri_g.normal)
        grad_s = _interface_temperature_gradient(mesh, tri, temperature, "solid", tri_g.normal)
        q_f += float(np.dot(-conductivity_fluid * grad_f, n_vec))
        q_s += float(np.dot(-conductivity_solid * grad_s, -n_vec))
    return q_f, q_s


def interface_flux_imbalance(q_fluid_w: float, q_solid_w: float) -> float:
    denom = max(abs(q_fluid_w), abs(q_solid_w), 1e-12)
    return float(abs(q_fluid_w + q_solid_w) / denom)


def _interface_temperature_gradient(
    mesh: IntegralMesh,
    tri: np.ndarray,
    temperature: np.ndarray,
    side: str,
    interface_normal: np.ndarray,
) -> np.ndarray:
    """P1 ∇T on interface triangle using the adjacent element on the requested side."""
    tri_set = {int(x) for x in tri}
    tets = mesh.fluid_tets if side == "fluid" else mesh.solid_tets
    face_center = mesh.nodes[tri].mean(axis=0)
    n_hat = interface_normal / max(float(np.linalg.norm(interface_normal)), 1e-30)
    best_score = -np.inf
    best_overlap = 0
    best_grad = np.zeros(3)
    for tet in tets:
        nodes_tet = {int(tet[i]) for i in range(4)}
        overlap = len(tri_set & nodes_tet)
        if overlap < 2:
            continue
        opp_nodes = [int(tet[i]) for i in range(4) if int(tet[i]) not in tri_set]
        if not opp_nodes:
            continue
        opp_pt = mesh.nodes[opp_nodes[0]]
        sign = -1.0 if side == "fluid" else 1.0
        orient = float(np.dot(opp_pt - face_center, sign * n_hat))
        if overlap < best_overlap or (overlap == best_overlap and orient <= best_score):
            continue
        vol, grads = _tet_gradients(mesh.nodes[tet])
        g = np.zeros(3)
        for i in range(4):
            g += grads[i] * temperature[tet[i]]
        best_overlap = overlap
        best_score = orient
        best_grad = g
    return best_grad
