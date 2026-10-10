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
    Returns (Q_fluid_to_solid, Q_solid_to_fluid) — should be opposite for consistent coupling.
    """
    from coldplate_studio.mesh.interface_geometry import analyze_interface_geometry

    geom = analyze_interface_geometry(mesh)
    q_f = q_s = 0.0
    nodes = mesh.nodes
    for tri_g, tri_m in zip(geom.triangles, mesh.interface_triangles):
        tri = tri_m
        grad_t = _interface_temperature_gradient(mesh, tri, temperature, "fluid", conductivity_fluid)
        grad_ts = _interface_temperature_gradient(mesh, tri, temperature, "solid", conductivity_solid)
        n_vec = tri_g.normal * tri_g.area_m2
        q_f += float(np.dot(-conductivity_fluid * grad_t, n_vec))
        q_s += float(np.dot(-conductivity_solid * grad_ts, n_vec))
    return q_f, q_s


def _interface_temperature_gradient(
    mesh: IntegralMesh,
    tri: np.ndarray,
    temperature: np.ndarray,
    side: str,
    k: float,
) -> np.ndarray:
    """P1 gradient of T on interface triangle using one adjacent tet."""
    key = tuple(sorted(int(x) for x in tri))
    tets = mesh.fluid_tets if side == "fluid" else mesh.solid_tets
    for tet in tets:
        local = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
        for a, b, c in local:
            if tuple(sorted((int(tet[a]), int(tet[b]), int(tet[c])))) == key:
                vol, grads = _tet_gradients(mesh.nodes[tet])
                g = np.zeros(3)
                for i in range(4):
                    g += grads[i] * temperature[tet[i]]
                return g
    return np.zeros(3)
