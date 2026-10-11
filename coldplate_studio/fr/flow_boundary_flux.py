"""Oriented boundary fluxes and discrete divergence (P1 fluid tets)."""

from __future__ import annotations

import numpy as np

from coldplate_studio.fr.tet_p1 import _tet_gradients
from coldplate_studio.mesh.integral_model import IntegralMesh


def fluid_boundary_outward_area_vectors(mesh: IntegralMesh) -> dict[tuple[int, int, int], np.ndarray]:
    """Area vector (2× area × unit outward normal) for each exterior fluid boundary triangle."""
    local = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    face_count: dict[tuple[int, int, int], int] = {}
    area_vecs: dict[tuple[int, int, int], np.ndarray] = {}
    for tet in mesh.fluid_tets:
        pts = mesh.nodes[tet]
        for a, b, c in local:
            key = tuple(sorted((int(tet[a]), int(tet[b]), int(tet[c]))))
            face_count[key] = face_count.get(key, 0) + 1
            opp = next(i for i in range(4) if i not in (a, b, c))
            v0, v1, v2 = pts[a], pts[b], pts[c]
            n_vec = np.cross(v1 - v0, v2 - v0)
            if float(np.dot(n_vec, pts[opp] - v0)) > 0.0:
                n_vec = -n_vec
            area_vecs[key] = n_vec
    return {k: v for k, v in area_vecs.items() if face_count.get(k, 0) == 1}


def oriented_face_flux(
    mesh: IntegralMesh,
    velocity: np.ndarray,
    face_keys: list[tuple[int, int, int]],
    area_vecs: dict[tuple[int, int, int], np.ndarray],
) -> float:
    """∫_Γ u·n dA with n outward from the fluid domain (m³/s)."""
    flux = 0.0
    for key in face_keys:
        n_vec = area_vecs[key]
        i0, i1, i2 = key
        u_face = (velocity[i0] + velocity[i1] + velocity[i2]) / 3.0
        flux += float(np.dot(u_face, n_vec) / 2.0)
    return flux


def discrete_divergence_l2(mesh: IntegralMesh, velocity: np.ndarray) -> float:
    div_sq = vol_sum = 0.0
    for tet in mesh.fluid_tets:
        vol, grads = _tet_gradients(mesh.nodes[tet])
        div = sum(float(np.dot(grads[:, k], velocity[tet, k])) for k in range(3))
        div_sq += vol * div**2
        vol_sum += vol
    return float(np.sqrt(div_sq / max(vol_sum, 1e-30)))


def add_pressure_traction_rhs(
    rhs: np.ndarray,
    n_nodes: int,
    face_keys: list[tuple[int, int, int]],
    pressure_pa: float,
    area_vecs: dict[tuple[int, int, int], np.ndarray],
) -> None:
    """
    Weak natural BC: add ∫_Γ p (n·v) dA to the momentum RHS (Stokes sign convention).

    ``area_vecs`` stores 2×area×n_outward per face key (consistent with ``oriented_face_flux``).
    """
    dim_u = 3 * n_nodes
    for key in face_keys:
        n_vec = area_vecs[key]
        area = 0.5 * float(np.linalg.norm(n_vec))
        for node in key:
            for comp in range(3):
                rhs[comp * n_nodes + int(node)] += pressure_pa * (area / 3.0) * (n_vec[comp] / max(2.0 * area, 1e-30))


def normalized_mass_imbalance(inlet_flux_m3_s: float, outlet_flux_m3_s: float) -> float:
    """|Φ_in + Φ_out| / mean(|Φ_in|, |Φ_out|) — opposite signs expected for through-flow."""
    mean_flux = 0.5 * (abs(inlet_flux_m3_s) + abs(outlet_flux_m3_s))
    return float(abs(inlet_flux_m3_s + outlet_flux_m3_s) / max(mean_flux, 1e-12))
