"""Mass continuity diagnostics for P1 incompressible flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.fr.flow_p1_operators import assemble_weak_divergence, lumped_divergence_at_nodes
from coldplate_studio.fr.flow_p1_penalty import (
    _fluid_boundary_outward_area_vectors,
    _oriented_face_flux,
)
from coldplate_studio.mesh.boundary_faces import classify_boundary_faces
from coldplate_studio.mesh.integral_model import IntegralMesh


@dataclass
class MassContinuityReport:
    divergence_l2: float
    volume_mean_div: float
    weak_div_l2: float
    total_exterior_flux: float
    wall_flux: float
    fsi_flux: float
    inlet_flux: float
    outlet_flux: float
    inlet_plus_outlet: float
    mass_flux_relative_imbalance: float
    notes: list[str]


def _divergence_l2(mesh: IntegralMesh, velocity: np.ndarray) -> float:
    div_sq = 0.0
    vol_sum = 0.0
    for tet in mesh.fluid_tets:
        from coldplate_studio.fr.tet_p1 import _tet_gradients

        vol, grads = _tet_gradients(mesh.nodes[tet])
        div = 0.0
        for k in range(3):
            div += float(np.dot(grads[:, k], velocity[tet, k]))
        div_sq += vol * div**2
        vol_sum += vol
    return float(np.sqrt(div_sq / max(vol_sum, 1e-30)))


def analyze_mass_continuity(
    mesh: IntegralMesh,
    velocity: np.ndarray,
    inlet_faces: list[tuple[int, int, int]],
    outlet_faces: list[tuple[int, int, int]],
) -> MassContinuityReport:
    area_vecs = _fluid_boundary_outward_area_vectors(mesh)
    boundary = classify_boundary_faces(mesh)
    inlet_set = set(inlet_faces)
    outlet_set = set(outlet_faces)
    wall_flux = fsi_flux = total_ext = 0.0
    for face in boundary.faces:
        key = face.nodes
        if key not in area_vecs:
            continue
        phi = _oriented_face_flux(mesh, velocity, [key], area_vecs)
        total_ext += phi
        if key in inlet_set:
            continue
        if key in outlet_set:
            continue
        if face.face_type == "fluid_solid_interface":
            fsi_flux += phi
        else:
            wall_flux += phi
    phi_in = _oriented_face_flux(mesh, velocity, inlet_faces, area_vecs)
    phi_out = _oriented_face_flux(mesh, velocity, outlet_faces, area_vecs)
    mean_flux = 0.5 * (abs(phi_in) + abs(phi_out))
    rel = abs(phi_in + phi_out) / max(mean_flux, 1e-12)
    n = mesh.nodes.shape[0]
    u_flat = np.zeros(3 * n)
    for k in range(3):
        u_flat[k * n : (k + 1) * n] = velocity[:, k]
    d_op = assemble_weak_divergence(mesh)
    weak_div = d_op @ u_flat
    m_lump = np.zeros(n)
    for tet in mesh.fluid_tets:
        from coldplate_studio.fr.tet_p1 import _tet_gradients

        vol, _ = _tet_gradients(mesh.nodes[tet])
        for i in range(4):
            m_lump[tet[i]] += vol / 4.0
    weak_div_norm = float(np.sqrt(np.sum((weak_div**2) * m_lump) / max(m_lump.sum(), 1e-30)))
    vol_mean_div = float(np.mean(lumped_divergence_at_nodes(mesh, velocity)[m_lump > 0]))
    notes = []
    if abs(total_ext) > 1e-10:
        notes.append(
            f"Global exterior flux Σ u·n dA = {total_ext:.6e} (should ≈ volume integral div u)."
        )
    if abs(wall_flux) > 1e-8:
        notes.append(f"Nonzero wall/FSI leakage flux wall={wall_flux:.6e} fsi={fsi_flux:.6e} (no-slip should give ~0).")
    return MassContinuityReport(
        divergence_l2=_divergence_l2(mesh, velocity),
        volume_mean_div=vol_mean_div,
        weak_div_l2=weak_div_norm,
        total_exterior_flux=total_ext,
        wall_flux=wall_flux,
        fsi_flux=fsi_flux,
        inlet_flux=phi_in,
        outlet_flux=phi_out,
        inlet_plus_outlet=phi_in + phi_out,
        mass_flux_relative_imbalance=float(rel),
        notes=notes,
    )
