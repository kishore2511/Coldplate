"""Grad-div and PSPG scaling for P1–P1 Stokes (fin channel)."""

from __future__ import annotations

import numpy as np

from coldplate_studio.mesh.integral_model import IntegralMesh

# Reference length from mesh_size≈0.08 fin channel (calibrated once; not tuned per mesh to pass gates).
FIN_CHANNEL_H_REF_M = 0.034


def mean_fluid_mesh_length(mesh: IntegralMesh) -> float:
    from coldplate_studio.fr.tet_p1 import _tet_gradients

    vols = []
    for tet in mesh.fluid_tets:
        vol, _ = _tet_gradients(mesh.nodes[tet])
        vols.append(vol)
    h = float(np.mean(vols)) ** (1.0 / 3.0)
    return max(h, 1e-6)


def scaled_grad_div_beta(
    mesh: IntegralMesh,
    *,
    beta_ref: float = 100.0,
    h_ref: float = FIN_CHANNEL_H_REF_M,
) -> float:
    """
    Momentum grad-div coefficient β(h) = β_ref (h_ref/h)².

    Keeps the same effective penalty scale as β_ref at the reference length; coarser meshes
    receive smaller β to avoid over-constraining the discrete velocity field.
    """
    h = mean_fluid_mesh_length(mesh)
    return float(beta_ref * (h_ref / max(h, 1e-8)) ** 2)
