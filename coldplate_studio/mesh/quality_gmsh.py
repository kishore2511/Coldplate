"""Mesh validity checks on integral meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldplate_studio.mesh.integral_model import IntegralMesh


@dataclass
class MeshValidityReport:
    n_fluid_tets: int
    min_tet_volume: float
    negative_volume_count: int
    ok: bool


def check_mesh_validity(mesh: IntegralMesh) -> MeshValidityReport:
    from coldplate_studio.fr.tet_p1 import _tet_gradients

    neg = 0
    vmin = float("inf")
    for tet in mesh.fluid_tets:
        vol, _ = _tet_gradients(mesh.nodes[tet])
        if vol <= 0:
            neg += 1
        vmin = min(vmin, vol)
    if vmin == float("inf"):
        vmin = 0.0
    return MeshValidityReport(
        n_fluid_tets=int(mesh.fluid_tets.shape[0]),
        min_tet_volume=float(vmin),
        negative_volume_count=neg,
        ok=neg == 0 and mesh.fluid_tets.shape[0] > 0,
    )
