"""Volume-quadrature evaluation of P2 velocity (for MMS error norms)."""

from __future__ import annotations

import numpy as np

from coldplate_studio.fr.flow_p2_dof_map import P2VelocityDofMap
from coldplate_studio.fr.tet_p2_basis import tet_quadrature_4pt
from coldplate_studio.fr.tet_p1 import _tet_gradients
from coldplate_studio.mesh.integral_model import IntegralMesh


def volume_l2_velocity_error(
    mesh: IntegralMesh,
    dof_map: P2VelocityDofMap,
    u_flat: np.ndarray,
    u_exact: np.ndarray,
    *,
    u_exact_at_point: callable | None = None,
) -> float:
    """‖u_h − u_ex‖_{L2(Ω)} via 4-point quadrature on fluid tets."""
    q_pts, q_w = tet_quadrature_4pt()
    err_sq = vol_sum = 0.0
    for tet in mesh.fluid_tets:
        vol, _ = _tet_gradients(mesh.nodes[tet])
        loc = dof_map.local_dof_indices(tet)
        for q, wq in zip(q_pts, q_w):
            dvol = vol * wq
            u_num = np.zeros(3)
            from coldplate_studio.fr.tet_p2_basis import p2_shape_values

            phi = p2_shape_values(q)
            for a in range(10):
                ia = int(loc[a])
                for comp in range(3):
                    u_num[comp] += u_flat[3 * ia + comp] * phi[a]
            x = sum(q[k] * mesh.nodes[tet[k]] for k in range(4))
            if u_exact_at_point is not None:
                u_ex = np.asarray(u_exact_at_point(np.asarray(x, dtype=float)), dtype=float).reshape(3)
            else:
                u_ex = sum(q[k] * u_exact[tet[k]] for k in range(4))
            err_sq += dvol * float(np.sum((u_num - u_ex) ** 2))
            vol_sum += dvol
    return float(np.sqrt(err_sq / max(vol_sum, 1e-30)))
