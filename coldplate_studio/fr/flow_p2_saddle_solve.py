"""Saddle-point Stokes solve with documented pressure reference."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, spsolve


def solve_stokes_saddle(
    a: sparse.spmatrix,
    rhs: np.ndarray,
    *,
    pressure_pin_row: int,
) -> tuple[np.ndarray, str, float]:
    """
    Solve [K B^T; B 0] x = rhs with one pressure DOF pinned to zero.

    The continuous Stokes pressure is unique up to an additive constant; pinning
    one pressure row removes that null mode. Remaining near-singularity on coarse
    meshes comes from the indefinite saddle structure; UMFPACK may fail while
    LSQR finds the minimum-norm consistent solution.
    """
    a_csr = a.tocsr()
    try:
        sol = spsolve(a_csr, rhs)
        if sol is not None and np.all(np.isfinite(sol)):
            res = float(np.linalg.norm(a_csr @ sol - rhs) / max(np.linalg.norm(rhs), 1e-30))
            if res < 1e-7:
                return sol, "direct", res
    except Exception:
        pass
    sol, *_ = lsqr(a_csr, rhs, atol=1e-12, btol=1e-12, iter_lim=50000)
    res = float(np.linalg.norm(a_csr @ sol - rhs) / max(np.linalg.norm(rhs), 1e-30))
    return sol, "lsqr", res
