"""Saddle-point Stokes solve with documented pressure reference."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, minres, spsolve


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
    nrm = max(float(np.linalg.norm(rhs)), 1e-30)

    def _rel(sol: np.ndarray) -> float:
        return float(np.linalg.norm(a_csr @ sol - rhs) / nrm)

    try:
        sol = spsolve(a_csr, rhs)
        if sol is not None and np.all(np.isfinite(sol)):
            res = _rel(sol)
            if res < 1e-8:
                return sol, "direct", res
    except Exception:
        pass
    sol_ls, *_ = lsqr(a_csr, rhs, atol=1e-12, btol=1e-12, iter_lim=20000)
    res_ls = _rel(sol_ls)
    if res_ls < 1e-8:
        return sol_ls, "lsqr", res_ls
    # LSQR residual not tight: MINRES matches the symmetric indefinite saddle.
    sol_mr, info = minres(a_csr, rhs, rtol=1e-12, maxiter=min(8000, max(a_csr.shape[0], 1)))
    if np.all(np.isfinite(sol_mr)):
        res_mr = _rel(sol_mr)
        if res_mr <= res_ls:
            return sol_mr, "minres", res_mr
    return sol_ls, "lsqr", res_ls
