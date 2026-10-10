"""GPU-backed elementary operators (CPU reference always available)."""

from __future__ import annotations

import numpy as np

from coldplate_studio.gpu.backend import _probe_cuda
from coldplate_studio.gpu.spmv import spmv_diagonal_cpu, spmv_diagonal_gpu


def lumped_mass_apply(mass_lumped: np.ndarray, x: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """y_i = M_i x_i with diagonal lumped mass."""
    if use_gpu:
        cuda_ok, _ = _probe_cuda()
        if cuda_ok:
            return spmv_diagonal_gpu(mass_lumped, x)
    return spmv_diagonal_cpu(mass_lumped, x)


def gpu_benchmark_procedure() -> str:
    return (
        "Local GPU verification:\n"
        "  1. Install CUDA + numba with cuda support.\n"
        "  2. pytest -m gpu tests/test_gpu_spmv.py\n"
        "  3. python -c \"from coldplate_studio.gpu.operators import lumped_mass_apply; "
        "import numpy as np; d=np.ones(10**6); x=np.ones(10**6); "
        "lumped_mass_apply(d,x,use_gpu=True)\"\n"
        "Solver kernels (full FR/P1 operators) are not yet on GPU — only diagonal SpMV."
    )
