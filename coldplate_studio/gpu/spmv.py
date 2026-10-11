"""GPU sparse-like operations (verification kernels)."""

from __future__ import annotations

import numpy as np

from coldplate_studio.gpu.backend import _probe_cuda


def spmv_diagonal_cpu(diag: np.ndarray, x: np.ndarray) -> np.ndarray:
    return diag * x


def spmv_diagonal_gpu(diag: np.ndarray, x: np.ndarray) -> np.ndarray:
    cuda_ok, _ = _probe_cuda()
    if not cuda_ok:
        raise RuntimeError("CUDA not available for GPU spmv")
    try:
        from numba import cuda

        @cuda.jit
        def _kernel(d, xin, out):
            i = cuda.grid(1)
            if i < d.size:
                out[i] = d[i] * xin[i]

        d_dev = cuda.to_device(np.asarray(diag, dtype=np.float64))
        x_dev = cuda.to_device(np.asarray(x, dtype=np.float64))
        out_dev = cuda.device_array_like(x_dev)
        n = diag.size
        threads = 256
        blocks = (n + threads - 1) // threads
        _kernel[blocks, threads](d_dev, x_dev, out_dev)
        return out_dev.copy_to_host()
    except Exception as exc:
        raise RuntimeError(f"GPU kernel failed: {exc}") from exc
