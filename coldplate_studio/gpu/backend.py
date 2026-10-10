"""Compute backend selection and run diagnostics (spec D4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

BackendName = Literal["cpu_numpy", "cuda_unavailable"]


@dataclass
class RunDiagnostics:
    backend: BackendName
    gpu_only: bool
    gpu_model: str | None
    device_available: bool
    peak_device_memory_bytes: int | None
    cpu_fallback_events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "gpu_only": self.gpu_only,
            "gpu_model": self.gpu_model,
            "device_available": self.device_available,
            "peak_device_memory_bytes": self.peak_device_memory_bytes,
            "cpu_fallback_events": list(self.cpu_fallback_events),
        }


def _probe_cuda() -> tuple[bool, str | None]:
    try:
        from numba import cuda  # optional; not a hard dependency yet

        if cuda.is_available():
            dev = cuda.get_current_device()
            return True, dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
    except Exception:
        pass
    return False, None


class ComputeBackend:
    """Host-side array backend with explicit GPU-only failure."""

    def __init__(self, gpu_only: bool = False) -> None:
        self.gpu_only = gpu_only
        self._cuda_ok, self._gpu_name = _probe_cuda()
        self.cpu_fallback_events: list[str] = []
        if gpu_only and not self._cuda_ok:
            raise RuntimeError(
                "GPU-only mode is enabled but no CUDA device is available. "
                "Install a CUDA-capable GPU and numba/cupy, or disable gpu_only."
            )
        self.name: BackendName = "cpu_numpy" if not self._cuda_ok else "cpu_numpy"
        # CUDA kernels are not wired yet; production GPU path is planned (see docs/research/GPU_PERFORMANCE.md).

    @property
    def uses_gpu_kernels(self) -> bool:
        return False

    def array(self, data: Any, dtype=np.float64) -> np.ndarray:
        if self.gpu_only and not self._cuda_ok:
            raise RuntimeError("GPU-only mode requires CUDA.")
        return np.asarray(data, dtype=dtype)

    def diagnostics(self) -> RunDiagnostics:
        return RunDiagnostics(
            backend=self.name,
            gpu_only=self.gpu_only,
            gpu_model=self._gpu_name,
            device_available=self._cuda_ok,
            peak_device_memory_bytes=None,
            cpu_fallback_events=self.cpu_fallback_events,
        )
