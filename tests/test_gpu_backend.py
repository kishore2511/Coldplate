import pytest

from coldplate_studio.gpu.backend import ComputeBackend


def test_cpu_backend_available():
    backend = ComputeBackend(gpu_only=False)
    diag = backend.diagnostics()
    assert diag.backend == "cpu_numpy"
    assert diag.device_available is False or diag.gpu_model is not None


def test_gpu_only_fails_without_device():
    backend = ComputeBackend(gpu_only=False)
    if not backend.diagnostics().device_available:
        with pytest.raises(RuntimeError):
            ComputeBackend(gpu_only=True)
