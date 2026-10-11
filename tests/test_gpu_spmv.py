import numpy as np
import pytest

from coldplate_studio.gpu.backend import _probe_cuda
from coldplate_studio.gpu.spmv import spmv_diagonal_cpu, spmv_diagonal_gpu


def test_spmv_cpu():
    d = np.array([1.0, 2.0, 3.0])
    x = np.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(spmv_diagonal_cpu(d, x), d * x)


@pytest.mark.gpu
def test_spmv_gpu_matches_cpu():
    if not _probe_cuda()[0]:
        pytest.skip("CUDA not available")
    d = np.linspace(1, 10, 128)
    x = np.linspace(-1, 1, 128)
    np.testing.assert_allclose(spmv_diagonal_gpu(d, x), spmv_diagonal_cpu(d, x), rtol=1e-10)
