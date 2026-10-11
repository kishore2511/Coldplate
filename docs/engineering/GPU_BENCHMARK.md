# GPU benchmark procedure

This cloud agent environment has **no CUDA GPU**. Solver assembly and linear solves are **CPU-only**; do not claim GPU-accelerated flow/thermal until those kernels exist.

## What is verified today

| Component | Device | Test |
|-----------|--------|------|
| Diagonal SpMV | CPU always; CUDA if available | `test_spmv_cpu`, `test_spmv_gpu_matches_cpu` (skipped without CUDA) |
| `require_gpu()` | — | `test_gpu_only_fails_without_device` raises when CUDA missing |

Modules: `coldplate_studio/gpu/spmv.py`, `coldplate_studio/gpu/operators.py`.

## Local reproduction (Windows, NVIDIA RTX 3050 Ti)

1. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) matching your driver (3050 Ti is CUDA-capable).
2. In the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
$env:CUDA_VISIBLE_DEVICES="0"
pytest tests/test_gpu_spmv.py -v
python -c "from coldplate_studio.gpu.operators import gpu_benchmark_procedure; print(gpu_benchmark_procedure())"
```

3. Record in your notes: GPU name, driver, CUDA version, dtype (float64), matrix size `n`, host/device memory if reported, CPU vs GPU time, speedup.

Expected on 3050 Ti for diagonal SpMV smoke test: **parity** with CPU within rtol; speedup is problem-size dependent (small `n` may show no gain).

## Not measured (out of scope for this milestone)

- P1 Stokes / PSPG assembly or solve on GPU
- Thermal / CHT operator assembly on GPU
- Full FR high-order kernels

Until those are implemented and covered by CPU/GPU parity tests, `gpu_solver_kernels` stays **implemented_unverified**.
