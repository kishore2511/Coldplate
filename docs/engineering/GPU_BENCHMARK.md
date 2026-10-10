# GPU benchmark procedure

This cloud environment has **no CUDA GPU**. `gpu_solver_kernels` remains **unverified**.

## What runs on GPU today

- `coldplate_studio/gpu/spmv.py` — diagonal SpMV via Numba CUDA when available.
- `coldplate_studio/gpu/operators.py` — `lumped_mass_apply(..., use_gpu=True)` wrapper.

## Local verification

```bash
pytest -m gpu tests/test_gpu_spmv.py
python3 -c "from coldplate_studio.gpu.operators import gpu_benchmark_procedure; print(gpu_benchmark_procedure())"
```

Full P1/FR operator assembly and solves remain CPU-only until ported.
