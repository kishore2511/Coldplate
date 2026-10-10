# GPU Performance Strategy

## Reference: PyFR (R-002, R-003)

PyFR v2.0.3 highlights relevant to ColdPlate:

- Multiple GPU backends (CUDA, HIP, OpenCL, Metal)
- CUDA-aware / HIP-aware MPI for strong scaling
- Task graphs to reduce API overhead (e.g., 2D incompressible cylinder case speedup cited in R-002)
- Matrix-free FR residual evaluation as primary cost

ColdPlate will **not** claim PyFR-level scaling until measured on target hardware.

## ColdPlate backend (ADR-009)

Module: `coldplate_studio/gpu/backend.py`

| Mode | Behavior |
|------|----------|
| Default | CPU NumPy reference |
| `gpu_only=True` | Raises if CUDA unavailable |
| Production (future) | CUDA kernels for FR volume/face ops, device-resident fields |

### Run diagnostics (spec D4)

`RunDiagnostics` records:

- Backend name
- GPU model (if probed)
- CPU fallback events
- Peak device memory (future)

## Optimization strategies (spec F)

Priority order for implementation:

1. Matrix-free FR operators (avoid assembled global Jacobians)
2. Batched element kernels
3. Reuse metric/interpolation data across time steps
4. IMEX splitting for stiff phase-change / conduction
5. Preconditioned pressure solvers (once incompressible scheme selected)
6. Local refinement before global p-refinement
7. Mixed precision only after stability/conservation study

## Environment constraints (this repository CI)

- Cloud agent VM: **no NVIDIA GPU** detected (`nvidia-smi` unavailable)
- GPU tests marked `@pytest.mark.gpu` — skip until hardware present
- Performance baselines must be recorded on user workstation or GPU CI runner

## Profiling plan

| Stage | Tooling |
|-------|---------|
| CPU reference | `cProfile`, pytest timings |
| CUDA | Nsight Systems, CUDA events in backend |
| Memory | `cudaMemGetInfo` / CuPy pool stats |

## Measured results (milestone 1)

| Benchmark | Hardware | Result |
|-----------|----------|--------|
| FR 1D pytest suite | CPU x86 cloud | ~0.3 s total (8 tests) |
| Gmsh channel mesh | CPU | ~0.02 s meshing (logged by Gmsh) |

No GPU speedup claims at this milestone.
