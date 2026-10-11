# Phase 1 — Implementation Audit (source-verified)

**Branch:** `cursor/coldplate-studio-foundation-341b`  
**Date:** 2026-10-10  
**pytest:** `17 passed in ~2.85s` (this environment)

## Discretization taxonomy

| Path | Module | Governing operator | FR status | Evidence | Limitations |
|------|--------|-------------------|-----------|----------|-------------|
| 1D scalar advection–diffusion | `fr/scalar_solver.py`, `fr/operators.py` | `u_t + a u_x = ν u_xx` | **Huynh-style FR** (LGL, Radau corrections, Riemann flux) | `test_fr_elements`, `test_fr_scalar` — polynomial steady residual, order > 2 advection | 1D only; not production 3D |
| 3D structured hex | `fr/deriv3d.py`, `fr/solver3d.py` | tensor FR `∇`, Laplacian, scalar transport | **FR tensor product** | `test_fr_3d` | **Verification only**; not integral production mesh |
| 3D integral production | `fr/tet_p1.py`, `fr/solver_integral.py` | `M Ṫ = -K T - C T`, VOF α | **P1 Galerkin / lumped mass** (lowest-order nodal DG) | `test_fr_3d`, channel e2e | **Not high-order FR**; constant `velocity` in `C` is **not** a flow solve |
| Phase change | `solver_integral` | phenomenological `ṁ(T,α)` | prototype source | superheated mass-loss test | **Not boiling**; R-011 unread |
| Incompressible (new) | `fr/flow_p1_penalty.py` | penalty Stokes / steady | P1 vector FEM | `test_flow_channel.py` | Steady; penalty β; not FR high-order |

## GPU

| Component | Executes on GPU? | Evidence |
|-----------|------------------|----------|
| `gpu/backend.py` | **No** solver kernels; `uses_gpu_kernels == False` | `test_gpu_backend` |
| `gpu/spmv.py` | Optional CUDA matvec when Numba CUDA available | `test_gpu_spmv` (skipped if no device) |

## Geometry / integral mesh

| Feature | Module | Verified by |
|---------|--------|-------------|
| OCC fragment channel | `mesh/integral_pipeline.py` | `test_integral_pipeline`, preflight |
| Fluid component count | `mesh/fluid_connectivity.py` | single component channel |
| FSI triangle area | `connectivity/interface_maps.py` | preflight + `test_interface_geometry` |
| FSI normals / orientation | `mesh/interface_geometry.py` | `test_interface_geometry` |
| User ports (no position guess) | `connectivity/ports.py` | `test_user_ports` |
| Exterior vs FSI boundary faces | `mesh/boundary_faces.py` | preflight |

## Capability registry honesty

Unit tests prove **code paths run**; they do not alone justify `benchmark_validated`. High-order FR on 3D production mesh is **not** implemented. Registry updated in `capabilities.py` + `VERIFICATION_EVIDENCE.md`.
