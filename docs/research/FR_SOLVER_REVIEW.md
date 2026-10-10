# Flux Reconstruction Solver Review

## Governing equations (roadmap)

| Stage | Model | Notes |
|-------|-------|-------|
| M6 (current) | Scalar advection–diffusion | 1D FR, CPU reference |
| M8 | Incompressible Navier–Stokes | Low-Mach liquid coolant |
| M9 | Energy + solid conduction | Conjugate heat transfer |
| M10–11 | Two-phase + phase change | Separate multiphase module |

Compressible Euler/NS (PyFR default heritage) is **not** the primary cold-plate target unless Ma > 0.3 is justified.

## FR formulation summary (R-001)

Huynh FR solves conservation laws in **differential form** on piecewise polynomials:

1. Store solution at solution points (collocation).
2. Form discontinuous element flux \(F(u)\).
3. Add left/right **correction functions** so the reconstructed flux is continuous at interfaces up to the Riemann state.
4. Differentiate the reconstructed flux to obtain the volume term.

Choosing **right Radau** correction functions recovers a stable nodal DG scheme on tensor-product elements (R-001, R-004).

### ColdPlate implementation (v0.1.0)

- Module: `coldplate_studio/fr/`
- Element nodes: Legendre–Gauss–Lobatto (LGL)
- Differentiation: barycentric Lagrange derivative matrix
- Riemann solver: upwind for linear advection (verification)
- Mapping: \(d/dx = (2/h)\, d/d\xi\) on \([-1,1]\) reference interval

**Not yet implemented:** 2D/3D volume/face loops, curved metrics, BR2/LDG diffusion on unstructured meshes, entropy/aliasing filters (PyFR features in R-002).

## PyFR v2.0.3 as reference (R-002, R-003)

| Topic | PyFR capability (surveyed) | ColdPlate plan |
|-------|---------------------------|----------------|
| Spatial scheme | High-order FR on hex/prism/tet/pyramid | Hex/quad first, then prisms for boundary layers |
| Compressible NS | Yes | Not primary |
| Incompressible NS | Yes — artificial compressibility + dual-time + multigrid | Independent implementation; compare on cylinder/channel cases |
| GPU backends | CUDA, HIP, OpenCL, Metal, OpenMP | CUDA first for production |
| Multiphase / VOF / boiling | **Not identified** in v2.0.3 release paper | In-house multiphase module |
| CHT / moving interface | External coupling not native | Native interface graph + coupling operators |

**Attribution rule:** Any future cross-benchmark against PyFR must cite PyFR and respect BSD-3-Clause; no copied source without compliance review.

## Pressure–velocity coupling candidates (ADR-003)

| Method | FR compatibility | Expected cost | Verification |
|--------|------------------|---------------|--------------|
| Artificial compressibility | Used in PyFR incompressible solver | Iterative pseudo-time per step | Manufactured solution + channel flow |
| Projection / pressure Poisson | Requires elliptic solves on FR operators | Preconditioner design critical | Divergence norms |
| All-Mach preconditioning | Research-grade | Higher implementation cost | Low-Mach limit study |

## Stability and aliasing (future)

PyFR employs modal filtering, anti-aliasing quadrature, artificial viscosity, and entropy filtering (R-002). ColdPlate will add analogous controls only after baseline schemes pass spec G FR tests.

## GPU matrix-free operators (preview)

Target structure (R-002):

- Element-local batched kernels for `∇·F` and face corrections
- Persistent device arrays for solution, metrics, and interpolation operators
- Host-device transfers only for I/O, diagnostics, and linear solver reductions

## Verification status (this milestone)

| Test | Result |
|------|--------|
| Lagrange derivative exactness on polynomials | pytest `test_fr_elements` |
| Constant state zero residual | pytest `test_fr_scalar` |
| Polynomial steady advection residual | pytest `test_fr_scalar` |
| Advection refinement order > 2 | pytest `test_fr_scalar` |

Capability state: **numerically verified** for 1D scalar FR only.
