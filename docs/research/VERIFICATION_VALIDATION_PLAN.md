# Verification and Validation Plan

Aligned with spec G and implementation order I.

## Gate sequence

| Gate | Milestone | Automated tests | Capability state target |
|------|-----------|-----------------|-------------------------|
| G0 | Research + ADR | Docs present | `implemented_unverified` |
| G1 | Geometry + connectivity | `test_geometry_*`, `test_connectivity` | `numerically_verified` |
| G2 | FR scalar 1D | `test_fr_*` | `numerically_verified` |
| G3 | GPU backend smoke | `test_gpu_backend` (+ `@gpu` later) | `numerically_verified` |
| G4 | Incompressible flow | Planned `tests/flow/` | TBD |
| G5 | CHT | Planned `tests/thermal/` | TBD |
| G6 | Two-phase interface | Planned `tests/twophase/` | TBD |
| G7 | Phase change / Stefan | Planned | TBD |
| G8 | Boiling experiments | Manual + regression | `experimentally_validated` |

## Geometry and mesh (spec G)

| Case | Status | Tolerance |
|------|--------|-----------|
| Boolean channel extraction | ✅ PASS | fluid + solid + interface tags |
| Disconnected cavity detection | Planned | graph component count |
| Conformal interface consistency | Planned | area match fluid/solid faces |
| FR mesh mapping Jacobians | Partial (1D) | \(J>0\) |

## FR tests (spec G)

| Case | Status | Tolerance |
|------|--------|-----------|
| Polynomial derivative matrix | ✅ PASS | 1e-10 |
| Constant advection RHS | ✅ PASS | 1e-12 |
| Cubic steady advection | ✅ PASS | 1e-9 |
| Sin wave advection order | ✅ PASS | order > 2 | 
| Diffusion accuracy | Planned | manufactured solution |
| CPU/GPU operator match | Planned | 1e-12 rel (when GPU exists) |

## Flow and thermal (planned)

| Case | Method |
|------|--------|
| Poiseuille / channel | Analytical velocity |
| Heated channel energy balance | Integral enthalpy flux |
| CHT interface flux | Match `q_fluid + q_solid ≈ 0` |

## Two-phase (planned)

| Case | Acceptance |
|------|------------|
| Static bubble Laplace | \(\Delta p\) vs \(\sigma \kappa\) |
| Parasitic currents | Max spurious velocity < threshold |
| Phase fraction bounds | \([0,1]\) |
| Mass conservation | Drift < configured tolerance per 100 steps |
| Stefan problem | Interface position vs analytical |

## Boiling validation (planned)

| Source | Status |
|--------|--------|
| Gaertner pool boiling (R-013) | Data not yet ingested — case blocked |
| Literature CHF correlations | Not validation alone |

## Reporting per milestone (spec J)

Each milestone PR must include:

- Equations and numerical method used
- pytest summary
- Benchmark numbers (if applicable)
- Known limitations
- Next gate requirements

## Current pytest summary (v0.2.0)

```
13 passed in ~4.1s (CPU, Gmsh + 3D FR thermal/VOF + VTK/contours)
```

| Case | File |
|------|------|
| 3D VOF boundedness | `tests/test_fr_3d.py` |
| 3D phase-change mass loss (prototype) | `tests/test_fr_3d.py` |
| 3D thermal diffusion peak decay | `tests/test_fr_3d.py` |
| VTK + contour export | `tests/test_postprocess_channel.py` |
| Channel end-to-end | `tests/test_postprocess_channel.py` |
