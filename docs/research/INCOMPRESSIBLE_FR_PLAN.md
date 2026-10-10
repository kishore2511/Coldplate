# Incompressible Flow on Integral Mesh — Plan

## Current implementation (this milestone)

**Module:** `fr/flow_p1_penalty.py`  
**Equations (steady precursor):**

\[
-\nu \nabla^2 \mathbf{u} + \beta \nabla(\nabla\cdot\mathbf{u}) = \mathbf{f}, \quad \mathbf{f} = (-\partial p/\partial x, 0, 0)
\]

- **Discretization:** P1 tetrahedra, vector Galerkin Laplacian + divergence penalty (not high-order FR).
- **BC:** no-slip on `boundary_fluid_nodes`; inlet/outlet faces **user-confirmed** via `connectivity/ports.py`.
- **Test:** `test_channel_stokes_mass_conservation` — flux imbalance metric on coarse mesh.

## Target FR-compatible path (not yet coded)

1. **Artificial compressibility** (Chorin; see R-002 §3.6 for literature context only) on FR operators:
   - \(\partial \mathbf{u}/\partial t + \nabla p/\rho = -\mathbf{u}\cdot\nabla\mathbf{u} + \nu\nabla^2\mathbf{u}\)
   - \(\partial p/\partial t + \rho c^2 \nabla\cdot\mathbf{u} = 0\)
2. Replace penalty steady solve with time-accurate FR face fluxes for momentum.
3. **Pressure:** dual-time or projection on FR divergence matrix; measure \(\|\nabla\cdot\mathbf{u}\|\).

## Staged roadmap (spec phase 3)

| Stage | Deliverable | Gate |
|-------|-------------|------|
| 3a | Steady penalty Stokes + user ports | mass flux imbalance test |
| 3b | Transient artificial compressibility P1 | Poiseuille profile L2 |
| 3c | FR hex/tet momentum + energy split | mesh/time convergence |
| 3d | Solid conduction + FSI heat flux | flux continuity |
