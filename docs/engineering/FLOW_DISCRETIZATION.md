# Flow discretization status

## Production integral path (current)

| Component | Discretization | Notes |
|-----------|----------------|-------|
| Steady Stokes | P1 vector Galerkin + grad-div penalty | Not inf-sup stable alone; β tuned |
| Transient | Chorin–Temam projection on P1 | Lumped div → Poisson; γP stabilization |
| High-order FR momentum | **Not implemented** | 1D/hex FR remain verification-only |

## Taylor–Hood P2–P1 evaluation

**Pros:** inf-sup stable, standard for incompressible tet/hex.  
**Cons:** requires P2 mesh/order, face DOFs, separate pressure space — not yet wired to Gmsh integral export.

**Decision:** Continue P1 + penalty/projection for integral tet meshes; use **structured tet benchmarks** with **x-periodic penalty coupling** for developed-flow verification. Taylor–Hood is the planned upgrade when Gmsh high-order groups are integrated (ADR TBD).

## Developed-channel benchmark

- Geometry: `mesh/structured_tet_box.py`
- BC: y no-slip, x periodic (penalty ties inlet/outlet node pairs), natural z faces
- Analytical: `u_x = (G/2ν) y(H−y)`, `Q = G W H³/(12ν)`
- Recorded error (nx=8→16): ux L2 **~1e-9** (profile); bulk Q estimate ~17% off mean-flux quadrature (documented)

## Fin channel (realistic)

`test_channel_stokes_mass_conservation` — **not** Poiseuille validation; ~19.7% port imbalance from penalty + 3D geometry.
