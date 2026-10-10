# Flow discretization status

## Production integral path (current)

| Component | Discretization | Notes |
|-----------|----------------|-------|
| Steady Stokes (verification) | P1–P1 mixed + PSPG (τ = h²/(4ν)) | `flow_p1_pspg_mixed.py`; pressure pin |
| Steady Stokes (fin channel) | P1 + grad-div penalty | Legacy path; ~19.7% port imbalance documented |
| Transient | Chorin–Temam projection on P1 | Lumped div → Poisson; γP stabilization |
| High-order FR momentum | **Not implemented** | 1D/hex FR remain verification-only |

## Taylor–Hood P2–P1 evaluation

**Pros:** inf-sup stable, standard for incompressible tet/hex.  
**Cons:** requires P2 mesh/order, face DOFs, separate pressure space — not yet wired to Gmsh integral export.

**Decision:** Implement **PSPG P1–P1 mixed Stokes** for verification benchmarks (no arbitrary β as sole incompressibility mechanism). Keep penalty Stokes for fin-channel regression only. Taylor–Hood P2–P1 remains the planned upgrade when P2 Gmsh export exists (ADR TBD).

**Why not Taylor–Hood now:** current production mesh is P1-only from Gmsh; P2 velocity spaces and inf-sup DOF bookkeeping are not wired. PSPG reuses existing P1 operators (`assemble_weak_divergence`, viscous K, pressure Laplacian).

## Developed-channel benchmark

- Geometry: `mesh/structured_tet_box.py`
- BC: y no-slip, x periodic (penalty ties inlet/outlet node pairs), natural z faces
- Analytical: `u_x = (G/2ν) y(H−y)`, `Q = G W H³/(12ν)`
- Recorded error (nx=8→16, penalty path): ux L2 **~1e-9** (profile); cross-section Q via `flow_flux_diagnostics` **~2.8%** vs GWH³/(12ν)
- PSPG periodic channel (nx=10): ux L2 vs parabolic **< 5e-3**; ‖div u‖_L2 **< 0.05** (`test_flow_pspg.py`)

## Fin channel (realistic)

`test_channel_stokes_mass_conservation` — **not** Poiseuille validation; ~19.7% port imbalance from penalty + 3D geometry.
