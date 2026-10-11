# Flow discretization status

## Production integral path (current)

| Component | Discretization | Notes |
|-----------|----------------|-------|
| Steady Stokes (verification) | P1–P1 mixed + PSPG (τ = h²/(4ν)) | `flow_p1_pspg_mixed.py`; pressure pin |
| Steady Stokes (fin channel, **production v0.5**) | **Taylor–Hood P2–P1** mixed | `flow_p2p1_stokes.py`; quadratic velocity on fluid edges+vertices; LSQR saddle solve |
| Steady Stokes (fin channel, legacy) | PSPG + body force + grad-div β(h)=β_ref(h_ref/h)² | `formulation="pspg"`; see v0.4.1 notes |
| Steady Stokes (fin channel, legacy) | P1 + grad-div penalty | `formulation="penalty"`; regression |
| Transient | Chorin–Temam projection on P1 | Lumped div → Poisson; γP stabilization |
| High-order FR momentum | **Not implemented** | 1D/hex FR remain verification-only |

## Taylor–Hood P2–P1 evaluation

**Pros:** inf-sup stable, standard for incompressible tet/hex.  
**Cons:** requires P2 mesh/order, face DOFs, separate pressure space — not yet wired to Gmsh integral export.

**v0.5:** **Taylor–Hood P2–P1** implemented on P1 Gmsh meshes via subparametric elevation (edge midpoint DOFs). PSPG P1–P1 remains for regression (`formulation="pspg"`).

**FR high-order:** still verification-only (1D/hex); not production incompressible flow.

## Developed-channel benchmark

- Geometry: `mesh/structured_tet_box.py`
- BC: y no-slip, x periodic (penalty ties inlet/outlet node pairs), natural z faces
- Analytical: `u_x = (G/2ν) y(H−y)`, `Q = G W H³/(12ν)`
- Recorded error (nx=8→16, penalty path): ux L2 **~1e-9** (profile); cross-section Q via `flow_flux_diagnostics` **~2.8%** vs GWH³/(12ν)
- PSPG periodic channel (nx=10): ux L2 vs parabolic **< 5e-3**; ‖div u‖_L2 **< 0.05** (`test_flow_pspg.py`)

## Fin channel (realistic)

Integral fin channel — v0.5 P2–P1 passes mass and divergence gates on mesh sizes 0.12, 0.08, 0.06 (see `FIN_CHANNEL_V05.md`).
