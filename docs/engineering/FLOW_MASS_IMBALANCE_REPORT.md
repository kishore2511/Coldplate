# Steady channel mass-imbalance diagnosis (integral fin channel)

**Case:** `build_integral_channel_model(mesh_size=0.08)`, penalty Stokes (`β=100`, `ν=1e-3`, `dp/dx=-1`), user-confirmed inlet/outlet faces at `x≈0` / `x≈L`.

## Reproduced metrics (pytest baseline)

| Metric | Value |
|--------|-------|
| ‖div **u**‖_L2 | ≈ 1.84×10⁻³ |
| Oriented Φ_in | ≈ +4.15×10⁻⁴ m³/s |
| Oriented Φ_out | ≈ −5.06×10⁻⁴ m³/s |
| \|Φ_in+Φ_out\| / mean(\|Φ\|) | ≈ **19.7%** |

Outward-normal convention: area vector from fluid tet face, flipped so it points out of the fluid element; Φ = ∫ **u**·**n** dA.

## Root cause (numerical evidence)

1. **Finite divergence penalty, not discrete incompressibility.** The steady solve enforces ∇·**u** ≈ 0 through a penalty term, not a divergence-free subspace. Measured ‖div **u**‖_L2 ≈ 1.8×10⁻³ while global exterior flux Σ_∂Ω **u**·**n** dA tracks the same order (Gauss: ∫_Ω div **u** dV ≈ boundary flux). Port imbalance is therefore dominated by **non-zero discrete divergence**, not a sign error on normals.

2. **Equal-order P1 without inf-sup stabilization on the steady problem.** Penalty factor `β=100` is modest for this mesh; raising `β` reduces ‖div **u**‖ but worsens conditioning. This is expected for P1–P1-type treatments without Taylor–Hood or grad-div scaling tied to mesh size.

3. **Body-force driving with natural inlet/outlet.** Inlet/outlet nodes are not constrained; only walls (and solid-only DOFs) are no-slip. Flux balance is not enforced as a constraint—only weakly via penalty.

4. **Geometric port grouping.** Inlet/outlet faces are selected by centroid tolerance on `x`; fin obstruction splits the inlet plane into many triangles. All are included, but **developing flow** and **3D fin wake** invalidate a 1D flux balance interpretation at coarse resolution.

5. **Wall leakage flux is negligible.** Diagnostics show |Φ_wall|, |Φ_FSI| ≪ |Φ_ports| on no-slip walls, confirming outward normals and sign convention (Φ_in > 0, Φ_out < 0 for streamwise flow).

## v0.3.1 update (port tags)

Re-audit at `950dc91` showed the **dominant error** on reported 191% / 19.7% / 164% metrics was **including non-streamwise exterior fluid faces** in inlet/outlet flux integrals (x-tolerance only). With **normal-aligned port selection**, PSPG+β=100 at mesh_size=0.08 matches penalty at **~5%** imbalance; coarser `0.12` and finer `0.06` remain **>100%** with the same β.

Residual imbalance at the reference mesh is still **discretization** (grad-div + non-zero ‖div u‖), not oriented-flux sign errors. See `FIN_CHANNEL_V031.md`.

## Conclusion (penalty baseline, tol-only ports)

The ~19.7% port metric on **tol-only** face sets is **consistent with penalty-Stokes discretization error, non-zero div **u**, and port-face grouping** — not a bug in oriented flux integration alone.

See `analyze_mass_continuity()` in `fr/flow_diagnostics.py` and `test_flow_diagnostics.py`.
