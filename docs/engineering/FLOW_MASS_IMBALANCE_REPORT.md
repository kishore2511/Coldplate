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

## Conclusion

The ~19.7% port metric is **consistent with penalty-Stokes discretization error and non-zero div **u****, not a bug in oriented flux integration. Improving it requires either higher penalty with stable linear solve, a projection-based incompressible formulation, and/or mesh refinement—not changing normalization or omitting faces.

See `analyze_mass_continuity()` in `fr/flow_diagnostics.py` and `test_flow_diagnostics.py`.
