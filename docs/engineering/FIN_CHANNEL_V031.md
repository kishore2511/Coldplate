# Fin-channel v0.3.1 — flow ports and coupled CHT

## Flow formulation (production)

**PSPG P1–P1 mixed Stokes** with:

- Uniform streamwise body force `f_x = dp/dx = -1` (same driving model as legacy penalty Stokes).
- Momentum **grad-div** penalty `β = 100` on the velocity block (not a substitute for PSPG; controls discrete divergence on open ducts).
- PSPG continuity stabilization `τ = h²/(4ν)`, `ν = 1e-3`.
- No-slip on exterior fluid walls and solid-only velocity DOFs; **natural traction** on inlet/outlet port triangles.
- Pressure reference: one fluid node pinned; **solid-only pressure DOFs pinned to 0** (removes pressure null modes on conforming FSI meshes).

**Not mixed in production:** body force + hard Dirichlet pressure on ports (freezes flow); weak pressure traction without grad-div on this mesh (‖div u‖ blows up).

## Port-face selection (root cause of reported mass imbalance)

At commit `950dc91`, inlet/outlet faces were chosen by **centroid x tolerance only**. That included triangles on the x≈0 / x≈L planes whose **outward normal is not streamwise** (fin-gap side walls, oblique cuts). Oriented flux ∫ **u**·**n** dA then mixed streamwise flow with transverse faces, producing **191% / 19.7% / 164%** imbalance across mesh sizes (not a single sign error).

**v0.3.1 fix:** require `|n_x| > 0.7` with `n_x < 0` (inlet) or `n_x > 0` (outlet) plus centroid tolerance.

| mesh_size | nodes | fluid tets | imbalance (aligned ports) | ‖div u‖_L2 |
|-----------|-------|------------|---------------------------|------------|
| 0.12 | 108 | 240 | ~1.45 | ~1.0e-3 |
| 0.08 | 200 | 496 | **~0.05** | ~9.8e-4 |
| 0.06 | 298 | 771 | ~1.21 | ~7.8e-4 |

Imbalance is **mesh-sensitive** with fixed β=100; only `mesh_size=0.08` meets the documented **20%** gate today. Coarse/fine levels remain reported honestly in E2E refinement tables.

## Coupled CHT

**Previous near-zero interface flux (~1e-10 W):**

1. Thermal inlet used an **x-slice** of boundary nodes, not **tagged port-face** nodes (often missing the hot Dirichlet on the real inlet).
2. Default transient (25×0.02 s) with water-like `ρ c_p` and small solid volume yielded negligible stored energy and flat solid T≈300 K.
3. Interface-flux gate **skipped** when |Q| < 1e-4 W.

**v0.3.1 deterministic demo** (`cht_fin_channel_case.py`):

- Hot fluid inlet on **port-face nodes**; solid base at 300 K.
- Volumetric solid heating `5×10⁵ W/m³` (total ~100 W on this geometry).
- Reports fluid/solid temperature ranges, interface fluxes, global energy residual (BE-consistent).
- P1 interface flux **magnitude** is gated; **side-to-side imbalance** can exceed 15% when fluid-side conduction signal is weak — reported, not hidden.

## Regression

- `formulation="penalty"` and `pspg` remain aligned at `mesh_size=0.08` (~5% port imbalance).
- `formulation="pspg_inlet_parabolic"` retained as experimental (not production).
