# Fin-channel v0.4.1 — flow conservation audit and conservative CHT reporting

## Scope

Diagnose v0.4 fin-channel **mass imbalance** on coarse/fine meshes and **~99% local interface-flux mismatch** in the coupled fin-heating case. Improve the formulation only where justified; do not retune `β_ref` to force three mesh levels through the 20% gate.

## Flow discretization (verified in code)

**Weak form** (`flow_p1_pspg_mixed.py`):

- Momentum: ν ∫ ∇u·∇v − ∫ p ∇·v = ∫ f·v (viscous stiffness + weak divergence transpose).
- Continuity: ∫ q ∇·u + τ ∫ ∇q·∇p = 0 with PSPG τ = h²/(4ν).
- Optional momentum grad-div: β ∫ (∇·u)(∇·v) with β(h) = β_ref (h_ref/h)² (`flow_stabilization.py`).

**Production fin channel:** `body_force_legacy`, open inlet/outlet (no pressure traction), no-slip walls, scaled β, **pressure unknowns on fluid nodes only** (`fluid_pressure_only=True`), single fluid pressure pin. Legacy regression: `formulation="pspg_legacy_body_force"` uses full-node pressure + solid-only p=0 pins and fixed β=100.

**Port flux diagnostic:** `oriented_face_flux` on normal-aligned exterior triangles (`flow_channel_boundaries.py`, |n_x|>0.7). **Mass imbalance:** |Φ_in + Φ_out| / mean(|Φ_in|, |Φ_out|) (`normalized_mass_imbalance`).

**Acceptance gates** (`flow_gates.py`): solver residual ≤ 1e-8, ‖div u‖_L2 ≤ 0.005, mass imbalance ≤ 0.20.

### Mesh refinement (PSPG, scaled β, fluid-only pressure)

| mesh_size | nodes | fluid tets | p DOFs | Φ_in (m³/s) | Φ_out (m³/s) | imbalance | ‖div u‖_L2 | gates |
|-----------|-------|------------|--------|-------------|--------------|-----------|------------|-------|
| 0.12 | 108 | 237 | 72 | 2.27e-4 | −3.83e-5 | 1.42 | 1.39e-3 | fail |
| 0.08 | 200 | 483 | 136 | 3.90e-4 | −4.08e-4 | 0.047 | 1.00e-3 | **pass** |
| 0.06 | 298 | 771 | 204 | 1.56e-4 | −3.48e-5 | 1.27 | 6.56e-4 | fail |

### Root causes (flow)

1. **Diagnostics (v0.3.1):** x-only port selection inflated imbalance; fixed by streamwise normal alignment — not the dominant error at v0.4.1.
2. **Discretization:** Equal-order P1–P1 + PSPG + grad-div on a **3D developing open duct** does not yield mesh-monotone port flux balance; ‖div u‖_L2 stays O(10⁻³) while port imbalance swings between ~5% and >100% across the three Gmsh sizes.
3. **Pressure space (v0.4.1):** Restricting pressure to fluid nodes removes arbitrary solid-only pins; **does not** fix coarse/fine port imbalance (same table as v0.4).
4. **Taylor–Hood P2–P1:** Not implemented (P1-only Gmsh export). Documented in `FLOW_DISCRETIZATION.md`; PSPG remains production with stated limitations.

## CHT: global conservation vs local P1 flux

**Assembly:** Shared-node P1 temperature on fluid and solid tets; semi-implicit backward Euler step (`cht_transient.py`).

**Local side fluxes:** `interface_heat_flux` integrates k ∂T/∂n on interface triangles using adjacent-tet gradients (overlap-based orientation, v0.4).

**Global budget:** `transient_energy_balance` (`cht_energy.py`) closes

ΔE/dt ≈ Q_f + Q_s + (Φ_in − Φ_out) + Q_vol

with **implied fluid interface power** Q_f,implied = −(Q_s + ΔE/dt − (Φ_in − Φ_out) − Q_vol) from the same BE residual.

**Analytical FSI bar:** `test_fsi_conduction_analytical.py` — series heat rate, |Q_f+Q_s|/max(|Q|) < 12%, nx=8,16 and unequal k (nx=24, k1=2, k2=20).

**Coupled fin channel (flow + volumetric solid heat):** Local |Q_f| can be ≪ |Q_s| (thin fluid thermal boundary layer, advection-dominated fluid); **side-ratio gating** in `cht_gates.py` avoids treating noisy P1 gradients as global non-conservation. E2E reports local Q_f, Q_s, implied Q_f, enthalpy ports, ΔE, and energy residual.

**Zero-velocity conduction check:** `cht_fin_channel_conduction.py` — hot inlet, cold solid-only base nodes (`solid_base_dirichlet_nodes` excludes shared FSI nodes), u=0, mild volumetric solid heat (measurable flux on coarse mesh).

## End-to-end workflow

`fin_channel_e2e.py`: preflight → flow → **flow gates** → CHT only if flow passes → CHT gates. Refinement table always recorded; overall `ok` false if reference mesh fails flow or CHT gates.

## Numerical verification vs physical validation

All results above are **numerical verification** on procedural integral geometry. No experimental or high-order FR production claims.
