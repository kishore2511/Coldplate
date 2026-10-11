# Implementation audit baseline (branch `cursor/coldplate-studio-foundation-341b`)

## Test suite (reproduced)

`python3 -m pytest -q` → **42 passed, 1 skipped** (CUDA SpMV).

## Flow diagnostics definitions

| Quantity | Definition | Module |
|----------|------------|--------|
| ‖div u‖_L2 | √(Σ_e V_e (div u_e)² / Σ V_e) | `flow_periodic_stokes._divergence_l2` |
| Port flux Φ | ∫_Γ u·n dA (outward fluid normal) | `flow_p1_penalty._oriented_face_flux` |
| Mass imbalance | \|Φ_in+Φ_out\| / mean(\|Φ\|) | `flow_diagnostics`, `flow_channel` |
| Cross-section Q | trapezoid_y(u_x) × W at x=x_plane | `flow_flux_diagnostics` |
| Analytical Q | G W H³ / (12ν) | `analytical_parabolic_flow_rate` |

## Formulation choice (this milestone)

**Selected: PSPG-stabilized P1–P1 mixed Stokes** (`fr/flow_p1_pspg_mixed.py`).  
**Not selected (documented):** Taylor–Hood P2–P1 on integral Gmsh meshes (requires P2 mesh export and mixed element infrastructure).

Penalty Stokes remains for fin-channel regression; **not** the primary incompressibility path for verification.

## Fin channel

~**19.7%** port imbalance — penalty + 3D geometry; **not** validated by periodic parabolic benchmark.

## CHT

Conforming T with region-specific K, M; interface flux via oriented FSI normals. Energy balance includes ΔE/Δt, interface fluxes, inlet/outlet enthalpy proxy (`cht_energy.py`).
