# Fin-channel v0.5 — Taylor–Hood P2–P1 flow and Galerkin interface flux

## Flow formulation (production)

**Taylor–Hood P2–P1 Stokes** on linear Gmsh tets (subparametric quadratic velocity, linear pressure):

- Modules: `tet_p2_basis.py`, `flow_p2_dof_map.py`, `flow_p2p1_operators.py`, `flow_p2p1_stokes.py`
- Weak form: ν∫∇u·∇v − ∫p∇·v = ∫f·v, ∫q∇·u = 0
- Velocity DOFs: fluid vertices + edge midpoints; pressure DOFs: fluid vertices (P1)
- No-slip: vertex pins + edge-midpoint pins on wall/no-slip segments
- Solver: sparse LSQR on saddle system (consistent rank-deficient modes removed by constraints)
- Divergence diagnostic: ‖B u‖₂ / √Ω (discrete weak divergence), not P1 projection of vertex values

**Legacy paths retained:** `formulation="pspg"`, `penalty`, `pspg_legacy_body_force` in `run_channel_flow`.

### Fin-channel refinement (P2–P1, body force drive)

| mesh_size | mass imb. | ‖B u‖/√Ω | gates |
|-----------|-----------|----------|-------|
| 0.12 | ~0.045 | ~3.6e-13 | pass |
| 0.08 | ~0.021 | ~3.0e-7 | pass |
| 0.06 | ~0.093 | ~5.4e-7 | pass |

Mass imbalance: |Φ_in+Φ_out|/mean(|Φ_in|,|Φ_out|) with P2 port quadrature.

## CHT interface

- **Local P1 gradient fluxes** (`thermal_p1.interface_heat_flux`): post-processed, can disagree when |Q_f|≪|Q_s|
- **Galerkin nodal exchange** (`thermal_interface_flux.galerkin_interface_conduction_rates`): Σ_interface (K_side T)_i — discrete weak-form conductive transfer; balanced on analytical FSI bar (imbalance &lt; 5%)
- E2E reports both local and Galerkin quantities plus global BE implied flux

## Verification status

Numerical verification on procedural geometry only — not experimental validation.
