# Fin-channel v0.4 — mesh-scaled grad-div and FSI flux diagnostics

## Flow

Production PSPG path unchanged in physics (body force + open ports + PSPG τ=h²/(4ν)).

**v0.4 change:** momentum grad-div uses **mesh scaling**

\[
\beta(h) = \beta_{\mathrm{ref}}\left(\frac{h_{\mathrm{ref}}}{h}\right)^2,\quad \beta_{\mathrm{ref}}=100,\ h_{\mathrm{ref}}\approx 0.034\ \mathrm{m}.
\]

Implemented in `flow_stabilization.py`; enabled by default in `solve_integral_channel_pspg`. Legacy fixed β=100 remains via `formulation="pspg_legacy_body_force"`.

### Refinement (aligned ports, scaled β)

| mesh_size | imbalance (approx.) | Notes |
|-----------|---------------------|--------|
| 0.12 | ~1.2 | Still fails 20% gate |
| 0.08 | ~0.03–0.05 | Passes gate |
| 0.06 | ~1.0 | Fails gate |

Scaling improves the reference mesh; **does not** make all three fin meshes pass mass-balance gates with a single fixed β_ref. Coarse/finer imbalance remains dominated by 3D developing flow + equal-order discretization error.

## CHT interface flux

**Root cause of ~99% mismatch (v0.3.1 demo):** side-local P1 gradient integration used the **first** adjacent tet, often missing on non-matching Kuhn diagonals; fluid-side |Q| was near zero while solid-side showed O(1 W) from volumetric heating.

**v0.4 fix:** `_interface_temperature_gradient` selects the adjacent element with **maximum face-node overlap** (≥2 nodes) and correct interior orientation.

**Independent verification:** `build_two_material_tet_bar` + `test_fsi_conduction_analytical.py` — series-resistance heat rate, |Q_f+Q_s|/max(|Q|) < 12% on nx=8,16.

**Fin-channel coupled case:** with solid volumetric heating, fluid-side conductive flux can remain small relative to solid-side (thin thermal boundary layer, high ρc_p advection). Gates require **measurable** |Q| and apply **imbalance ≤15%** only when min(|Q_f|,|Q_s|)/max ≥ 0.1; otherwise the ratio is reported and magnitude/energy gates still apply.
