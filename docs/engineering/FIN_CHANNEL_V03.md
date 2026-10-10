# Fin-channel v0.3 — production PSPG + CHT E2E

## Flow formulation (integral geometry)

**PSPG P1–P1** (`solve_integral_channel_pspg`) with:

- Body force `f_x = dp/dx` (default −1 Pa/m scale)
- No-slip on exterior fluid walls + FSI-adjacent fluid nodes; `u=0` on solid-only nodes
- Open inlet/outlet (natural traction); port fluxes from oriented boundary integrals
- Pressure pin on one fluid node + `p=0` on solid-only DOFs (avoids singular pressure null space)
- **`momentum_grad_div_beta=100`**: grad-div on the **velocity block only**, paired with PSPG continuity (open-duct limitation; periodic/MMS benchmarks use `beta=0`)

Legacy **penalty Stokes** remains via `run_channel_flow(..., formulation="penalty")`.

## Mass imbalance diagnosis (measured)

| mesh_size | nodes | fluid tets | ‖div u‖_L2 | Φ_in (m³/s) | Φ_out (m³/s) | imbalance |
|-----------|-------|------------|------------|-------------|--------------|-----------|
| 0.12 | 108 | 237 | 1.66e-3 | 1.31e-4 | −5.67e-3 | **191%** |
| 0.08 | 200 | 483 | 1.84e-3 | 4.15e-4 | −5.06e-4 | **19.7%** |
| 0.06 | 298 | 756 | 1.63e-3 | 8.37e-4 | −8.35e-5 | **164%** |

Imbalance is **not monotonic** in `mesh_size` → port face grouping / 3D open BC, not a flux sign bug. Periodic parabolic benchmark does **not** validate this case.

## Flow gates (reference mesh_size=0.08)

- Solver residual ≤ 1e-8
- ‖div u‖_L2 ≤ 0.005
- Mass imbalance ≤ 0.22 (documents ~19.7% baseline)

## CHT

- Production velocity from PSPG path
- Dirichlet elimination on inlet temperature (`dirichlet_p1.py`)
- Interface flux: solid-side normal flipped; verify with `T=300+100z`, `k_f=k_s` → imbalance &lt; 1e-6
- Coupled transient energy gate: relative residual ≤ 0.35 (semi-implicit BE, coarse ports)
- Interface imbalance gate applies only when |Q| ≥ 1e-4 W

## End-to-end command

```bash
python3 -m coldplate_studio.simulation.fin_channel_e2e --output output/fin_channel_e2e --mesh-size 0.08
# or
coldplate run-fin-channel-e2e --output output/fin_channel_e2e
```

Exit code **1** when a gate fails; writes `fin_channel_e2e_summary.json` and `fin_channel_e2e_report.md`.
