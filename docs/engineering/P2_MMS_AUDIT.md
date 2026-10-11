# P2–P1 manufactured Stokes MMS audit (sin-π box)

**Manufactured fields** (`flow_manufactured.manufactured_velocity_field`):

- \(u_x = U_0\, y(H-y)\, z(W-z)\,\sin(\pi x/L)\), \(u_y=u_z=0\), zero on all six box faces.
- \(p = \partial p/\partial x \cdot x\) with \(\partial p/\partial x = \texttt{grad\_p}\) (default \(-1\)).

**Strong form** matching weak assembly `assemble_p2p1_stokes` (\(\nu\int\nabla u\cdot\nabla v - \int p\,\mathrm{div}\,v = \int f\cdot v\)):

\[
f = -\nu\Delta u + \nabla p,\quad f_x = -\nu\Delta u_x + \texttt{grad\_p}.
\]

## Root causes identified

| Issue | Effect | Status |
|-------|--------|--------|
| MMS forcing used \(f_x = +\nu\Delta u_x - \texttt{grad\_p}\) (sign error vs weak form) | Biased solution; masked as “small” vertex error on coarse meshes | **Fixed** (`flow_p2_mms.stokes_mms_force_vector`) |
| Refinement only in \(x\) with fixed \(n_y,n_z\) | Non-uniform \(h\); apparent non-monotone \(u_x\) errors | **Fixed** — `uniform_refinement` uses \(h=L/n_x\) in all directions |
| Treating \(n_x=n_y\) as “uniform” on a 1×0.1×0.2 box | 10× cell aspect ratio; LSQR solution blew up (\(\|u\|_\infty \gg 1\)) | **Fixed** — \(n_y=\mathrm{round}(H/h)\), \(n_z=\mathrm{round}(W/h)\) |
| Exact velocity \(\notin\) P2 per tet globally; P1 pressure on Q1 | Volume \(\|u\|_{L^2}\) plateaus; pressure \(L^2\) relative \(\mathcal{O}(1)\) | **Open** (D-010) — not a substitute for channel verification |
| Indefinite saddle; UMFPACK singular | LSQR used; residual small but not certifying optimal order | **Mitigated** (D-001) |

## Measured results (uniform \(h\), \(\nu=10^{-3}\), `grad_p=-1`, after fixes)

```
| nx | h    | ‖ux‖_L2 vert | ‖u‖_L2 vol P2 | ‖div‖   | residual |
|----|------|--------------|---------------|---------|----------|
| 4  | 0.25 | 2.64e-03     | 4.69e-03      | ~1e-15  | 4.2e-12  |
| 8  | 0.125| 2.78e-03     | 4.69e-03      | ~1e-15  | 1.7e-11  |
| 12 | 0.083| 2.83e-03     | 4.69e-03      | ~1e-15  | 1.7e-11  |
```

- **Discrete divergence** and **algebraic residual** satisfy tight tolerances.
- **Velocity** vertex \(L^2\) \(\approx 0.27\%\) of \(\|u\|_\infty\) scale; volume P2 \(L^2\) plateaus \(\approx 0.47\%\).
- **Observed optimal \(P2\) convergence rate is not claimed** — volume error is flat across 4→12; pressure relative error remains \(\mathcal{O}(1)\).

## Production fin-channel (separate acceptance)

Taylor–Hood P2–P1 with port BCs: mass imbalance \(<20\%\), divergence and residual gates on mesh sizes 0.12 / 0.08 / 0.06 (`test_flow_refinement_gates.py`). This is the **production** path; box MMS is a **supplementary** diagnostic.
