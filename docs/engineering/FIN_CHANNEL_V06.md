# Fin-channel v0.6 — verified saddle treatment and full-P2 thermal advection

## P2–P1 saddle system

- Continuous pressure null space: constant p; **one pressure DOF pinned to 0**.
- Solver: `solve_stokes_saddle` tries sparse direct solve, falls back to **LSQR** on indefinite `[K B^T; B 0]` when UMFPACK reports singularity (structurally indefinite, not necessarily wrong BCs).
- `solver_method` recorded on `P2P1StokesResult` / `FlowSolveResult`.

## Manufactured Stokes MMS

- `stokes_mms_force_vector`: f = ν∇²u − ∇p for documented polynomial u (see `flow_manufactured`).
- RHS assembled with P2 test functions and 4-point tet quadrature (`assemble_p2_mms_body_force_rhs`).
- `run_p2p1_mms_study`: reports ux L2, semi-H1, mean-zero pressure L2, ‖B u‖/√Ω, residual.

## Thermal advection

- `assemble_p2_velocity_advection`: ∫ ρ c_p (u_h·∇ψ_j) ψ_i with full **P2** u_h at quadrature.
- `ConjugateHeatTransferSolver(..., p2_velocity_field=...)` used from fin-channel coupled case when flow is P2–P1.

## Fin-channel flow (unchanged v0.5 table, P2–P1)

All three mesh sizes pass current flow gates (mass imb &lt; 20%, weak div).

## Limitations

- LSQR may still be used on difficult meshes; direct path preferred when residual &lt; 1e-7.
- MMS convergence rates are problem- and mesh-dependent; tests use conservative improvement band, not optimal O(h³) claim.
- Local interface P1 flux remains diagnostic; Galerkin exchange is the weak-form conservative metric.
