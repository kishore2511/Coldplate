# Defect register

| ID | Severity | Subsystem | Summary | Status |
|----|----------|-----------|---------|--------|
| D-001 | Numerical limitation | P2–P1 Stokes | UMFPACK reports an exactly singular saddle (`MatrixRankWarning`). Production and polynomial MMS runs use LSQR with algebraic residual typically `< 1e-8`. MINRES is used only if that residual is not tight. | **Mitigated v0.7.2** — not a certified nonsingular factorization |
| D-004 | Confirmed limitation | CHT interface | Gradient and Galerkin fluid/solid fluxes disagree (imbalance ~1) on fin meshes 0.12/0.08/0.06; `\|Q_f\| ≪ \|Q_s\|`. The CHT gate skips consistency when the side ratio is `< 0.1`, so `cht_gates_ok` can be true without local conservation. Solid `T_max` stays ≈ 300 K. | **Open** — `docs/engineering/cht_refinement_v072.json` |
| D-010 | Confirmed defect (fixed for in-space MMS) | P2–P1 | O(1) pressure error had several causes: pressure block used `+Bᵀ` instead of `−Bᵀ`; wall edges with one interior endpoint were pinned; 5-tet hex split was nonconforming; quadrature weights summed to 1/6; nonzero Dirichlet did not lift divergence rows. | **Fixed for polynomial channel** — interior `‖p‖` and `‖u‖` ~ 1e-8. Boundary pressure DOFs under full-wall Dirichlet remain undetermined (excluded from that norm). Sin-π field is not in P2; volume error plateaus ~ 4.6e-3. |
| D-011 | Confirmed defect | CHT energy | Port enthalpy in/out is ~1e9–1e10 W while volumetric heating is 100 W and solid temperature rise is ~0.02 K. Relative energy residual can still pass the 0.35 gate. | **Open** |
| D-002 | Insufficient verification | P2–P1 | MMS used uniform body force in early tests | **Fixed v0.6** — `flow_p2_mms.py` full forcing |
| D-003 | Confirmed defect | CHT advection | Vertex-only velocity ignored P2 edge DOFs | **Fixed v0.6** — `thermal_p2_advection.py` |
| D-005 | Missing implementation | FR 3D production | No tet/hex FR flow/thermal production | Open |
| D-006 | Missing implementation | GPU | No verified GPU flow/thermal solve | Open |
| D-007 | Documentation | Capabilities | Risk of overstating FR/GPU/experimental validation | Ongoing — matrix in `VERIFICATION_MATRIX.md` |
| D-008 | Test / formulation | P2–P1 MMS | Sin-π box MMS is valid (homogeneous Dirichlet on ∂Ω). Non-monotone **vertex** error under x-only refinement was largely **mesh anisotropy**, not random noise. Parabolic-in-y “MMS” with all-wall no-slip is **invalid** — do not use. | **Documented** — `P2_MMS_AUDIT.md`, `uniform_refinement` |
| D-009 | Confirmed defect | P2–P1 MMS forcing | Body force used \(f=\nu\Delta u-\nabla p\) instead of \(f=-\nu\Delta u+\nabla p\) matching weak form | **Fixed** — `stokes_mms_force_vector` sign corrected |

Add rows as audits find reproducible issues. Do not delete resolved items; mark status and link commit.
