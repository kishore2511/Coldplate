# Defect register

| ID | Severity | Subsystem | Summary | Status |
|----|----------|-----------|---------|--------|
| D-001 | Numerical limitation | P2–P1 Stokes | UMFPACK direct solve fails on indefinite saddle; LSQR/direct hybrid used | **Mitigated v0.6** — `flow_p2_saddle_solve.py`, pressure pin documented |
| D-002 | Insufficient verification | P2–P1 | MMS used uniform body force in early tests | **Fixed v0.6** — `flow_p2_mms.py` full forcing |
| D-003 | Confirmed defect | CHT advection | Vertex-only velocity ignored P2 edge DOFs | **Fixed v0.6** — `thermal_p2_advection.py` |
| D-004 | Numerical limitation | CHT local flux | P1 gradient side flux noisy when \|Q_f\|≪\|Q_s\| | Open — report Galerkin + global budget |
| D-005 | Missing implementation | FR 3D production | No tet/hex FR flow/thermal production | Open |
| D-006 | Missing implementation | GPU | No verified GPU flow/thermal solve | Open |
| D-007 | Documentation | Capabilities | Risk of overstating FR/GPU/experimental validation | Ongoing — matrix in `VERIFICATION_MATRIX.md` |
| D-008 | Test / formulation | P2–P1 MMS | Sin-π box MMS is valid (homogeneous Dirichlet on ∂Ω). Non-monotone **vertex** error under x-only refinement was largely **mesh anisotropy**, not random noise. Parabolic-in-y “MMS” with all-wall no-slip is **invalid** — do not use. | **Documented** — `P2_MMS_AUDIT.md`, `uniform_refinement` |
| D-009 | Confirmed defect | P2–P1 MMS forcing | Body force used \(f=\nu\Delta u-\nabla p\) instead of \(f=-\nu\Delta u+\nabla p\) matching weak form | **Fixed** — `stokes_mms_force_vector` sign corrected |
| D-010 | Numerical limitation | P2–P1 MMS | After D-009 fix: volume \(\|u\|_{L^2}\) plateaus \(\sim 4.7\times10^{-3}\); relative pressure error \(\mathcal{O}(1)\) on P1 pressure space for linear \(p\) | **Open** — velocity/div/residual gates OK; pressure MMS not used for COMPLETE |

Add rows as audits find reproducible issues. Do not delete resolved items; mark status and link commit.
