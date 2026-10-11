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

Add rows as audits find reproducible issues. Do not delete resolved items; mark status and link commit.
