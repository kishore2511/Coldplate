# Verification matrix (summary)

| Capability | Formulation | Key tests | Refinement / acceptance |
|------------|-------------|-----------|-------------------------|
| P2–P1 Stokes (production) | Taylor–Hood tet | `test_flow_p2p1.py`, `test_flow_refinement_gates.py` | Fin mesh 0.12/0.08/0.06: mass \<20%, div, residual gates |
| P2–P1 box MMS | Manufactured sin-π | `test_flow_p2_mms.py` | `P2_MMS_AUDIT.md`; uniform \(h\); **no** optimal order claim |
| PSPG P1–P1 | Stabilized | `test_flow_pspg.py`, `test_fin_channel_pspg.py` | Regression |
| P1 thermal | Galerkin | `test_thermal_conduction.py` | Two-material bar |
| P2 thermal advection | \(u_h\) P2, \(T\) P1 | `test_thermal_p2_advection.py` | Full P2 field in `ConjugateHeatTransferSolver` |
| CHT interface | Gradient + Galerkin | `test_thermal_galerkin_interface.py`, `test_cht_verification.py` | Energy residual; FSI bar |
| E2E fin channel | P2 flow + CHT | `test_fin_channel_e2e.py` | JSON: `solver_method`, Galerkin flux, energy fields |
| Hex FR scalar (non-production) | FR strong form | `test_fr_3d_advdiff_mms.py` | Operator audit only — **not** production |

Physical / experimental validation: **none claimed**.
