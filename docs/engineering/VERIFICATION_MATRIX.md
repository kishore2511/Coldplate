# Verification matrix (summary)

| Capability | Formulation | Key tests | Refinement evidence |
|------------|-------------|-----------|---------------------|
| P2–P1 Stokes | Taylor–Hood tet | `test_flow_p2p1.py`, `test_flow_p2_mms.py` | MMS nx=4,6,8; fin 0.12/0.08/0.06 |
| PSPG P1–P1 | Stabilized | `test_flow_pspg.py`, `test_fin_channel_pspg.py` | Regression only |
| P1 thermal | Galerkin | `test_thermal_conduction.py` | 1D conduction |
| P2 thermal advection | u_h P2, T P1 | `test_thermal_p2_advection.py` | Constant-u parity |
| CHT interface | Shared-node + Galerkin flux | `test_thermal_galerkin_interface.py`, FSI bar | nx=8,16 |
| E2E fin channel | P2 flow + CHT | `test_fin_channel_e2e.py` | mesh 0.08 gate |

Physical / experimental validation: **none claimed**.
