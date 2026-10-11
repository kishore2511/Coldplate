# ColdPlate Studio — master engineering roadmap

**Branch:** `cursor/coldplate-studio-foundation-341b`  
**HEAD:** post v0.7.1 closure session (P2 MMS audit + completion contract)  
**Regression:** run `pytest` — expect 63+ passed, 1 skipped (CUDA)

See `COMPLETION_CONTRACT.md` for **COMPLETE** vs **IN PROGRESS** rules.

## Production focus (this program)

| Capability | Status | Evidence / gap |
|------------|--------|----------------|
| Integral fin-channel **P2–P1** steady flow | **IN PROGRESS** | Gates pass 0.12/0.08/0.06 (`test_flow_refinement_gates`); box MMS velocity ~0.3–0.5% after D-009; pressure MMS open (D-010); saddle LSQR (D-001) |
| **CHT** with P2 advection + Galerkin flux | **IN PROGRESS** | `thermal_p2_advection.py`, `cht_fin_channel_case.py`, E2E JSON; local gradient flux noisy (D-004); energy gates documented |
| Fin-channel **E2E** (flow → CHT → reports) | **IN PROGRESS** | `fin_channel_e2e.py`; multi-mesh flow gates; CHT metrics in JSON; not COMPLETE until CHT + flow acceptance fully met |
| P1 PSPG / penalty (legacy regression) | **IN PROGRESS** | Verified regressions; not production default |
| 3D FR production / GPU / boiling / GUI | **PLANNED** | Prototype hex operators exist; **not** in production workflow |

## Phase map (honest)

| Phase | Scope | Status |
|-------|--------|--------|
| 0 | Audit, defect register, architecture | **IN PROGRESS** |
| 1 | P2–P1 flow + MMS + fin verification | **IN PROGRESS** |
| 2 | CHT conservative interface, P2 advection | **IN PROGRESS** |
| 3 | Geometry / mesh QA | **IN PROGRESS** |
| 4 | 3D FR production | **PLANNED** |
| 5–10 | GPU, multiphase, GUI, release | **PLANNED** |

## Next corrective actions (ordered)

1. **P2–P1:** Pressure recovery / MMS metric for P1 \(p\) (D-010); optional MINRES/UMFPACK pivoting study for coarse meshes (D-001).
2. **CHT:** Mesh/time refinement table for coupled fin; tighten local flux diagnostics vs Galerkin (D-004).
3. **E2E:** Parametric gates across fin mesh sizes (flow + CHT) without hard-coding 0.08 only.
4. Do **not** expand FR/GPU scope until rows above reach COMPLETE per contract.

## Session resume

Update commit SHA, pytest count, and the single next corrective action after each agent run.
