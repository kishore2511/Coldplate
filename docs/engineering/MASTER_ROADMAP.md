# ColdPlate Studio — master engineering roadmap

**Branch:** `cursor/coldplate-studio-foundation-341b`  
**HEAD:** v0.7.2 — P2–P1 saddle corrections; CHT refinement evidence. Production flow and CHT remain **IN PROGRESS**.

## Production focus

| Capability | Status | Why not COMPLETE |
|------------|--------|------------------|
| P2–P1 fin-channel flow | **IN PROGRESS** | Mass/div/residual gates pass 0.12/0.08/0.06. Polynomial MMS (u in P2) velocity and **interior** pressure errors ~1e-8. Saddle still singular for UMFPACK (D-001). |
| CHT with P2 advection | **IN PROGRESS** | P2 velocity is wired into advection. Local and Galerkin interface fluxes do not match (D-004). Port enthalpy magnitude is unphysical (D-011). |
| Fin-channel E2E | **IN PROGRESS** | JSON/Markdown reports and nonzero exit on failed flow gates. Depends on the two rows above. |
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

1. **D-011:** Trace port enthalpy assembly; the 1e9 W scale must be explained or corrected before any energy gate can define COMPLETE.
2. **D-004:** Interface flux formulation (gradient vs Galerkin) on the fin mesh; do not treat a skipped side-ratio gate as conservation.
3. **D-001:** Remove or constrain the boundary-pressure null space so a direct saddle factorization is nonsingular on the channel problem.

## Session resume

Update commit SHA, pytest count, and the single next corrective action after each agent run.
