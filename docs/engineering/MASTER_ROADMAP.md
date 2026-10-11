# ColdPlate Studio — master engineering roadmap

**Branch:** `cursor/coldplate-studio-foundation-341b`  
**Last verified baseline:** v0.6 (post-`dad6bf0`) — 59 pytest passed, 1 skipped

## Dependency-ordered program

| Phase | Scope | Status (v0.6) |
|-------|--------|----------------|
| 0 | Repository audit, defect register, architecture | **In progress** (this milestone) |
| 1 | Steady P2–P1 flow, MMS, fin-channel verification | **Active** — saddle docs, MMS, P2 advection |
| 1.4 | Transient incompressible (projection) | Implemented P1; needs P2 coupling |
| 2 | CHT conservative interface, full-P2 advection | **Active** — Galerkin flux + P2 advection |
| 3 | Geometry / fluid extraction / mesh QA | Partial (integral channel, preflight) |
| 4 | 3D FR production solver | Planned (1D/hex verification only) |
| 5 | GPU solver kernels | SpMV probe only |
| 6 | Two-phase / boiling | Planned |
| 7 | Design optimization | Planned |
| 8 | Integrated GUI | Not started |
| 9–10 | Reporting, packaging, release | Partial (E2E JSON/MD reports) |

## Next executable actions (after v0.6 commit)

1. Extend MMS to record observed convergence orders in CI with mesh-tolerant bands.
2. Transient P2–P1 or consistent P2 velocity in projection step.
3. CHT mesh-refinement table for coupled fin with P2 advection (automated in E2E).
4. FR 3D scalar advection-diffusion milestone (Phase 4.2 entry).

## Session resume

Update this file at the end of each agent session with commit SHA, test counts, and the single next task.
