# Research Register Audit (Solver Specification Crosswalk)

**Audit date:** 2026-10-10 (revision 2)  
**Detailed design decisions:** [`SPEC_TOPIC_DECISIONS.md`](SPEC_TOPIC_DECISIONS.md)

## Access log

| ID | Source | Access outcome |
|----|--------|----------------|
| R-002 | arXiv:2408.16509 (PyFR v2.0.3 paper) | **Full text read** — FR, GPU, incompressible NS §3.6; no multiphase boiling |
| R-003 | PyFR v2.0.3 installation docs | **Official page read** |
| R-014 | Gmsh 4.14 manual | **Official manual read** — OCC, BooleanFragments, mesh |
| R-007 | Rider & Kothe 1998 JCP | **Abstract/metadata only** — not full PDF |
| R-001 | Huynh 2007 AIAA | **Not retrieved** |
| R-010 | Brackbill 1992 | **Not retrieved** |
| R-011 | Hardt & Wondra 2008 | **Not retrieved** |
| R-013 | Gaertner 1965 | **Not retrieved** |

## Spec crosswalk (priority topics)

| Spec § | Topic | Register / code | Validation |
|--------|-------|-----------------|------------|
| B2 | Fluid extraction | `integral_pipeline`, `geometry/extraction` | pytest geometry + integral |
| B3 | Connectivity graph | `connectivity/graph` | `test_connectivity` |
| B4 | FSI interface | `interface_maps`, fragment + `getAdjacencies` | interface area, triangle count |
| B3 | Connected fluid volumes | `mesh/fluid_connectivity` | single component channel test |
| B5 | Pre-solve gates | `diagnostics/preflight` | `test_integral_preflight_passes` |
| C | Integral mesh → FR | `IntegralMesh`, `tet_p1`, `solver_integral` | end-to-end channel |
| D | In-house FR | `fr/*` — **not PyFR** | 1D/hex + P1 tet tests |
| E1–E2 | Two-phase (prep) | `SPEC_TOPIC_DECISIONS` U-01 | passive α only; PLIC open |
| E3–E4 | Phase change / conservation | `conservation.py`, `solver_integral` | α audit; Stefan open |
| D4 | GPU | `gpu/backend` | fail-closed; kernels planned |

## Claims discipline (revision 2)

- Production path: **integral Gmsh tet mesh → P1 FR/DG thermal + passive VOF** — not structured hex resampling.
- PyFR: **method/GPU literature only** — not integrated.
- **Not claimed:** full literature review; validated boiling; two-phase PLIC solver.
