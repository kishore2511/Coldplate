# Research Register Audit (Solver Specification Crosswalk)

**Audit date:** 2026-10-10  
**Method:** Primary sources were opened and read where accessible (full PDF/HTML). Publisher paywalls and timeouts are recorded explicitly. Search-engine snippets were **not** used as evidence unless confirmed against a primary page.

## Access log (this audit)

| ID | Source | Access outcome |
|----|--------|----------------|
| R-002 | PyFR v2.0.3 paper, [arXiv:2408.16509](https://arxiv.org/abs/2408.16509) | **Full text read** (HTML/PDF extract) |
| R-003 | PyFR v2.0.3 installation docs | **Official page read** ([installation](https://pyfr.readthedocs.io/en/v2.0.3/installation.html)) |
| R-014 | Gmsh 4.14 manual | **Official manual read** (BooleanFragments, OCC kernel, physical groups) |
| R-007 | Rider & Kothe 1998, *JCP* 141:112–152 | **Abstract/metadata read** via [ACM DL DOI landing](https://doi.org/10.1006/jcph.1998.5906); **full PDF not retrieved** (paywall) |
| R-001 | Huynh 2007 AIAA-2007-4079 | **Not retrieved** in this audit (NASA NTRS timeout); retained as cited ADR basis from prior register |
| R-010 | Brackbill et al. 1992 JCP | **Not retrieved** (DOI fetch failed) |
| R-011 | Hardt & Wondra 2008 IJHMT | **Not retrieved** (DOI timeout) |
| R-013 | Gaertner 1965 pool boiling | **Not retrieved** |

## Specification topic crosswalk

| Spec topic (§) | Register coverage | Primary evidence used | ColdPlate decision | Validation requirement |
|----------------|-------------------|----------------------|--------------------|------------------------|
| FR / high-order (D) | `FR_SOLVER_REVIEW.md` | arXiv:2408.16509 (FR GPU, elements, filters); Huynh FR cited ADR-001 | Independent FR core; 1D verified; **3D tensor-product hex FR** for scalar thermal/VOF | Polynomial/RK4 gates G2; 3D diffusion/VOF tests |
| GPU (D4,F) | `GPU_PERFORMANCE.md` | PyFR installation (CUDA≥11.4, HIP, OpenCL); arXiv perf sections | Fail-closed `gpu_only`; CUDA kernels planned | CPU/GPU operator parity `@pytest.mark.gpu` |
| Fluid extraction (B2) | `FLUID_DOMAIN_AND_MESHING.md` | Gmsh manual §BooleanFragments, OCC import | `occ.fragment` for procedural channel | `test_geometry_extraction` |
| Interface connectivity (B3–B4) | `interface_maps.py`, connectivity graph | Gmsh `getAdjacencies` volume tags on surfaces | Patch list + areas; user confirms ports | Interface triangle count > 0; area > 0 |
| Mesh / FR mapping (C) | `mesh/fr_mapping.py`, `gmsh_export.py` | Gmsh manual mesh + physical groups | **Dual representation:** FR structured hex for solver; Gmsh tets for VTK | VTK export tests |
| Incompressible flow (D2) | ADR-003 | arXiv §3.6: artificial compressibility + dual-time [32–35] | Deferred; PyFR is reference only | Channel Poiseuille + divergence norms |
| Two-phase interface (E1–E2) | `TWO_PHASE_METHODS_REVIEW.md` | Rider & Kothe abstract (PLIC, unsplit) | **Passive VOF scalar** on 3D FR grid now; PLIC face fluxes later | Boundedness + mass trend tests |
| Phase change (E3–E4) | `BOILING_AND_PHASE_CHANGE.md` | Hardt & Wondra **not read**; phenomenological prototype only | `m_dot ∝ (T-T_sat) α(1-α)` flagged **unverified** | Stefan + energy balance before boiling claims |
| Conservation (E4) | `solver3d.conservation_report` | — | Track liquid mass, thermal energy, α bounds | Assert bounds; mass decreases when superheated |
| V&V (G) | `VERIFICATION_VALIDATION_PLAN.md` | — | Extended with 3D tests | pytest 12+ cases |

## Major design choices (trade-offs and implementation)

### 1. Automatic fluid-domain extraction (Gmsh OCC)

**Source:** Gmsh manual — OpenCASCADE kernel performs Booleans/fragments on native representations; `BooleanFragments` preserves shared boundaries between volumes (tutorial t16).

**Trade-offs:** Exact B-rep Booleans fail on dirty CAD/TPMS; implicit methods trade geometric fidelity for robustness.

**Implementation:** `extract_channel_fluid_domain` uses `occ.fragment`; fluid/solid classification by topology + bbox heuristics (procedural); interface surfaces via `getAdjacencies(2, tag)` **volume** list (`up`).

**Validation:** Fluid and solid volumes present; ≥1 interface surface; connectivity diagnostics pass before simulation export.

### 2. Fluid–solid interface connectivity

**Source:** Spec B4 (conformal shared boundary); Gmsh fragment workflow.

**Trade-offs:** Mortar coupling deferred (ADR-006) until conformal meshing fails on imported STEP.

**Implementation:** `connectivity/interface_maps.py` builds patch areas from interface triangles exported with the mesh.

**Validation:** `interface_triangles > 0`, `total_area_m2 > 0` in channel end-to-end test.

### 3. FR compatibility with two-phase formulation

**Source:** Rider & Kothe 1998 abstract — unsplit multidimensional transport; PyFR arXiv — face-based FR flux formulation.

**Trade-offs:** Coupled VOF/LS (ADR-008 provisional) adds curvature field cost; tensor hex FR cannot use arbitrary Gmsh tets without hex meshing or DGFEM on tets (future).

**Implementation (current):** Passive VOF and temperature use **P1 FR/DG simplex operators** on the **same integral Gmsh tet mesh** (`IntegralModel`). Structured-hex FR modules remain for operator verification only (ADR-007).

**Validation:** VOF ∈ [0,1]; mass decreases under superheat with prototype phase change; static bubble / PLIC tests still **open**.

### 4. GPU execution

**Source:** PyFR docs + arXiv (CUDA/HIP backends, device pointers, task graphs).

**Trade-offs:** Development in CPU NumPy reference first; GPU port must preserve operator semantics.

**Implementation:** Backend fails closed; no silent CPU fallback in `gpu_only` mode.

**Validation:** Pending CUDA hardware; kernel parity tests planned.

### 5. Transient phase change (prototype)

**Source:** Spec E3; Hardt & Wondra **unread** — prototype must not be labeled validated boiling.

**Trade-offs:** Phenomenological `m_dot` enables pipeline testing without resolved interface evaporation.

**Implementation:** `PhaseChangeConfig` in `fr/solver3d.py` with latent heating and α source.

**Validation:** Superheated mass-loss test; **Stefan problem required** before `benchmark_validated`.

## Unresolved decisions

| ID | Decision | Options | Proposed resolving experiment |
|----|----------|---------|------------------------------|
| U-01 | Primary two-phase integrator | Geometric PLIC+unsplit (R-007) vs coupled VOF/LS (R-009) | 2D/3D static bubble Laplace + mass drift on FR grid |
| U-02 | FR on Gmsh tets vs hex-only | Tet FR operators vs hex meshing of fluid domain | Compare Poiseuille error on hex vs tet mesh of same channel |
| U-03 | Incompressible pressure scheme | Artificial compressibility (PyFR path) vs projection | Manufactured solution + mass conservation vs pseudo-time cost |
| U-04 | Phase-change closure | Sharp-interface evaporation vs RPI wall boiling | 1D Stefan analytical + energy balance |
| U-05 | CHT coupling | Conformal vs mortar on imported STEP | Heat flux conservation across `interface_maps` patches |

## Claims discipline

- **Not claimed:** Full literature review; validated boiling; GPU speedup; PLIC interface reconstruction.
- **Claimed (with tests):** 3D FR-based thermal + passive VOF on structured hex; Gmsh extraction + interface triangles; VTK + contour post-processing.

Update `RESEARCH_REGISTER.md` when primary texts are obtained for R-001, R-010, R-011, R-013.
