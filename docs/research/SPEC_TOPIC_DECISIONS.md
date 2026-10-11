# Specification Topics — Design Decisions with Sources

**Purpose:** Audit the research register against the solver specification **before** implementing the full two-phase solver. Each subsection cites accessible primary material, states trade-offs, implementation implications, and validation requirements. Full texts not retrieved are marked **not reviewed** — no findings are invented from them.

**Primary access log (this revision):**

| ID | Source | Inspected material |
|----|--------|-------------------|
| R-001 | Huynh, AIAA-2007-4079 | **Not reviewed** (NTRS fetch timeout) — FR method attributed from secondary citations in ADR-001 |
| R-002 | Vincent et al., arXiv:2408.16509 | **Full paper text** — FR on mixed elements, GPU backends, incompressible NS via artificial compressibility + dual-time (§3.6); **no VOF/boiling** |
| R-003 | PyFR v2.0.3 installation | **Official docs** — CUDA ≥11.4, HIP, OpenCL, OpenMP; **reference only**, not a ColdPlate dependency |
| R-007 | Rider & Kothe, JCP 141 (1998) | **Abstract/landing page only** (DOI 10.1006/jcph.1998.5906) — PLIC, unsplit 2D VOF |
| R-010 | Brackbill et al., JCP 100 (1992) | **Not reviewed** (fetch timeout) |
| R-011 | Hardt & Wondra, IJHMT (2008) | **Not reviewed** (fetch timeout) |
| R-014 | Gmsh 4.14 manual | **Official manual** — OCC kernel, `BooleanFragments`, physical groups, mesh on native B-rep |

---

## B — Automatic fluid-domain extraction

**Spec:** Boolean coolant domain, cavity classification, validation before meshing.

**Source (R-014):** Gmsh performs Booleans/fragments on the **native OCC representation**; `BooleanFragments` makes interfaces conformal between operands.

**Decision:** `build_integral_channel_model()` — single `occ.fragment` of envelope and fin; fluid/solid volume tags; tet mesh with centroid-based fluid/solid split.

**Trade-offs:**

| Approach | Pros | Cons |
|----------|------|------|
| OCC fragment (chosen) | Exact shared FSI for procedural CAD | Fails on dirty STEP / TPMS without fallback |
| Voxel/implicit void | Robust for lattices | Geometric tolerance; stair-steps |
| Bounding-box fluid mask | Simple | Not spec-compliant for arbitrary CAD |

**Implementation:** `mesh/integral_pipeline.py`, `mesh/fluid_connectivity.py` (connected fluid components).

**Validation:** `test_geometry_extraction`, `test_integral_pipeline`, `test_channel_has_single_connected_fluid_component`, preflight gate `disconnected_fluid`.

---

## B3–B4 — Connectivity and fluid–solid interface

**Spec:** Connectivity graph; automatic FSI patches; no guessed inlet/outlet.

**Source (R-014):** Fragment preserves shared surfaces; mesh `getAdjacencies(2,·)` returns adjacent **volumes** in first list.

**Decision:** Graph (`connectivity/graph.py`) + triangle patches with areas (`interface_maps.py`) + **preflight** (`diagnostics/preflight.py`).

**Trade-offs:** Conformal FSI from fragment vs mortar coupling (ADR-006) when meshes diverge.

**Validation:** `interface_triangles > 0`, `total_area_m2 > 0`, `run_integral_preflight` in channel case.

---

## D — Flux Reconstruction (in-house method only)

**Spec:** Original FR solver; not mislabeled FV.

**Source (R-002):** Describes **FR spatial discretization** and GPU execution patterns; ColdPlate adopts **method ideas only** (ADR-002).

**Decision:**

- Verification: 1D/hex tensor FR (`fr/operators.py`, `fr/deriv3d.py`).
- Production 3D: **P1 tet** on integral mesh (`fr/tet_p1.py`, `fr/solver_integral.py`) — lowest-order nodal FR/DG branch.

**FR ↔ two-phase compatibility (ahead of full solver):**

| Requirement | P1 tet + passive α today | Target two-phase |
|-------------|--------------------------|------------------|
| Face fluxes for α | Galerkin advection matrix | PLIC unsplit (R-007 abstract) or coupled LS |
| Curvature / σ | Not implemented | CSF (R-010, not reviewed) |
| High-order | Hex FR operators tested separately | hp hex on `IntegralMesh` |

**Validation:** `test_fr_*`, `test_fr_3d`, `test_integral_*`; static bubble / PLIC **open** (U-01).

---

## D4 / F — GPU execution

**Source (R-003, R-002 §performance):** CUDA/HIP backends, device-resident data, task graphs for small cases.

**Decision:** `gpu/backend.py` — fail-closed `gpu_only`; **no FR kernels yet**.

**Trade-off:** CPU reference correctness before CUDA parity tests.

**Validation:** `@pytest.mark.gpu` operator parity (planned); `test_gpu_backend`.

---

## E3–E4 — Conservation and transient phase change

**Spec:** Mass, energy, α bounds; phase change not trivial boiling.

**Source (R-011):** **Not reviewed** — prototype uses phenomenological `m_dot ∝ (T-T_sat) α(1-α)` in `solver_integral.py`.

**Decision:** `ConservationReport` + `diagnostics/conservation.py` audits; phase change **disabled** for validation gates until Stefan case exists.

**Trade-offs:** Phenomenological source enables pipeline testing; must not be labeled boiling validation.

**Validation:** `audit_conservation_report`, `test_3d_phase_change_reduces_liquid_mass_when_superheated`; **Stefan benchmark** (U-04) before `benchmark_validated`.

---

## Unresolved decisions and benchmark experiments

| ID | Question | Experiment | Pass criterion |
|----|----------|------------|----------------|
| U-01 | PLIC+unsplit vs coupled VOF/LS | 2D/3D static bubble on FR mesh | \|Δp - σκ\| / σκ < 5%; mass drift < 1e-3/cycle |
| U-02 | P1 tet vs high-order hex FR | Same channel, matched DOF count | L2(T) vs refined reference; flux at FSI |
| U-03 | Pressure coupling | Manufactured incompressible solution | ‖∇·u‖ / ‖u‖ < 1e-3 |
| U-04 | Phase-change closure | 1D Stefan analytical | interface position error < 2% |
| U-05 | CHT FSI | Steady conjugate pipe | \|Q_fluid + Q_solid\| / \|Q_fluid\| < 1e-3 |

---

## Claims discipline

- Literature **not** fully reviewed: R-001, R-010, R-011, R-013.
- **Not claimed:** validated boiling, GPU production solver, geometric PLIC two-phase.
- **Claimed with tests:** integral extraction mesh, single fluid component (channel), preflight gates, α bounds audit, P1 thermal/VOF on integral mesh.
