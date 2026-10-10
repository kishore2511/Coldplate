# Architecture Decision Records (ADR)

## ADR-001 — Primary high-order spatial discretization

**Status:** Accepted (milestone 1)  
**Context:** ColdPlate Studio requires an original FR solver, not a relabeled FV/FD scheme.  
**Decision:** Implement Huynh-style Flux Reconstruction with Radau correction functions (nodal DG-equivalent branch) on reference elements, starting with 1D verification and extending to quadrilateral/hexahedral elements before prisms/tets.  
**Evidence:** R-001, R-004  
**Consequences:** Correction-function choice must be documented per element family; alternative FR branches (SD-like) are deferred until stability studies exist.

## ADR-002 — PyFR relationship

**Status:** Accepted  
**Context:** Spec requires PyFR as technical reference for GPU FR architecture.  
**Decision:** Treat PyFR v2.0.3 as an **external reference implementation** (BSD-3-Clause). Do not vendor or silently reuse PyFR source; independently implement published FR mathematics. Benchmark against PyFR where licenses and problem definitions allow.  
**Evidence:** R-002, R-003  
**Consequences:** Feature parity claims must distinguish ColdPlate capabilities from PyFR (e.g., PyFR provides incompressible NS via artificial compressibility + dual-time; it does **not** provide boiling/VOF multiphysics in the surveyed release notes).

## ADR-003 — Incompressible / low-Mach pathway

**Status:** Proposed (implementation milestone 8)  
**Context:** Liquid cold-plate coolant is incompressible/low-Mach.  
**Decision:** Evaluate (a) pressure-projection, (b) artificial compressibility, (c) all-Mach preconditioning. **Provisional:** prototype artificial-compressibility FR following documented PyFR approach as one candidate, with divergence and mass-conservation gates before acceptance.  
**Evidence:** R-005, R-006, R-002 §3.6  
**Consequences:** Pressure solver cost and stiffness for phase change must be measured explicitly (spec F).

## ADR-004 — CAD kernel and fluid extraction

**Status:** Accepted  
**Context:** Automatic coolant-domain extraction is first-class.  
**Decision:** Use **Gmsh OpenCASCADE kernel** for Booleans, fragmentation, STEP import, and initial meshing. Document voxel/implicit fallback for TPMS when Booleans fail tolerance checks.  
**Evidence:** R-014  
**Consequences:** Cloud/dev images require `libGLU` for Gmsh Python bindings; CI must install it.

## ADR-005 — Connectivity and boundary classification

**Status:** Accepted  
**Context:** Spec forbids guessing BCs from face location alone.  
**Decision:** Build a semantic connectivity graph (NetworkX internally; exportable JSON) with automatic **candidates** and mandatory user confirmation for inlet/outlet/symmetry. Block solver start on missing required interfaces.  
**Consequences:** UI layer must surface unclassified faces (spec B5).

## ADR-006 — Fluid–solid interface coupling

**Status:** Accepted  
**Context:** Conjugate heat transfer requires matched or documented nonconformal coupling.  
**Decision:** Prefer conformal interface meshes from shared Boolean boundaries; implement mortar/conservative interpolation only when conformal meshing fails, with explicit conservation tests.  
**Consequences:** Separate data structures for stationary wall vs moving liquid–vapor interface (spec B4, E).

## ADR-007 — Meshing subsystem boundary

**Status:** Accepted  
**Decision:** Mesh generation is a **separate subsystem** from the FR residual engine. Gmsh produces labeled groups; `mesh.fr_mapping` converts to FR element data with Jacobian/orientation tests.  
**Evidence:** R-014, spec C

## ADR-008 — Two-phase primary interface method (provisional)

**Status:** Proposed — selection gate open  
**Options:** Geometric VOF/PLIC (R-007), coupled level-set/VOF (R-009), conservative level-set (R-008).  
**Provisional lean:** **coupled geometric VOF + level-set redistancing** for FR compatibility and curvature quality, unless GPU complexity forces geometric VOF-only for milestone 10.  
**Final selection:** Record in `TWO_PHASE_METHODS_REVIEW.md` after static-bubble and deformation prototypes.

## ADR-009 — GPU execution model

**Status:** Accepted (phased)  
**Decision:** Matrix-free element-local operators; GPU-resident fields in production mode. Milestone 7 introduces CUDA kernels (target: CUDA via CuPy/Numba/custom). **GPU-only mode fails closed** if no device (see `gpu.backend`).  
**Evidence:** R-002, R-003 performance tuning docs  
**Consequences:** This cloud environment has no NVIDIA GPU; GPU tests are marked `@pytest.mark.gpu` and skipped until hardware is present.

## ADR-010 — Verification before AMR and boiling claims

**Status:** Accepted  
**Decision:** No dynamic AMR or boiling validation claims until fixed-mesh FR + interface transport gates in spec G pass.  
**Evidence:** Spec I, G
