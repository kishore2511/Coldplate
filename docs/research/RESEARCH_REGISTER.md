# ColdPlate Studio — Research Register

Living bibliography and evidence log for numerical architecture decisions. Each entry records what was reviewed, what is implementable in-house, and what remains uncertain.

**Legend:** ✅ accessible · ⚠️ partial access · ❌ not accessed (findings not asserted)

| ID | Resource | Year | Area | Access | ColdPlate relevance |
|----|----------|------|------|--------|---------------------|
| R-001 | Huynh, *A Flux Reconstruction Approach to High-Order Schemes Including Discontinuous Galerkin Methods*, AIAA-2007-4079 | 2007 | FR core | ❌ | Foundational FR — **full text not retrieved**; cited in ADR-001 |
| R-002 | Vincent, Witherden et al., *PyFR v2.0.3: Towards Industrial Adoption…*, arXiv:2408.16509 | 2024 | FR/GPU literature | ✅ | **Paper read** — FR method & GPU patterns only; not a ColdPlate dependency |
| R-003 | PyFR v2.0.3 documentation — installation, user guide, performance | 2024 | Software (ref) | ✅ | **Docs read** — backend requirements; not bundled |
| R-004 | Kopriva, *Implementing Spectral Element Methods* (nodal DG differentiation) | 2009 | FR/DG | ⚠️ | Barycentric differentiation matrices (implementation reference) |
| R-005 | Chorin, *A Numerical Method for Solving Incompressible Viscous Flow Problems* | 1967 | Pressure | ⚠️ | Projection — metadata only in register |
| R-006 | Chorin, *Numerical Solution of the Navier-Stokes Equations* (artificial compressibility) | 1967 | Low-Mach | ⚠️ | Referenced via R-002 §3.6; primary not re-read here |
| R-007 | Rider & Kothe, *Reconstructing Volume Tracking* | 1998 | VOF | ⚠️ | **Abstract only** — PLIC/geometric VOF candidate |
| R-008 | Olsson & Kreiss, *A Level Set Method for Fluid Interface Problems* | 2005 | Level-set | ⚠️ | Register entry — primary not inspected this audit |
| R-009 | Sussman, *A Level Set Approach for Accurate, Conservative… Coupled with Level Set* | 2003 | CLSVOF | ⚠️ | Register entry — primary not inspected this audit |
| R-010 | Brackbill, Kothe, Zemach, *A Continuum Method for Modeling Surface Tension* | 1992 | CSF | ❌ | **Not retrieved** — parasitic current control (planned read) |
| R-011 | Hardt & Wondra, *Evaporation Model for Interfacial Flows…* | 2008 | Phase change | ❌ | **Not retrieved** — do not assert model details |
| R-012 | Tanasawa, *Advances in Condensation Heat Transfer* (microlayer) | 1991 | Boiling | ⚠️ | Wall boiling fidelity limits |
| R-013 | Gaertner, *Photographic Study of Nucleate Pool Boiling…* | 1965 | Validation | ⚠️ | Experimental benchmark target (data access TBD) |
| R-014 | Gmsh documentation — OpenCASCADE kernel, Boolean, mesh | 2024 | Mesh/CAD | ✅ | Fluid extraction & meshing |
| R-015 | ASME K-16 / IEEE ITherm Cold Plate Design Competition materials (hanhuark repo) | 2024 | Benchmark | ✅ | Cold-plate FoM and geometry envelopes |

## Entry template (use for new sources)

```
### [ID] Full title
- Authors:
- Year:
- DOI / URL:
- Governing equations:
- Numerical method:
- Interface / phase-change treatment:
- Discretization (space/time):
- Mesh / geometry requirements:
- Pressure / coupling:
- GPU / performance data:
- Verification cases cited:
- Experimental validation cited:
- Limitations:
- Implementable in ColdPlate Studio:
- Uncertainties / follow-up:
```

## Initial implementability notes

| Capability | Status in repo (v0.3.0) | Evidence gate |
|------------|-------------------------|---------------|
| Integral mesh pipeline | `mesh.integral_pipeline` | `test_integral_pipeline` |
| Fluid component connectivity | `mesh.fluid_connectivity` | single-component channel test |
| Pre-solve preflight | `diagnostics.preflight` | `test_integral_preflight_passes` |
| P1 FR thermal + passive VOF | `fr.solver_integral` | `test_fr_3d`, channel e2e |
| Conservation α audit | `diagnostics.conservation` | `test_conservation_audit_rejects_invalid_alpha` |
| 1D / hex FR verification | `fr.scalar_solver`, `fr.deriv3d` | operator tests |
| Two-phase PLIC / LS solver | Planned | U-01 benchmarks |
| Validated phase change / boiling | Planned | U-04 Stefan; R-011 full text |

## Inaccessible references (do not invent findings)

| Resource | Reason |
|----------|--------|
| Full Gaertner (1965) digitized bubble statistics | Not retrieved in this milestone; conditions to be confirmed before validation case implementation |
| Proprietary cold-plate vendor internal test reports | Not available |

## Audit trail

Crosswalks: [`RESEARCH_AUDIT.md`](RESEARCH_AUDIT.md), [`SPEC_TOPIC_DECISIONS.md`](SPEC_TOPIC_DECISIONS.md) (2026-10-10).

Update this register when ADRs change or new verification cases are added.
