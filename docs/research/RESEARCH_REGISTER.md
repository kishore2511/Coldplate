# ColdPlate Studio — Research Register

Living bibliography and evidence log for numerical architecture decisions. Each entry records what was reviewed, what is implementable in-house, and what remains uncertain.

**Legend:** ✅ accessible · ⚠️ partial access · ❌ not accessed (findings not asserted)

| ID | Resource | Year | Area | Access | ColdPlate relevance |
|----|----------|------|------|--------|---------------------|
| R-001 | Huynh, *A Flux Reconstruction Approach to High-Order Schemes Including Discontinuous Galerkin Methods*, AIAA-2007-4079 | 2007 | FR core | ✅ | Foundational FR formulation for in-house solver |
| R-002 | Vincent, Witherden et al., *PyFR v2.0.3: Towards Industrial Adoption…*, arXiv:2408.16509 | 2024 | GPU FR CFD | ✅ | GPU architecture reference (not a code fork) |
| R-003 | PyFR v2.0.3 documentation — installation, user guide, performance | 2024 | Software | ✅ | Backend capabilities, element types, plugins |
| R-004 | Kopriva, *Implementing Spectral Element Methods* (nodal DG differentiation) | 2009 | FR/DG | ⚠️ | Barycentric differentiation matrices |
| R-005 | Chorin, *A Numerical Method for Solving Incompressible Viscous Flow Problems* | 1967 | Pressure | ✅ | Projection / pressure-correction context |
| R-006 | Chorin, *Numerical Solution of the Navier-Stokes Equations* (artificial compressibility) | 1967 | Low-Mach | ✅ | PyFR incompressible path uses related ideas |
| R-007 | Rider & Kothe, *Reconstructing Volume Tracking* | 1998 | VOF | ✅ | PLIC/geometric VOF candidate |
| R-008 | Olsson & Kreiss, *A Level Set Method for Fluid Interface Problems* | 2005 | Level-set | ✅ | Interface geometry, reinitialization cost |
| R-009 | Sussman, *A Level Set Approach for Accurate, Conservative… Coupled with Level Set* | 2003 | CLSVOF | ✅ | Coupled VOF/LS trade study |
| R-010 | Brackbill, Kothe, Zemach, *A Continuum Method for Modeling Surface Tension* | 1992 | CSF | ✅ | Parasitic current control |
| R-011 | Hardt & Wondra, *Evaporation Model for Interfacial Flows…* | 2008 | Phase change | ✅ | Interfacial mass transfer law |
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

| Capability | Status in repo (v0.1.0) | Evidence gate |
|------------|-------------------------|---------------|
| 1D FR scalar transport | `coldplate_studio.fr` + pytest | Polynomial / advection tests |
| Gmsh Boolean fluid extraction | `geometry.extraction` | Channel fixture test |
| Connectivity graph | `connectivity.graph` | Interface adjacency test |
| Incompressible 3D FR | Planned | R-001, R-005, R-006, R-002 |
| Two-phase interface | Planned | R-007–R-011 |
| Boiling validation | Planned | R-013 + literature survey |

## Inaccessible references (do not invent findings)

| Resource | Reason |
|----------|--------|
| Full Gaertner (1965) digitized bubble statistics | Not retrieved in this milestone; conditions to be confirmed before validation case implementation |
| Proprietary cold-plate vendor internal test reports | Not available |

## Audit trail

A specification crosswalk with primary-source access notes and unresolved decisions is maintained in [`RESEARCH_AUDIT.md`](RESEARCH_AUDIT.md) (updated 2026-10-10).

Update this register when ADRs change or new verification cases are added.
