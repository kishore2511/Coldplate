# Two-Phase Interface Methods Review

## Requirements (spec E)

- Conservative liquid/vapor mass (global and bounded phase fraction)
- Curvature and surface tension without dominant parasitic currents
- Large interface deformation, topology change (merge/breakup)
- Compatibility with high-order FR stencils (avoid unbounded stencil coupling)
- GPU-friendly advection/redistribution
- Distinct treatment of **moving liquid–vapor interface** vs **stationary fluid–solid wall**

## Method comparison (evidence-based)

| Method | Conservation | Interface sharpness | Curvature | FR coupling | GPU cost | Robustness |
|--------|--------------|---------------------|-----------|-------------|----------|------------|
| Algebraic VOF | Good w/ compression | Diffuse | Noisy | Moderate | Low | Fair |
| Geometric VOF/PLIC (R-007) | Excellent | Sharp | From reconstructed geometry | Face fluxes natural | Medium | Good |
| Level-set (R-008) | Non-conservative alone | Sharp | Good | Reinit stencils | Medium | Good |
| Conservative level-set | Improved mass | Sharp | Good | Coupled stencils | Higher | Research |
| Coupled VOF+LS (R-009) | Strong | Sharp | Good | Two fields + redist | Higher | Strong candidate |

## Primary method selection (provisional — ADR-008)

**Provisional primary:** coupled **geometric VOF transport** with **level-set redistancing** for curvature and contact angle.

**Rationale:**

- PLIC addresses mass conservation and sharp fluxes at FR faces (R-007).
- Level-set improves curvature stability vs height-function on unstructured meshes (R-008).
- Coupled approach limits spurious interface thickness (R-009).

**Fallback:** pure geometric VOF if GPU implementation complexity blocks milestone 10 schedule.

Final selection requires:

1. Static bubble / Laplace pressure test (spec G)
2. Parasitic current metric vs Brackbill CSF baseline (R-010)
3. Deforming interface translation on FR mesh

## Interface forces (R-010)

- Continuum Surface Force (CSF) model for surface tension
- Balanced-force discretizations to reduce spurious currents (literature follow-up in implementation milestone)
- Contact angle at walls: separate from liquid–vapor interface detection on solid walls

## FR-specific considerations

- Use face-based flux formulation for VOF fraction transport to match FR Riemann structure
- Avoid mixing low-order donor-cell advection with high-order momentum unless verified
- hp-refinement near interface deferred until fixed-mesh transport verified (spec C3, I)

## Implementation status

| Component | State |
|-----------|-------|
| Multiphase module package | Planned (`coldplate_studio.multiphase` — not yet created) |
| Interface transport | Planned |
| Surface tension | Planned |
| Contact angle | Planned |

## Uncertainties

- Sharp-interface phase change on coarse FR meshes near contact line (R-011, R-012)
- Need subgrid nucleation model when mesh cannot resolve microlayer (spec E3)
