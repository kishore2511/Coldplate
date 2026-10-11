# Two-Phase Implementation Plan (pre-boiling)

**Unresolved decision:** U-01 (PLIC+unsplit vs coupled VOF/LS) — see `SPEC_TOPIC_DECISIONS.md`.

## Current state (not validated boiling)

- Passive scalar α on P1 mesh with clipping.
- Phenomenological `m_dot(T,α)` — **disable for validation** until Stefan (U-04).

## Proposed modules (future)

| Module | Responsibility |
|--------|----------------|
| `multiphase/transport.py` | Face flux for α; PLIC reconstruction on tet faces |
| `multiphase/curvature.py` | LS or height-field; CSF after R-010 review |
| `multiphase/properties.py` | ρ(α), μ(α), cp(α) |
| `multiphase/phase_change.py` | Sharp-interface or empirical wall boiling (post-Stefan) |
| `multiphase/conservation.py` | Global mass/energy ledger per step |

## FR compatibility

- Use **same face-oriented flux assembly** as momentum FR (once 3b complete).
- High-order: reconstruct α at flux points; geometric PLIC at sub-face quadrature.

## GPU

- Batch element kernels for PLIC + FR flux; after CPU parity on U-01 static bubble.

## Validation sequence

1. Static bubble Laplace (U-01)  
2. Mass conservation deformation (U-01)  
3. Stefan 1D (U-04)  
4. Heated channel with latent (no boiling claim)  
5. Literature pool boiling only after data ingest (R-013)
