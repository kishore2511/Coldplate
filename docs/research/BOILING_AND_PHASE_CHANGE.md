# Boiling and Phase-Change Models

## Scope

ColdPlate Studio targets liquid cooling with eventual **nucleate boiling**, capillary-fed evaporators, and two-phase microchannels. Spec E3 forbids equating boiling with instantaneous vapor generation when \(T > T_{sat}\).

## Model families

| Fidelity | Description | When appropriate |
|----------|-------------|------------------|
| Resolved interface evaporation | Mass transfer tied to interface temperature / saturation (R-011) | Interface cells resolved |
| Stefan / analytical benchmarks | 1D/2D phase-change solutions | Verification (spec G) |
| Empirical wall boiling (e.g., heat-partition) | Wall heat flux split to evaporation | Subgrid nucleation |
| Microlayer evaporation (R-012) | Near-wall thin film | Requires submodel + wall mesh criteria |

## Interfacial mass transfer (R-011 — Hardt & Wondra class)

Typical sharp-interface form:

\[
\dot m = \frac{\lambda}{h_{lv}} (T_\Gamma - T_{sat})
\]

with kinetic corrections optional. Energy source terms must be **consistent** with mass sources in momentum and energy equations.

**ColdPlate requirements:**

- Document saturation pressure/temperature and property tables
- Track latent heat separately in global energy balance (spec E4)
- Sign convention tests in Stefan problem

## Nucleation and wall boiling

Unresolved nucleation requires explicit submodels:

- Nucleation site density / activation criteria
- Bubble departure diameter and frequency correlations (if using RPI-style partitioning)
- Contact angle hysteresis near wall

**Do not claim** resolved nucleate boiling when mesh spacing \(\gg\) bubble departure diameter and no submodel is active.

## Validation strategy (spec G, R-013)

1. Stefan problem (analytical)
2. Bubble growth in superheated liquid (canonical simulation benchmarks)
3. Pool boiling — Gaertner experiments (R-013) **pending data acquisition**
4. Conjugate boiling with heater block (literature CHF studies — to be registered)

Record for each validation: fluid, pressure, heat flux, geometry, BCs, measured quantity, uncertainty, model assumptions.

## Implementation status

| Item | State |
|------|-------|
| Phase-change source terms | Planned |
| Latent heat bookkeeping | Planned |
| Wall boiling closures | Planned |
| Experimental boiling validation | Planned |

## Applicability limits (initial)

- Low-pressure water and dielectric coolants at near-atmospheric pressure: first targets
- Critical heat flux prediction: not claimed without dedicated validation
- Two-fluid pressure equilibrium assumptions must be stated in case setup files
