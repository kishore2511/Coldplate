# Architecture (single-phase production path)

```
Gmsh integral mesh (P1 tets)
    → preflight / port tagging (flow_channel_boundaries)
    → Taylor–Hood P2–P1 Stokes (flow_p2p1_*)
    → flow gates (flow_gates)
    → CHT semi-implicit BE (cht_transient + thermal_p1 + thermal_p2_advection)
    → interface diagnostics (thermal_p1 local + thermal_interface_flux Galerkin)
    → energy budget (cht_energy)
    → fin_channel_e2e (gates + JSON/MD)
```

**Discretization families (do not conflate):**

- **P1 Galerkin:** thermal, PSPG/penalty legacy flow
- **P2–P1 Taylor–Hood:** production steady Stokes
- **FR (Huynh):** 1D scalar + structured hex operator prototypes only

**Data coupling:** `P2VelocityField` (coefficients + DOF map) passes from flow solve to thermal advection.
