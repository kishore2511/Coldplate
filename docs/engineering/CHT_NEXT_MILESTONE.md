# Conjugate heat transfer — next milestone design

## Existing assets

- `IntegralMesh.interface_triangles` + `mesh/interface_geometry.py` (oriented normals, fluid→solid).
- `connectivity/interface_maps.py` (paired fluid/solid interface nodes).
- P1 thermal operators in `fr/tet_p1.py` / `solver_integral.py` (scalar transport prototype).
- Steady/transient **vector** flow kernels (`flow_p1_penalty.py`, `flow_p1_transient.py`).

## Coupling approach (planned)

| Physics | Discretization | DOFs |
|---------|----------------|------|
| Fluid energy | Transient advection–diffusion of `T` with **solved** **u**(**x**,t) | P1 nodal `T` on fluid tets |
| Solid conduction | `ρ c ∂T/∂t = ∇·(k ∇T)` | P1 nodal `T` on solid tets |
| Interface | Continuous temperature `T_f = T_s`; heat flux `q = -k_f ∇T_f·n = -k_s ∇T_s·n` | Mortar on `interface_triangles` |

### Semi-implicit staggered step

1. Advance fluid momentum/pressure (transient projection solver).
2. Fluid energy: `(M_f/dt + K_f + C(u)) T_f^{n+1} = M_f/dt T_f^n` with Robin/Neumann from solid if needed.
3. Solid: `(M_s/dt + K_s) T_s^{n+1} = M_s/dt T_s^n + B_{fsi}(T_f^{n+1})`.
4. Interface flux audit: `Q_f = ∫_Γ q_f dA`, `Q_s = ∫_Γ q_s dA`, report `|Q_f + Q_s| / mean(|Q|)`.

### Independent tests (implementable now)

- Interface area match between fluid and solid classifications (`test_interface_geometry.py`).
- Equal-and-opposite flux **prerequisite**: discrete flux operator symmetry on shared triangles (`tests/test_cht_interface_flux.py` — scaffold).

## Not in scope

- Boiling, VOF phase change, or phenomenological latent-heat sources.
