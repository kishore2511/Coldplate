# ColdPlate Studio

Research-driven, GPU-first thermal–fluid simulation for cold plates, microchannels, TPMS structures, and two-phase liquid cooling.

## Status (v0.1.0 — milestone 1–2 foundation)

| Area | State |
|------|--------|
| Research register & ADRs | Implemented |
| Procedural fluid-domain extraction (Gmsh) | Implemented, tested |
| Connectivity graph & diagnostics | Implemented, tested |
| 1D FR scalar solver (CPU reference) | Numerically verified (pytest) |
| GPU CUDA kernels | Planned (fail-closed `gpu_only` API present) |
| Incompressible 3D flow, CHT, two-phase, boiling | Planned |

See `docs/research/` for evidence-backed design notes and `coldplate_studio/capabilities.py` for capability states.

## Quick start

```bash
# System dependency for Gmsh Python bindings (Debian/Ubuntu)
sudo apt-get install -y libglu1-mesa

pip install -e ".[dev]"
pytest -q
coldplate capabilities
coldplate smoke-geometry
```

## Repository layout

```
coldplate_studio/
  geometry/       # Validation, Boolean fluid extraction
  connectivity/   # Graph + pre-solve diagnostics
  mesh/           # Quality hooks, FR mapping
  fr/             # Flux Reconstruction core (1D scalar)
  gpu/            # Backend selection & run diagnostics
docs/research/    # Research register, ADRs, V&V plan
tests/            # Verification suite (spec G)
```

## Implementation order

Follows spec section I: research → geometry/connectivity → mesh → FR verification → GPU → incompressible flow → energy/CHT → two-phase → boiling → AMR.

## License

MIT (see `LICENSE` when added). Third-party references (e.g., PyFR BSD-3-Clause) are documented in research notes; no PyFR source is vendored in this repository.
