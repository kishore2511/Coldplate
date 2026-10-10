# Fluid Domain Extraction and Meshing

## CAD and Boolean workflow (spec B, R-014)

### Supported inputs (roadmap)

| Input | v0.1.0 | Notes |
|-------|--------|-------|
| Procedural channel + fin | ✅ `extract_channel_fluid_domain` | Gmsh OCC fragment |
| STEP/STP solids | Planned | `validate_step_path` + `gmsh.merge` |
| Multi-body assemblies | Planned | Fragment + connectivity labels |
| TPMS / lattices | Planned | Implicit/voxel fallback when Booleans fail |

### Extraction algorithm (current)

1. Define fluid **envelope** (bounding coolant region).
2. Subtract or fragment solid obstructions.
3. Label volumes (fluid vs solid) via topology + bounding-box heuristics (procedural) or user rules (CAD).
4. Detect **shared surfaces** with `getAdjacencies` (volume tags in `up` list for surfaces).
5. Assign physical groups: `coolant`, `fin_solid`, `fluid_solid_interface`.

### Diagnostics (spec B5)

`connectivity.diagnostics` reports:

- Missing fluid/solid components
- Missing interface patches
- Inlet/outlet unset (user confirmation required)

## Connectivity graph (spec B3)

`connectivity.graph.build_connectivity_from_extraction` produces:

- Volume nodes (`fluid`, `solid`)
- Interface patch node with surface tag list
- Edges linking volumes to interface

Future: port nodes, dead-end detection via graph search on fluid subgraph.

## Meshing (spec C, Gmsh)

| Feature | Status |
|---------|--------|
| Volume mesh generation | ✅ in channel demo |
| Physical group preservation | ✅ |
| Boundary layer refinement | Planned |
| Interface-aware refinement | Planned |
| Quality metrics (Jacobian) | Stub `mesh.quality` |
| FR mapping | ✅ 1D mapping helper |

### Gmsh environment note

Python `gmsh` module requires system `libGLU.so.1` (install `libglu1-mesa` on Debian/Ubuntu).

### TPMS / lattice fallback (documented, not implemented)

When exact Booleans are impractical:

1. Represent solid as implicit field (e.g., gyroid level-set).
2. Sample on structured grid; classify cells (fluid/solid).
3. Document geometric tolerance vs mesh spacing.

Risks: stair-step interfaces, non-conformal CHT unless mortar coupling applied (ADR-006).

## FR mesh compatibility (spec C2)

Before claiming high-order curved elements:

- Verify Jacobian positivity per element
- Face orientation consistency
- Geometric error vs CAD surface

v0.1.0 implements `mesh.fr_mapping.map_uniform_1d_elements` as pattern for metric storage.

## Tests

| Test | File |
|------|------|
| Channel Boolean + interface surfaces | `tests/test_geometry_extraction.py` |
| Connectivity graph | `tests/test_connectivity.py` |
