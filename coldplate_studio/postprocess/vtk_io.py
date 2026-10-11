"""VTK export for 3D temperature and VOF fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_vtk_structured(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fields: dict[str, np.ndarray],
) -> None:
    """Write legacy VTK RECTILINEAR_GRID with point-centered scalars."""
    path = Path(path)
    nx, ny, nz = len(x), len(y), len(z)
    if any(f.shape != (nx, ny, nz) for f in fields.values()):
        raise ValueError("all fields must match grid shape")
    lines: list[str] = [
        "# vtk DataFile Version 3.0",
        "ColdPlate Studio export",
        "ASCII",
        "DATASET RECTILINEAR_GRID",
        f"DIMENSIONS {nx} {ny} {nz}",
        f"X_COORDINATES {nx} float",
        " ".join(f"{v:.8e}" for v in x),
        f"Y_COORDINATES {ny} float",
        " ".join(f"{v:.8e}" for v in y),
        f"Z_COORDINATES {nz} float",
        " ".join(f"{v:.8e}" for v in z),
        f"POINT_DATA {nx * ny * nz}",
    ]
    for name, arr in fields.items():
        flat = arr.ravel(order="F")
        lines.append(f"SCALARS {name} float 1")
        lines.append("LOOKUP_TABLE default")
        lines.extend(f"{v:.8e}" for v in flat)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vtk_unstructured_tets(
    path: Path,
    nodes: np.ndarray,
    tets: np.ndarray,
    point_fields: dict[str, np.ndarray],
) -> None:
    """Write legacy VTK UNSTRUCTURED_GRID for tetrahedral meshes."""
    path = Path(path)
    n_nodes = nodes.shape[0]
    lines = [
        "# vtk DataFile Version 3.0",
        "ColdPlate Studio tet export",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {n_nodes} float",
    ]
    for p in nodes:
        lines.append(f"{p[0]:.8e} {p[1]:.8e} {p[2]:.8e}")
    n_cells = tets.shape[0]
    size = n_cells * 5
    lines.append(f"CELLS {n_cells} {size}")
    for tet in tets:
        lines.append(f"4 {tet[0]} {tet[1]} {tet[2]} {tet[3]}")
    lines.append(f"CELL_TYPES {n_cells}")
    lines.extend("10" for _ in range(n_cells))
    lines.append(f"POINT_DATA {n_nodes}")
    for name, values in point_fields.items():
        if values.shape[0] != n_nodes:
            raise ValueError(f"field {name} must be nodal")
        lines.append(f"SCALARS {name} float 1")
        lines.append("LOOKUP_TABLE default")
        lines.extend(f"{v:.8e}" for v in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
