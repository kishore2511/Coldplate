"""Single integral mesh model: geometry, labels, and solver connectivity (spec B–C)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from coldplate_studio.geometry.extraction import FluidExtractionResult


@dataclass
class IntegralMesh:
    """
    One mesh representation shared by the FR solver and post-processor.

    Element type is recorded explicitly; high-order FR hex/prism extensions attach here.
    """

    nodes: np.ndarray  # (n, 3)
    fluid_tets: np.ndarray  # (m, 4) zero-based node indices
    solid_tets: np.ndarray  # (ms, 4) optional solid region
    interface_triangles: np.ndarray  # (k, 3)
    boundary_fluid_nodes: np.ndarray  # exterior coolant boundary (not FSI)
    physical_groups: dict[str, list[int]] = field(default_factory=dict)
    fr_element_family: str = "p1_tetrahedron"
    fr_polynomial_order: int = 1


@dataclass
class IntegralModel:
    """Geometry extraction + connectivity + mesh in one object."""

    extraction: FluidExtractionResult
    mesh: IntegralMesh
    fin_x0: float
    fin_thickness: float
    fin_height: float
