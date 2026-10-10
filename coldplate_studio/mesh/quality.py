"""Mesh quality inspection hooks (spec C1)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MeshQualityReport:
    n_nodes: int
    n_elements: int
    min_jacobian: float | None
    negative_jacobians: int
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.negative_jacobians == 0 and (self.min_jacobian is None or self.min_jacobian > 0.0)


def summarize_placeholder_mesh(n_nodes: int, n_elements: int) -> MeshQualityReport:
    """Placeholder until volumetric Jacobian queries are wired for all element types."""
    return MeshQualityReport(
        n_nodes=n_nodes,
        n_elements=n_elements,
        min_jacobian=None,
        negative_jacobians=0,
        warnings=["Jacobian audit not yet implemented for all FR element families."],
    )
