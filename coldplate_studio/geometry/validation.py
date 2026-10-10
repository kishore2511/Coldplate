"""Geometry validation prior to Boolean fluid extraction (spec B1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GeometryIssue:
    code: str
    message: str
    severity: str  # error | warning
    entity: str | None = None


@dataclass
class GeometryValidationReport:
    ok: bool
    issues: list[GeometryIssue] = field(default_factory=list)

    def add(self, code: str, message: str, severity: str, entity: str | None = None) -> None:
        self.issues.append(GeometryIssue(code, message, severity, entity))
        if severity == "error":
            self.ok = False


def validate_step_path(path: Path) -> GeometryValidationReport:
    report = GeometryValidationReport(ok=True)
    if not path.exists():
        report.add("missing_file", f"STEP file not found: {path}", "error")
        return report
    if path.suffix.lower() not in {".step", ".stp"}:
        report.add("unsupported_format", f"Expected STEP extension, got {path.suffix}", "warning")
    return report


def validate_gmsh_model_stats(stats: dict[str, Any], min_feature_length: float) -> GeometryValidationReport:
    """Validate gmsh model statistics after import/Boolean."""
    report = GeometryValidationReport(ok=True)
    n_vol = int(stats.get("volumes", 0))
    n_surf = int(stats.get("surfaces", 0))
    if n_vol == 0:
        report.add("no_volumes", "No volume entities found after geometry operations.", "error")
    if n_surf == 0:
        report.add("no_surfaces", "No surface entities found.", "error")
    bbox = stats.get("bbox_diagonal", None)
    if bbox is not None and bbox < min_feature_length:
        report.add(
            "tiny_bbox",
            f"Bounding-box diagonal {bbox:.3e} is below min_feature_length {min_feature_length:.3e}.",
            "warning",
        )
    return report
