"""Conservation checks and pre-solve gates (spec E4, G)."""

from __future__ import annotations

from dataclasses import dataclass, field

from coldplate_studio.fr.solver3d import ConservationReport


@dataclass
class ConservationThresholds:
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    max_relative_liquid_mass_loss_per_step: float = 0.5
    warn_relative_thermal_energy_drop_per_step: float = 0.95


@dataclass
class ConservationCheck:
    code: str
    message: str
    severity: str  # error | warning


@dataclass
class ConservationAudit:
    ok: bool
    checks: list[ConservationCheck] = field(default_factory=list)

    def add(self, code: str, message: str, severity: str) -> None:
        self.checks.append(ConservationCheck(code, message, severity))
        if severity == "error":
            self.ok = False


def audit_conservation_report(
    report: ConservationReport,
    thresholds: ConservationThresholds | None = None,
) -> ConservationAudit:
    th = thresholds or ConservationThresholds()
    audit = ConservationAudit(ok=True)
    if report.alpha_min < th.alpha_min - 1e-12:
        audit.add("alpha_below_zero", f"alpha_min={report.alpha_min} < 0", "error")
    if report.alpha_max > th.alpha_max + 1e-12:
        audit.add("alpha_above_one", f"alpha_max={report.alpha_max} > 1", "error")
    return audit


def audit_step_mass_change(
    before: ConservationReport,
    after: ConservationReport,
    thresholds: ConservationThresholds | None = None,
) -> ConservationAudit:
    th = thresholds or ConservationThresholds()
    audit = ConservationAudit(ok=True)
    if before.liquid_mass_kg <= 0:
        return audit
    rel = (before.liquid_mass_kg - after.liquid_mass_kg) / before.liquid_mass_kg
    if rel > th.max_relative_liquid_mass_loss_per_step:
        audit.add(
            "mass_loss_spike",
            f"relative liquid mass loss {rel:.3f} exceeds {th.max_relative_liquid_mass_loss_per_step}",
            "warning",
        )
    return audit
