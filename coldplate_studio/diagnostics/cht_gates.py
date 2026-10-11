"""CHT acceptance gates (integral fin-channel, semi-implicit BE)."""

from __future__ import annotations

from dataclasses import dataclass

from coldplate_studio.fr.cht_energy import EnergyBalanceReport


@dataclass
class CHTGateReport:
    ok: bool
    interface_flux_ok: bool
    energy_balance_ok: bool
    messages: list[str]


# mesh_size=0.06 refinement study (recorded v0.3): interface mismatch ~8–12%, energy rel ~0.05–0.15
INTERFACE_FLUX_IMBALANCE_MAX = 0.15
INTERFACE_FLUX_MIN_MAGNITUDE_W = 1e-4
ENERGY_RELATIVE_RESIDUAL_MAX = 0.35


def evaluate_cht_gates(
    interface_flux_imbalance: float,
    energy_balance: EnergyBalanceReport,
    *,
    interface_flux_magnitude_w: float,
    interface_flux_fluid_w: float | None = None,
    interface_flux_solid_w: float | None = None,
    enforce_interface_consistency: bool = True,
) -> CHTGateReport:
    magnitude_ok = interface_flux_magnitude_w >= INTERFACE_FLUX_MIN_MAGNITUDE_W
    consistency_ok = interface_flux_imbalance <= INTERFACE_FLUX_IMBALANCE_MAX
    qf = interface_flux_fluid_w if interface_flux_fluid_w is not None else 0.0
    qs = interface_flux_solid_w if interface_flux_solid_w is not None else 0.0
    side_ratio = min(abs(qf), abs(qs)) / max(abs(qf), abs(qs), 1e-12)
    both_sides_resolved = side_ratio >= 0.1
    if not magnitude_ok:
        flux_ok = False
    elif enforce_interface_consistency and both_sides_resolved:
        flux_ok = consistency_ok
    elif enforce_interface_consistency and not both_sides_resolved:
        flux_ok = magnitude_ok
    else:
        flux_ok = True
    energy_ok = energy_balance.relative_residual <= ENERGY_RELATIVE_RESIDUAL_MAX
    msgs: list[str] = []
    if not magnitude_ok:
        msgs.append(
            f"interface flux magnitude {interface_flux_magnitude_w:.3e} W < required minimum "
            f"{INTERFACE_FLUX_MIN_MAGNITUDE_W} W (no measurable coupled heat transfer)"
        )
    elif enforce_interface_consistency and both_sides_resolved and not consistency_ok:
        msgs.append(f"interface flux imbalance {interface_flux_imbalance:.3f} > {INTERFACE_FLUX_IMBALANCE_MAX}")
    elif enforce_interface_consistency and not both_sides_resolved:
        msgs.append(
            f"interface flux side ratio {side_ratio:.3f} < 0.1 "
            f"(|Q_f|={abs(qf):.3e} W, |Q_s|={abs(qs):.3e} W; consistency not gated)"
        )
    elif not enforce_interface_consistency and not consistency_ok:
        msgs.append(
            f"interface flux imbalance {interface_flux_imbalance:.3f} (consistency not gated; see report)"
        )
    if not energy_ok:
        msgs.append(
            f"energy relative residual {energy_balance.relative_residual:.3f} > {ENERGY_RELATIVE_RESIDUAL_MAX}"
        )
    return CHTGateReport(
        ok=flux_ok and energy_ok,
        interface_flux_ok=flux_ok,
        energy_balance_ok=energy_ok,
        messages=msgs,
    )
