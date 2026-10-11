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
    enforce_interface_consistency: bool = True,
) -> CHTGateReport:
    magnitude_ok = interface_flux_magnitude_w >= INTERFACE_FLUX_MIN_MAGNITUDE_W
    consistency_ok = interface_flux_imbalance <= INTERFACE_FLUX_IMBALANCE_MAX
    if not magnitude_ok:
        flux_ok = False
    elif enforce_interface_consistency:
        flux_ok = consistency_ok
    else:
        flux_ok = True
    energy_ok = energy_balance.relative_residual <= ENERGY_RELATIVE_RESIDUAL_MAX
    msgs: list[str] = []
    if not magnitude_ok:
        msgs.append(
            f"interface flux magnitude {interface_flux_magnitude_w:.3e} W < required minimum "
            f"{INTERFACE_FLUX_MIN_MAGNITUDE_W} W (no measurable coupled heat transfer)"
        )
    elif enforce_interface_consistency and not consistency_ok:
        msgs.append(f"interface flux imbalance {interface_flux_imbalance:.3f} > {INTERFACE_FLUX_IMBALANCE_MAX}")
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
