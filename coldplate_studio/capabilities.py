"""Capability states exposed to UI and reporting (spec section H)."""

from enum import Enum


class CapabilityState(str, Enum):
    PLANNED = "planned"
    IMPLEMENTED_UNVERIFIED = "implemented_unverified"
    NUMERICALLY_VERIFIED = "numerically_verified"
    BENCHMARK_VALIDATED = "benchmark_validated"
    EXPERIMENTALLY_VALIDATED = "experimentally_validated"


# Milestone 1–2 capability matrix (updated as features are verified).
CAPABILITY_MATRIX: dict[str, CapabilityState] = {
    "research_register": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "geometry_validation": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "fluid_domain_extraction": CapabilityState.NUMERICALLY_VERIFIED,
    "connectivity_graph": CapabilityState.NUMERICALLY_VERIFIED,
    "fluid_solid_interface": CapabilityState.NUMERICALLY_VERIFIED,
    "mesh_generation": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "fr_scalar_1d": CapabilityState.NUMERICALLY_VERIFIED,
    "integral_mesh_pipeline": CapabilityState.NUMERICALLY_VERIFIED,
    "fr_scalar_3d_thermal_vof": CapabilityState.NUMERICALLY_VERIFIED,
    "fr_p1_tet_on_integral_mesh": CapabilityState.NUMERICALLY_VERIFIED,
    "postprocess_vtk_contours": CapabilityState.NUMERICALLY_VERIFIED,
    "fluid_solid_interface_maps": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "fluid_volume_connectivity": CapabilityState.NUMERICALLY_VERIFIED,
    "preflight_diagnostics": CapabilityState.NUMERICALLY_VERIFIED,
    "conservation_audit": CapabilityState.NUMERICALLY_VERIFIED,
    "gpu_backend": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "incompressible_flow": CapabilityState.PLANNED,
    "conjugate_heat_transfer": CapabilityState.PLANNED,
    "two_phase_interface": CapabilityState.PLANNED,
    "phase_change_boiling": CapabilityState.PLANNED,
    "adaptive_refinement": CapabilityState.PLANNED,
    "design_optimization": CapabilityState.PLANNED,
}
