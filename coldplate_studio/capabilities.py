"""Capability states exposed to UI and reporting (spec section H)."""

from enum import Enum


class CapabilityState(str, Enum):
    PLANNED = "planned"
    IMPLEMENTED_UNVERIFIED = "implemented_unverified"
    NUMERICALLY_VERIFIED = "numerically_verified"
    BENCHMARK_VALIDATED = "benchmark_validated"
    EXPERIMENTALLY_VALIDATED = "experimentally_validated"


# See docs/engineering/VERIFICATION_EVIDENCE.md for benchmarks — not every pytest implies FR high-order.
CAPABILITY_MATRIX: dict[str, CapabilityState] = {
    "research_register": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "geometry_validation": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "fluid_domain_extraction": CapabilityState.NUMERICALLY_VERIFIED,
    "connectivity_graph": CapabilityState.NUMERICALLY_VERIFIED,
    "fluid_solid_interface": CapabilityState.NUMERICALLY_VERIFIED,
    "fsi_interface_geometry": CapabilityState.NUMERICALLY_VERIFIED,
    "user_confirmed_ports": CapabilityState.NUMERICALLY_VERIFIED,
    "mesh_generation": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "mesh_validity_checks": CapabilityState.NUMERICALLY_VERIFIED,
    "fr_scalar_1d_huynh": CapabilityState.NUMERICALLY_VERIFIED,
    "structured_hex_fr_operators": CapabilityState.NUMERICALLY_VERIFIED,
    "integral_mesh_pipeline": CapabilityState.NUMERICALLY_VERIFIED,
    "p1_scalar_thermal_transport": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "postprocess_vtk_contours": CapabilityState.NUMERICALLY_VERIFIED,
    "fluid_solid_interface_maps": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "fluid_volume_connectivity": CapabilityState.NUMERICALLY_VERIFIED,
    "preflight_diagnostics": CapabilityState.NUMERICALLY_VERIFIED,
    "conservation_audit": CapabilityState.NUMERICALLY_VERIFIED,
    "incompressible_p1_penalty_stokes": CapabilityState.NUMERICALLY_VERIFIED,
    "gpu_backend_probe": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "gpu_solver_kernels": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "fr_high_order_3d_production": CapabilityState.PLANNED,
    "incompressible_fr_transient": CapabilityState.PLANNED,
    "conjugate_heat_transfer": CapabilityState.PLANNED,
    "two_phase_interface": CapabilityState.PLANNED,
    "phase_change_boiling": CapabilityState.PLANNED,
    "adaptive_refinement": CapabilityState.PLANNED,
    "design_optimization": CapabilityState.PLANNED,
}
