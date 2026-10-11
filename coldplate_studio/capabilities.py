"""Production capability status (see docs/engineering/COMPLETION_CONTRACT.md)."""

from enum import Enum


class ProductionStatus(str, Enum):
    COMPLETE = "complete"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    PLANNED = "planned"


# Legacy UI states (deprecated for roadmap; kept for older reports).
class CapabilityState(str, Enum):
    PLANNED = "planned"
    IMPLEMENTED_UNVERIFIED = "implemented_unverified"
    NUMERICALLY_VERIFIED = "numerically_verified"
    BENCHMARK_VALIDATED = "benchmark_validated"
    EXPERIMENTALLY_VALIDATED = "experimentally_validated"


PRODUCTION_MATRIX: dict[str, ProductionStatus] = {
    "integral_fin_channel_p2p1_flow": ProductionStatus.IN_PROGRESS,
    "conjugate_heat_transfer_p2_advection": ProductionStatus.IN_PROGRESS,
    "fin_channel_single_phase_e2e": ProductionStatus.IN_PROGRESS,
    "steady_stokes_p2p1_box_mms": ProductionStatus.IN_PROGRESS,
    "incompressible_p1_pspg_mixed_stokes": ProductionStatus.IN_PROGRESS,
    "incompressible_p1_penalty_stokes": ProductionStatus.IN_PROGRESS,
    "fr_scalar_1d_huynh": ProductionStatus.IN_PROGRESS,
    "structured_hex_fr_operators": ProductionStatus.IN_PROGRESS,
    "fr_high_order_3d_production": ProductionStatus.PLANNED,
    "gpu_solver_kernels": ProductionStatus.PLANNED,
    "two_phase_boiling": ProductionStatus.PLANNED,
    "design_optimization": ProductionStatus.PLANNED,
    "integrated_gui": ProductionStatus.PLANNED,
}

# See docs/engineering/VERIFICATION_MATRIX.md — not every pytest implies production COMPLETE.
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
    "structured_hex_fr_operators": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "integral_mesh_pipeline": CapabilityState.NUMERICALLY_VERIFIED,
    "p1_scalar_thermal_transport": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "postprocess_vtk_contours": CapabilityState.NUMERICALLY_VERIFIED,
    "fluid_solid_interface_maps": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "fluid_volume_connectivity": CapabilityState.NUMERICALLY_VERIFIED,
    "preflight_diagnostics": CapabilityState.NUMERICALLY_VERIFIED,
    "conservation_audit": CapabilityState.NUMERICALLY_VERIFIED,
    "incompressible_p1_penalty_stokes": CapabilityState.NUMERICALLY_VERIFIED,
    "incompressible_p1_pspg_mixed_stokes": CapabilityState.NUMERICALLY_VERIFIED,
    "integral_fin_channel_pspg_flow": CapabilityState.NUMERICALLY_VERIFIED,
    "integral_fin_channel_p2p1_taylor_hood_flow": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "p2_velocity_thermal_advection": CapabilityState.NUMERICALLY_VERIFIED,
    "fin_channel_single_phase_e2e": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "incompressible_p1_transient_projection": CapabilityState.NUMERICALLY_VERIFIED,
    "flow_mass_continuity_diagnostics": CapabilityState.NUMERICALLY_VERIFIED,
    "steady_stokes_manufactured_solution": CapabilityState.IMPLEMENTED_UNVERIFIED,
    "developed_periodic_channel_flow": CapabilityState.NUMERICALLY_VERIFIED,
    "conjugate_heat_transfer_p1": CapabilityState.NUMERICALLY_VERIFIED,
    "thermal_conduction_p1": CapabilityState.NUMERICALLY_VERIFIED,
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
