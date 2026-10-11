# Verification Evidence Log

| Capability key | State | Benchmark / test | Expected | Recorded error / note |
|----------------|-------|------------------|----------|------------------------|
| `fr_scalar_1d_huynh` | numerically_verified | `test_polynomial_advection_exact_steady_residual` | cubic exact RHS | rtol 1e-9 |
| `fr_scalar_1d_huynh` | numerically_verified | `test_advection_convergence_order` | order > 2 | mesh refinement |
| `structured_hex_fr_operators` | numerically_verified | `test_fr_3d` | bounded α, diffusion peak decay | not production mesh |
| `p1_scalar_thermal_transport` | implemented_unverified | channel e2e | VTK + contours | no analytical T benchmark |
| `integral_mesh_pipeline` | numerically_verified | `test_integral_preflight_passes` | preflight ok | procedural channel only |
| `fluid_volume_connectivity` | numerically_verified | `test_channel_has_single_connected_fluid_component` | 1 component | |
| `fsi_interface_geometry` | numerically_verified | `test_interface_geometry` | area > 0, consistent normals | |
| `incompressible_p1_penalty_stokes` | numerically_verified | `test_channel_stokes_mass_conservation` (mesh_size=0.08) | \|Φ_in+Φ_out\|/mean < 35% | v0.3.1 aligned ports: imbalance≈5%; tol-only ports≈19.7% (historical); β=100, ν=1e-3 |
| `gpu_spmv` | implemented_unverified | `test_gpu_spmv` | CPU≈GPU | skipped without CUDA |
| `flow_mass_continuity_diagnostics` | numerically_verified | `test_channel_mass_continuity_diagnosis_reproduces_baseline` | imbalance 2–12% | aligned port faces; see FIN_CHANNEL_V031.md |
| `steady_stokes_manufactured_solution` | numerically_verified | MMS h=0.06→0.032 | ux L2 decreases | h=0.06: L2≈2.37e-2, L∞≈9.96e-2; h=0.032: L2≈2.28e-2; β=0 viscous solve |
| `incompressible_p1_transient_projection` | numerically_verified | `test_flow_transient.py` | div̄_after ≤ div̄_before, Φ_in>0, Φ_out<0 | fractional-step P1; not high-order FR |
| `quasi_2d_poiseuille_duct` | **invalid benchmark** | `test_quasi_2d_poiseuille_not_validated_hagen_poiseuille` | must fail vs theory | \|Q_err\|>90%; not Hagen–Poiseuille |
| `developed_periodic_channel_flow` | numerically_verified | `test_developed_parabolic_profile_structured_duct` | ux L2 vs parabolic; Q rel err < 6% | nx=8–16: L2 **~1e-9**; Q via trapezoid yz-plane; penalty+periodic γ |
| `thermal_conduction_p1` | numerically_verified | `test_1d_steady_conduction_linear_profile` | linear T | L2 err <0.5 K (Dirichlet on all faces) |
| `thermal_conduction_p1` | numerically_verified | `test_transient_diffusion_fourier_mode_decay` | Fourier decay | L2 err <0.08 after 5 steps |
| `incompressible_p1_pspg_mixed_stokes` | numerically_verified | `test_pspg_periodic_channel_profile_and_divergence` | ux err < 5e-3, div L2 < 0.05 | τ=h²/(4ν); not Taylor–Hood |
| `incompressible_p1_pspg_mixed_stokes` | numerically_verified | `test_pspg_manufactured_rhs_residual` | MMS ux L2 < 0.05 | linear solve residual < 1e-5 |
| `conjugate_heat_transfer_p1` | numerically_verified | `test_two_material_steady_conduction_interface` | \|T−T_analytical\| < 1 K | Dirichlet elimination on x faces |
| `conjugate_heat_transfer_p1` | numerically_verified | `test_integral_channel_interface_flux_opposite_signs` | \|Q_f+Q_s\|/max < 35% | coarse integral mesh |
| `conjugate_heat_transfer_p1` | numerically_verified | `test_cht_transient_energy_balance_relative_residual` | rel residual < 2 | semi-implicit BE; `cht_energy.py` |
| `conjugate_heat_transfer_p1` | numerically_verified | `test_cht_integral_channel_energy_increases_with_hot_inlet` | E_total increases | uses solved **u** from `run_channel_flow` |
| `integral_fin_channel_pspg_flow` | numerically_verified | `test_fin_channel_pspg_matches_penalty_mass_flux_at_reference_mesh` | PSPG≈penalty Φ at h=0.08 | β=100 open duct |
| `integral_fin_channel_pspg_flow` | numerically_verified | `test_fin_channel_pspg_refinement_table_recorded` | 0.08 imb < 12% | three-level table; see FIN_CHANNEL_V031.md |
| `integral_fin_channel_pspg_flow` | numerically_verified | `test_fin_channel_flow_refinement_gates_documented` | 0.08 gates pass; 0.12/0.06 fail | fluid-only pressure; see FIN_CHANNEL_V041.md |
| `conjugate_heat_transfer_p1` | numerically_verified | `test_fin_channel_conduction_interface_flux_and_energy_budget` | side ratio > 0.15, energy rel < 0.35 | u=0 conduction path |
| `integral_fin_channel_p2p1_taylor_hood_flow` | numerically_verified | `test_p2p1_fin_channel_mass_balance_reference_mesh` | solver residual, mass imb | Taylor–Hood production |
| `integral_fin_channel_p2p1_taylor_hood_flow` | numerically_verified | `test_fin_channel_flow_refinement_gates_documented` | three-level gates pass | P2–P1; FIN_CHANNEL_V05.md |
| `conjugate_heat_transfer_p1` | numerically_verified | `test_galerkin_interface_flux_balanced_on_analytical_bar` | Galerkin imb < 5% | weak-form interface exchange |
| `integral_fin_channel_pspg_flow` | numerically_verified | `test_flow_port_faces_have_streamwise_outward_normals` | \|n_x\|>0.5 on ports | normal-aligned selection |
| `fin_channel_single_phase_e2e` | numerically_verified | `test_fin_channel_e2e_flow_gates_pass_reference_mesh` | flow+CHT gates | mesh_size=0.08; volumetric solid heat demo |
| `conjugate_heat_transfer_p1` | numerically_verified | `test_fin_channel_cht_nonzero_interface_flux_with_volumetric_source` | \|Q_s\|>1e-4 W | port inlet BC; energy residual < 0.35 |
| `conjugate_heat_transfer_p1` | numerically_verified | `test_two_material_fsi_conduction_interface_flux_matches_series_solution` | imb < 12%, Q within 8% | FSI tet bar; see FIN_CHANNEL_V04.md |
| `conjugate_heat_transfer_p1` | numerically_verified | `test_integral_channel_interface_flux_continuity_linear_z` | imbalance &lt; 1e-6 | same k, T∝z |
