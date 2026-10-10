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
| `incompressible_p1_penalty_stokes` | numerically_verified | `test_channel_stokes_mass_conservation` (mesh_size=0.08) | \|Φ_in+Φ_out\|/mean < 35% | recorded: div L2≈1.84e-3, Φ_in≈4.15e-4 m³/s, Φ_out≈−5.06e-4 m³/s, imbalance≈19.7%; penalty β=100, ν=1e-3; not FR high-order |
| `gpu_spmv` | implemented_unverified | `test_gpu_spmv` | CPU≈GPU | skipped without CUDA |
