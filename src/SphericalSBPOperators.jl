module SphericalSBPOperators

using LinearAlgebra: dot, eigen, mul!, norm
import LinearAlgebra
using GenericSchur
using CairoMakie
using LaTeXStrings
using LinearSolve: KLUFactorization
using MultiFloats: Float64x4
using OrdinaryDiffEqHighOrderRK: TsitPap8
using OrdinaryDiffEqLowOrderRK: RK4
using OrdinaryDiffEqSDIRK: ImplicitMidpoint
using Printf: @sprintf
using SciMLBase: DiscreteCallback, ODEFunction, ODEProblem, init, remake, solve, step!,
                 successful_retcode
using SpecialFunctions: erf
using SparseArrays: SparseMatrixCSC, dropzeros!, findnz, sparse, spdiagm, spzeros
using SummationByPartsOperators:
                                 SafeMode,
                                 FastMode,
                                 MattssonNordström2004,
                                 derivative_operator,
                                 grid,
                                 mass_matrix
using JLD2

export SphericalOperators
export Staggered
export NonDiagonalMass
export MixedOrderDiagonalMass
export DiagonalExp
export staggered_spherical_operators
export validate_staggered, diagnose_staggered, interpret_diagnostics_staggered
export non_diagonal_spherical_operators
export non_diagonal_exp_spherical_operators
export diagonal_spherical_operators
export mixed_order_diagonal_spherical_operators
export diagonal_exp_spherical_operators
export scale_spherical_operators
export validate
export diagnose, interpret_diagnostics
export apply_even_gradient, apply_odd_derivative, apply_divergence
export enforce_odd!, check_odd
export compute_even_gradient_row_decoupled, verify_even_gradient_row_decoupled
export snap_sparse!
export sbp4_v_offdiag_pairs, sbp4_vector_mass, sbp4_scalar_mass_gradient
export sbp4_construct_divergence, sbp4_solve_accuracy_constraints
export sbp4_operators
export sbp6_v_offdiag_pairs, sbp6_vector_mass, sbp6_scalar_mass_gradient
export sbp6_construct_divergence, sbp6_solve_accuracy_constraints
export sbp6_operators
export sbp6_exp_v_offdiag_pairs, sbp6_exp_vector_mass, sbp6_exp_scalar_mass_gradient
export sbp6_exp_construct_divergence, sbp6_exp_solve_accuracy_constraints
export sbp6_exp_operators
export sbp8_v_offdiag_pairs, sbp8_vector_mass, sbp8_scalar_mass_gradient
export sbp8_construct_divergence, sbp8_solve_accuracy_constraints
export sbp8_operators
export default_wave_profile
export bumpb, bumpb_profile
export characteristic_initial_data
export even_gaussian_pi_zero_psi_profile, even_gaussian_pi_zero_psi_solution
export regular_left_moving_spherical_gaussian_profile
export regular_left_moving_spherical_gaussian_profile_prime
export regular_left_moving_spherical_gaussian_profile_second
export regular_left_moving_spherical_gaussian_phi_exact
export regular_left_moving_spherical_gaussian_pi_exact
export regular_left_moving_spherical_gaussian_psi_exact
export regular_left_moving_spherical_gaussian_solution
export analytic_wave_solution
export apply_symmetry_state!, apply_symmetry_rhs!, initialize_wave_state!
export apply_characteristic_bc_sat!, boundary_characteristics,
       boundary_characteristic_residual
export apply_symmetry_constraints!, apply_boundary_conditions!, apply_wave_constraints!
export check_wave_data_consistency
export check_potential_consistency
export wave_rhs!, wave_system_ode!, wave_energy
export wave_system_ode_vec!, wave_system_jac!, wave_system_jac_prototype
export wave_D_kernel, wave_Geven_kernel, wave_kernel_operators, WaveMatrixKernel
export wave_kernel_mul!, wave_D_mul!, wave_Geven_mul!
export ImplicitMidpoint, TsitPap8
export compute_stability_limit, estimate_max_timestep, estimate_wave_timestep
export solve_wave_ode, solve_wave, WaveEvolutionResult
export benchmark_wave_integrators
export diagnose_reflecting_energy_bump
export diagnose_reflecting_sat_energy_drift
export energy_rate
export WaveODEParams
export save_spectrum_plot, save_nullspace_plot
export save_reflection_heatmap, save_energy_trace
export save_dashboard, save_reflection_animation
export mytheme_aps, mytheme_aps_publication, mytheme_aps_spectrum
export gundlach_error, gundlach_error_norm
export plot_convergence_order, plot_wave_convergence_orders
export interactive_pointwise_wave_convergence_orders
export interactive_pointwise_wave_convergence_orders_gundlach
export interactive_pointwise_wave_scaled_errors
export interactive_pointwise_wave_scaled_errors_analytic
export interactive_pointwise_wave_scaled_errors_gundlach
export interactive_pointwise_wave_scaled_errors_gundlach_reference
export plot_pointwise_wave_scaled_errors_gundlach
export plot_wave_energy_history, plot_wave_energy_histories
export plot_divergence_comparison
export plot_wave_snapshot, plot_wave_snapshot_resolutions
export laplacian_matrix, laplacian_spectrum, save_laplacian_spectrum_plot
export save_laplacian_spectrum_sources_plot

include("operators/types.jl")
const SphericalOperators = DiagonalMassSphericalOperators
include("diagonal_mass/snap.jl")
include("diagonal_mass/fullgrid.jl")
include("diagonal_mass/folding.jl")
include("spectrum_precision.jl")
include("non_diagonal_mass/sbp4.jl")
include("non_diagonal_mass/sbp6.jl")
include("non_diagonal_mass/sbp6_exp.jl")
include("non_diagonal_mass/sbp8.jl")
include("diagonal_mass/construct.jl")
include("diagonal_mass/validation.jl")
include("diagonal_mass/diagnostics.jl")
include("diagonal_mass_mixed_order/MixedOrderDiagonalMass.jl")
include("diagonal_exp/DiagonalExp.jl")
include("non_diagonal_mass/NonDiagonalMass.jl")
include("staggered/Staggered.jl")
include("wave/Wave.jl")
include("convergence.jl")
include("plots/theme.jl")
include("plots/stability_plots.jl")
include("plots/convergence.jl")
include("plots/energy.jl")
include("plots/comparison.jl")
include("plots/snapshots.jl")
include("plots/spectrum_plots.jl")
include("operators/api.jl")

"""
    SphericalSBPOperators

Build spherical-symmetry SBP operators on `[0, R]` by folding Cartesian SBP operators
constructed on the mirrored grid `[-R, R]`.

Parity conventions at the origin:
- scalar-like fields are even under reflection;
- radial flux-like fields are odd under reflection, so `u(0) = 0`.

Accordingly:
- `Geven` maps even fields to odd derivatives;
- `D` maps odd radial fluxes to even divergence values.

The metric-weighted SBP masses are
```math
S = H_{\\mathrm{cart,half}}\\,\\mathrm{diag}(r^p), \\qquad
V = H_{\\mathrm{cart,half}}\\,\\mathrm{diag}(r^p),
```
and the discrete SBP relation is
```math
S D + G^T V = B, \\quad B = \\mathrm{diag}(0,\\dots,0,R^p).
```

For `p > 0`, `S[1,1] = 0` (and likewise `V[1,1] = 0`) at `r = 0`, so SBP does not constrain the origin row of
`D`. This row is fixed using the removable-singularity condition for odd fluxes:
```math
(Du)(0) = (p+1)u'(0),
```
implemented as
```math
D[1,:] = (p+1)\\,G_{\\mathrm{odd}}[1,:].
```

Rational arithmetic is supported through type inference from inputs (e.g. `R = 1//1`),
matching `SummationByPartsOperators`.
"""
SphericalSBPOperators

end
