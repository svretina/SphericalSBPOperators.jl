module SphericalSBPOperators

using LinearAlgebra: dot, eigen, mul!, norm
import LinearAlgebra
using LinearSolve: KLUFactorization
using OrdinaryDiffEqLowOrderRK: RK4
using OrdinaryDiffEqSDIRK: ImplicitMidpoint
using SciMLBase: DiscreteCallback, ODEFunction, ODEProblem, remake, solve
using SparseArrays: SparseMatrixCSC, dropzeros!, findnz, sparse, spdiagm, spzeros
using SummationByPartsOperators:
    FastMode,
    SafeMode,
    MattssonNordström2004,
    derivative_operator,
    grid,
    mass_matrix

export SphericalOperators
export spherical_operators
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
export sbp8_v_offdiag_pairs, sbp8_vector_mass, sbp8_scalar_mass_gradient
export sbp8_construct_divergence, sbp8_solve_accuracy_constraints
export sbp8_operators
export default_wave_profile
export bumpb, bumpb_profile
export characteristic_initial_data
export apply_symmetry_state!, apply_symmetry_rhs!, initialize_wave_state!
export apply_characteristic_bc_sat!, boundary_characteristics, boundary_characteristic_residual
export apply_symmetry_constraints!, apply_boundary_conditions!, apply_wave_constraints!
export check_wave_data_consistency
export check_potential_consistency
export wave_rhs!, wave_system_ode!, wave_energy
export wave_system_ode_vec!, wave_system_jac!, wave_system_jac_prototype
export ImplicitMidpoint
export compute_stability_limit, estimate_max_timestep, estimate_wave_timestep
export solve_wave_ode, solve_wave, WaveEvolutionResult
export benchmark_wave_integrators
export diagnose_reflecting_energy_bump
export diagnose_reflecting_sat_energy_drift
export energy_rate
export WaveODEParams

include("diagonal_mass/types.jl")
include("diagonal_mass/snap.jl")
include("diagonal_mass/fullgrid.jl")
include("diagonal_mass/folding.jl")
include("non_diagonal_mass/sbp4.jl")
include("non_diagonal_mass/sbp6.jl")
include("non_diagonal_mass/sbp8.jl")
include("diagonal_mass/construct.jl")
include("diagonal_mass/validation.jl")
include("diagonal_mass/diagnostics.jl")
include("diagonal_mass/wave_physics.jl")
include("diagonal_mass/wave_solver.jl")

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
