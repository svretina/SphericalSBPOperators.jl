module SphericalSBPOperators

using LinearAlgebra: dot, eigen, mul!, norm
using LinearSolve: KLUFactorization
using OrdinaryDiffEqLowOrderRK: RK4
using OrdinaryDiffEqSDIRK: ImplicitMidpoint
using SciMLBase: DiscreteCallback, ODEFunction, ODEProblem, remake, solve
using SparseArrays: SparseMatrixCSC, dropzeros!, findnz, sparse, spdiagm, spzeros
using SummationByPartsOperators:
    FastMode,
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

include("types.jl")
include("snap.jl")
include("fullgrid.jl")
include("folding.jl")
include("construct.jl")
include("validation.jl")
include("diagnostics.jl")
include("wave_physics.jl")
include("wave_solver.jl")

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

The metric-weighted SBP mass is
```math
H = H_{\\mathrm{cart,half}}\\,\\mathrm{diag}(r^p),
```
and the discrete SBP relation is
```math
H D + G^T H = B, \\quad B = \\mathrm{diag}(0,\\dots,0,R^p).
```

For `p > 0`, `H[1,1] = 0` at `r = 0`, so SBP does not constrain the origin row of
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
