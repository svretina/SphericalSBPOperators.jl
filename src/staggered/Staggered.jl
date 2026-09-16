module Staggered

using LinearAlgebra: dot
using SparseArrays: SparseMatrixCSC, dropzeros!, findnz, sparse, spdiagm, spzeros
using SummationByPartsOperators: FastMode, derivative_operator, grid, mass_matrix

using ..SphericalSBPOperators: StaggeredSphericalOperators,
                               snap_sparse!,
                               _default_atol,
                               _resolve_atol,
                               _maxabs,
                               _maxabs_sparse,
                               _build_full_grid_objects,
                               _build_folding_operators,
                               _boundary_closure_width_from_operator

export SphericalOperators
export spherical_operators
export scale_spherical_operators
export validate
export diagnose, interpret_diagnostics
export apply_even_gradient, apply_odd_derivative, apply_divergence
export enforce_odd!, check_odd

const SphericalOperators = StaggeredSphericalOperators
include("construct.jl")
include("validation.jl")
include("diagnostics.jl")

end
