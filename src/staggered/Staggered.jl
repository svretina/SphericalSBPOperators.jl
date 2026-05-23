module Staggered

using LinearAlgebra: dot
using SparseArrays: SparseMatrixCSC, dropzeros!, findnz, sparse, spdiagm, spzeros
using SummationByPartsOperators: FastMode, derivative_operator, grid, mass_matrix

using ..SphericalSBPOperators: StaggeredSphericalOperators, snap_sparse!

export SphericalOperators
export spherical_operators
export scale_spherical_operators
export validate
export diagnose, interpret_diagnostics
export apply_even_gradient, apply_odd_derivative, apply_divergence
export enforce_odd!, check_odd

const SphericalOperators = StaggeredSphericalOperators
include("snap.jl")
include("fullgrid.jl")
include("folding.jl")
include("construct.jl")
include("validation.jl")
include("diagnostics.jl")

end
