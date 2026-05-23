module NonDiagonalMass

using SparseArrays: SparseMatrixCSC
using SummationByPartsOperators: SafeMode

using ..SphericalSBPOperators:
                               NonDiagonalMassSphericalOperators,
                               _boundary_closure_width_from_operator,
                               _closure_diagnostics,
                               _default_scale_eltype,
                               _resolve_atol,
                               _scale_sparse_matrix,
                               _uniform_spacing,
                               sbp4_solve_accuracy_constraints,
                               sbp6_exp_solve_accuracy_constraints,
                               sbp6_solve_accuracy_constraints

export SphericalOperators
export spherical_operators
export sbp6_exp_spherical_operators
export scale_spherical_operators

const SphericalOperators = NonDiagonalMassSphericalOperators
include("construct.jl")

end
