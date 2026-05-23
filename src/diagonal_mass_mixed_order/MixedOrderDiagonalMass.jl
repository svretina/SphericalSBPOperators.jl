module MixedOrderDiagonalMass

using SparseArrays: sparse, spdiagm, spzeros
using SummationByPartsOperators: SafeMode

using ..SphericalSBPOperators:
                               DiagonalMassSphericalOperators,
                               _assert_global_even_interior_accuracy,
                               _boundary_closure_width_from_operator,
                               _build_folding_operators,
                               _build_full_grid_objects,
                               _closure_diagnostics,
                               _default_scale_eltype,
                               _extract_diagonal,
                               _repair_even_gradient_first_column!,
                               _resolve_atol,
                               _rows_with_nonzero_first_column,
                               _scale_sparse_matrix,
                               _set_divergence_rows!,
                               _set_origin_row!,
                               _uniform_spacing,
                               apply_divergence,
                               apply_even_gradient,
                               apply_odd_derivative,
                               check_odd,
                               diagnose,
                               enforce_odd!,
                               interpret_diagnostics,
                               scale_spherical_operators as _base_scale_spherical_operators,
                               snap_sparse!,
                               validate

export SphericalOperators
export spherical_operators
export scale_spherical_operators
export validate
export diagnose, interpret_diagnostics
export apply_even_gradient, apply_odd_derivative, apply_divergence
export enforce_odd!, check_odd

const SphericalOperators = DiagonalMassSphericalOperators

include("construct.jl")

end
