module DiagonalExp

using SummationByPartsOperators: SafeMode

using ..SphericalSBPOperators:
                               DiagonalMassSphericalOperators,
                               scale_spherical_operators as _base_scale_spherical_operators,
                               spherical_operators as _base_spherical_operators,
                               apply_divergence,
                               apply_even_gradient,
                               apply_odd_derivative,
                               check_odd,
                               diagnose,
                               enforce_odd!,
                               interpret_diagnostics,
                               validate

export SphericalOperators
export spherical_operators
export scale_spherical_operators
export validate
export diagnose, interpret_diagnostics
export apply_even_gradient, apply_odd_derivative, apply_divergence
export enforce_odd!, check_odd

const SphericalOperators = DiagonalMassSphericalOperators

"""
    spherical_operators(source; kwargs...)

Experimental diagonal-mass construction that extends the exact coupled near-origin
repair by two additional rows beyond the folded origin-coupled rows. This keeps the
standard diagonal path intact while allowing experimentation with a wider repaired
boundary block.
"""
function spherical_operators(source;
                             accuracy_order,
                             N,
                             R,
                             p::Int = 2,
                             mode = SafeMode(),
                             atol = nothing,
                             snap_factor::Float64 = 64.0,
                             custom_stencil_cols::Union{Nothing, Vector{Int}} = nothing,
                             return_repair_info::Bool = false,
                             return_canonical::Bool = false,
                             target_eltype::Union{Nothing, Type} = nothing,
                             mass_solver::Symbol = :seed,
                             mass_solver_opts::NamedTuple = (;),
                             seed_banded::Bool = true,
                             seed_band_scale::Real = 1 // 10^12)
    return _base_spherical_operators(source;
                                     accuracy_order = accuracy_order,
                                     N = N,
                                     R = R,
                                     p = p,
                                     mode = mode,
                                     atol = atol,
                                     snap_factor = snap_factor,
                                     custom_stencil_cols = custom_stencil_cols,
                                     return_repair_info = return_repair_info,
                                     return_canonical = return_canonical,
                                     target_eltype = target_eltype,
                                     mass_solver = mass_solver,
                                     mass_solver_opts = mass_solver_opts,
                                     seed_banded = seed_banded,
                                     seed_band_scale = seed_band_scale,
                                     additional_repair_rows = 2,
                                     optimize_downstream_divergence = true)
end

@inline function scale_spherical_operators(ops::SphericalOperators,
                                           R;
                                           target_eltype::Union{Nothing, Type} = nothing,
                                           atol = nothing)
    return _base_scale_spherical_operators(ops, R;
                                           target_eltype = target_eltype,
                                           atol = atol)
end

end
