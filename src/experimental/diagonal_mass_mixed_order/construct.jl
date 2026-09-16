"""
    scale_spherical_operators(ops, R; target_eltype=nothing, atol=nothing)

Delegate to the standard diagonal-mass scaling path. The mixed-order variant
shares the same operator storage type and scaling rules.
"""
@inline function scale_spherical_operators(ops::SphericalOperators,
                                           R;
                                           target_eltype::Union{Nothing, Type} = nothing,
                                           atol = nothing)
    return _base_scale_spherical_operators(ops, R;
                                           target_eltype = target_eltype,
                                           atol = atol)
end

"""
    spherical_operators(source; accuracy_order, N, R, p=2, mode=SafeMode(),
                        atol=nothing, snap_factor=64.0,
                        custom_stencil_cols=nothing, return_canonical=false,
                        target_eltype=nothing, mass_solver=:seed,
                        mass_solver_opts=(;), seed_banded=true,
                        seed_band_scale=1//10^12)

Construct diagonal-mass folded spherical operators on `[0, R]` using a mixed
Cartesian accuracy pair:

- `Geven` uses the requested Cartesian accuracy order `s`;
- `Godd` uses Cartesian accuracy order `s + 2`.

All mass terms, coupled near-origin `Geven` repairs, and SBP-derived divergence
rows away from the origin follow the standard diagonal-mass construction path.
Only the origin row `D[1, :] = (p + 1) * Godd[1, :]` uses the higher-order odd
derivative operator.
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
                             return_canonical::Bool = false,
                             target_eltype::Union{Nothing, Type} = nothing,
                             mass_solver::Symbol = :seed,
                             mass_solver_opts::NamedTuple = (;),
                             seed_banded::Bool = true,
                             seed_band_scale::Real = 1 // 10^12)
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    Nint = Int(N)
    Nint > 0 || throw(ArgumentError("`N` must be positive."))

    even_accuracy = Int(accuracy_order)
    even_accuracy > 0 || throw(ArgumentError("`accuracy_order` must be positive."))
    odd_accuracy = even_accuracy + 2

    R_canonical = big(Nint) // 1
    Dfull_even, xfull_even, Gfull_even, Hfull_even = _build_full_grid_objects(source;
                                                                               accuracy_order = even_accuracy,
                                                                               N = Nint,
                                                                               R = R_canonical,
                                                                               mode = mode)
    _, xfull_odd, Gfull_odd, _ = _build_full_grid_objects(source;
                                                          accuracy_order = odd_accuracy,
                                                          N = Nint,
                                                          R = R_canonical,
                                                          mode = mode)

    xfull_even == xfull_odd ||
        throw(ArgumentError("Mixed-order diagonal-mass construction requires matching Cartesian grids for orders $even_accuracy and $odd_accuracy."))

    T = eltype(xfull_even)
    atol_construct = _resolve_atol(T, nothing)

    r, Rop, Eeven, Eodd = _build_folding_operators(xfull_even; atol = atol_construct)
    Nh = length(r)

    Geven = sparse(Rop * Gfull_even * Eeven)
    Godd = sparse(Rop * Gfull_odd * Eodd)
    snap_sparse!(Geven; snap_factor = snap_factor)
    snap_sparse!(Godd; snap_factor = snap_factor)

    half_factor = convert(T, 1) / convert(T, 2)
    Hcart_half = sparse(half_factor * (transpose(Eeven) * Hfull_even * Eeven))
    metric = spdiagm(0 => r .^ p)
    S_seed = sparse(Hcart_half * metric)
    snap_sparse!(S_seed; snap_factor = snap_factor)
    S = S_seed
    V = copy(S_seed)
    closure_from_operator = _boundary_closure_width_from_operator(Dfull_even)

    B = spzeros(T, Nh, Nh)
    B[end, end] = r[end]^p
    snap_sparse!(B; snap_factor = snap_factor)

    Sdiag = _extract_diagonal(S; atol = atol_construct)
    Vdiag = _extract_diagonal(V; atol = atol_construct)
    Bdiag = fill(zero(T), Nh)
    Bdiag[end] = B[end, end]

    repair_info = _repair_even_gradient_first_column!(Geven,
                                                      Dfull_even,
                                                      Gfull_even,
                                                      xfull_even,
                                                      Sdiag,
                                                      Vdiag,
                                                      Bdiag,
                                                      r;
                                                      p = p,
                                                      atol = atol_construct,
                                                      custom_stencil_cols = custom_stencil_cols)
    interior_accuracy = repair_info.interior_accuracy
    rows_repaired = repair_info.rows_to_solve

    remaining_corrupted = _rows_with_nonzero_first_column(Geven; atol = atol_construct)
    isempty(remaining_corrupted) ||
        throw(ArgumentError("First column cleanup failed; nonzero rows remain: $remaining_corrupted"))

    closure_pattern = _closure_diagnostics(Geven)
    right_closure = isnothing(closure_from_operator) ?
                    closure_pattern.closure_width_right :
                    Int(closure_from_operator)

    _assert_global_even_interior_accuracy(Geven,
                                          r,
                                          interior_accuracy,
                                          rows_repaired,
                                          right_closure;
                                          atol = atol_construct)

    RHS = sparse(B - transpose(Geven) * V)
    D = spzeros(T, Nh, Nh)
    _set_divergence_rows!(D, RHS, Sdiag)
    _set_origin_row!(D, Godd, p)
    snap_sparse!(D; snap_factor = snap_factor)

    closure_width = isnothing(closure_from_operator) ?
                    closure_pattern.closure_width_right :
                    max(closure_pattern.closure_width_right, closure_from_operator)

    ops_canonical = SphericalOperators(r,
                                       S,
                                       V,
                                       B,
                                       Geven,
                                       Godd,
                                       D,
                                       closure_width,
                                       interior_accuracy,
                                       p,
                                       convert(T, R_canonical),
                                       source,
                                       mode,
                                       atol_construct,
                                       snap_factor,
                                       length(xfull_even),
                                       Nh)

    return_canonical && return ops_canonical

    Tout = isnothing(target_eltype) ? _default_scale_eltype(R) : target_eltype
    Tout <: Real || throw(ArgumentError("`target_eltype` must be a subtype of `Real`."))

    return scale_spherical_operators(ops_canonical,
                                     R;
                                     target_eltype = Tout,
                                     atol = atol)
end
