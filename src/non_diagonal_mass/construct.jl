function _resolve_non_diagonal_points(N::Union{Nothing, Integer},
                                      points::Union{Nothing, Integer})
    if isnothing(N) && isnothing(points)
        throw(ArgumentError("Provide either `N` or `points`."))
    end

    points_from_N = if isnothing(N)
        nothing
    else
        Nint = Int(N)
        Nint > 0 || throw(ArgumentError("`N` must be positive."))
        Nint + 1
    end

    points_int = isnothing(points) ? points_from_N : Int(points)
    points_int > 1 || throw(ArgumentError("`points` must be > 1."))
    if !isnothing(points_from_N) && !isnothing(points) && points_int != points_from_N
        throw(ArgumentError("`points` must equal `N + 1` when both are provided."))
    end
    return points_int
end

function _resolve_non_diagonal_grid(points::Int, R, h)
    if isnothing(R) && isnothing(h)
        throw(ArgumentError("Provide `R` or `h` for non-diagonal operator construction."))
    end
    h_use = isnothing(h) ? (R / (points - 1)) : h
    R_use = isnothing(R) ? (h_use * (points - 1)) : R
    return R_use, h_use
end

@inline _non_diagonal_canonical_radius(points::Int) = big(points - 1) // 1

function _infer_non_diagonal_closure_width(setup, solved, Geven)
    candidates = Int[]

    if hasproperty(solved, :closure_right)
        push!(candidates, Int(getproperty(solved, :closure_right)))
    end
    if hasproperty(setup, :Dfull)
        inferred = _boundary_closure_width_from_operator(getproperty(setup, :Dfull))
        if !isnothing(inferred)
            push!(candidates, Int(inferred))
        end
    end

    push!(candidates, Int(_closure_diagnostics(Geven).closure_width_right))
    return max(0, maximum(candidates))
end

function _build_non_diagonal_ops(source,
                                 solved;
                                 snap_factor::Float64 = 0.0)
    setup = solved.setup
    r = setup.r
    Geven = setup.Geven
    T = eltype(r)
    closure_width = _infer_non_diagonal_closure_width(setup, solved, Geven)

    return SphericalOperators(r,
                              setup.Hcart_half,
                              solved.S,
                              solved.V,
                              solved.B,
                              Geven,
                              setup.Godd,
                              solved.D,
                              closure_width,
                              Int(setup.accuracy_order),
                              Int(setup.p),
                              convert(T, setup.R),
                              source,
                              setup.mode,
                              convert(T, setup.atol),
                              snap_factor,
                              length(setup.xfull),
                              length(r))
end

"""
    scale_spherical_operators(ops, R; target_eltype=nothing, atol=nothing)

Scale non-diagonal-mass operators from their current spacing to a new physical radius `R`.
"""
function scale_spherical_operators(ops::SphericalOperators,
                                   R;
                                   target_eltype::Union{Nothing, Type} = nothing,
                                   atol = nothing)
    ops.Nh >= 2 ||
        throw(ArgumentError("At least two half-grid points are required for scaling."))

    Tout = isnothing(target_eltype) ? _default_scale_eltype(R) : target_eltype
    Tout <: Real || throw(ArgumentError("`target_eltype` must be a subtype of `Real`."))

    N = ops.Nh - 1
    Rtarget = convert(Tout, R)
    Δr_target = Rtarget / convert(Tout, N)

    Tsrc = eltype(ops.r)
    atol_src = _resolve_atol(Tsrc, nothing)
    Δr_src = _uniform_spacing(ops.r; atol = atol_src)
    scale_ratio = Δr_target / convert(Tout, Δr_src)

    r_scaled = Vector{Tout}(undef, ops.Nh)
    @inbounds for i in eachindex(ops.r)
        r_scaled[i] = convert(Tout, ops.r[i]) * scale_ratio
    end

    S_scaled = _scale_sparse_matrix(ops.S, scale_ratio^(ops.p + 1), Tout;
                                    snap_factor = ops.snap_factor)
    V_scaled = _scale_sparse_matrix(ops.V, scale_ratio^(ops.p + 1), Tout;
                                    snap_factor = ops.snap_factor)
    H_scaled = _scale_sparse_matrix(ops.H, scale_ratio, Tout;
                                    snap_factor = ops.snap_factor)
    B_scaled = _scale_sparse_matrix(ops.B, scale_ratio^ops.p, Tout;
                                    snap_factor = ops.snap_factor)
    Geven_scaled = _scale_sparse_matrix(ops.Geven, inv(scale_ratio), Tout;
                                        snap_factor = ops.snap_factor)
    Godd_scaled = _scale_sparse_matrix(ops.Godd, inv(scale_ratio), Tout;
                                       snap_factor = ops.snap_factor)
    D_scaled = _scale_sparse_matrix(ops.D, inv(scale_ratio), Tout;
                                    snap_factor = ops.snap_factor)

    atol_scaled = _resolve_atol(Tout, atol)

    return SphericalOperators(r_scaled,
                              H_scaled,
                              S_scaled,
                              V_scaled,
                              B_scaled,
                              Geven_scaled,
                              Godd_scaled,
                              D_scaled,
                              ops.closure_width,
                              ops.accuracy_order,
                              ops.p,
                              Rtarget,
                              ops.source,
                              ops.mode,
                              atol_scaled,
                              ops.snap_factor,
                              ops.M_full,
                              ops.Nh)
end

"""
    spherical_operators(source; accuracy_order, N=nothing, points=nothing, R=nothing, h=nothing,
                        p=2, mode=SafeMode(), atol=nothing, return_canonical=false,
                        target_eltype=nothing, exact_solve=true, verbose=false, kwargs...)

Construct non-diagonal-mass folded spherical operators using the existing order-specific
SBP4/SBP6 split-mass solvers.

Implementation details:
- construction is performed on a canonical grid with `Δr = 1`;
- the split-mass solve itself is left unchanged and runs on that reference grid;
- the final operator set is then scaled to the requested physical radius.

`N` follows the diagonal-mass API and denotes the number of subintervals on `[0, R]`;
the folded grid therefore contains `N + 1` points. Use `points` only when you want
to specify the inclusive node count directly.
"""
function spherical_operators(source;
                             accuracy_order::Int,
                             N::Union{Nothing, Integer} = nothing,
                             points::Union{Nothing, Integer} = nothing,
                             R = nothing,
                             h = nothing,
                             p::Int = 2,
                             mode = SafeMode(),
                             atol = nothing,
                             return_canonical::Bool = false,
                             target_eltype::Union{Nothing, Type} = nothing,
                             exact_solve::Bool = true,
                             verbose::Bool = false,
                             kwargs...)
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    return_canonical &&
        throw(ArgumentError("`return_canonical=true` is not supported in non-diagonal mode."))

    points_int = _resolve_non_diagonal_points(N, points)
    R_use, h_use = _resolve_non_diagonal_grid(points_int, R, h)
    R_canonical = _non_diagonal_canonical_radius(points_int)

    solved = if accuracy_order == 4
        sbp4_solve_accuracy_constraints(source;
                                        accuracy_order = accuracy_order,
                                        points = points_int,
                                        h = 1,
                                        R = R_canonical,
                                        p = p,
                                        mode = mode,
                                        atol = atol,
                                        exact_solve = exact_solve,
                                        verbose = verbose)
    elseif accuracy_order == 6
        sbp6_solve_accuracy_constraints(source;
                                        accuracy_order = accuracy_order,
                                        points = points_int,
                                        h = 1,
                                        R = R_canonical,
                                        p = p,
                                        mode = mode,
                                        atol = atol,
                                        exact_solve = exact_solve,
                                        verbose = verbose,
                                        kwargs...)
    else
        throw(ArgumentError("Non-diagonal unified mode currently supports only `accuracy_order ∈ {4,6}`; got $accuracy_order. Use the specialized SBP8 constructor directly if needed."))
    end

    ops_canonical = _build_non_diagonal_ops(source, solved)
    Tout = isnothing(target_eltype) ? _default_scale_eltype(R_use) : target_eltype
    Tout <: Real || throw(ArgumentError("`target_eltype` must be a subtype of `Real`."))

    return scale_spherical_operators(ops_canonical,
                                     R_use;
                                     target_eltype = Tout,
                                     atol = atol)
end

"""
    sbp6_exp_spherical_operators(source; N=nothing, points=nothing, R=nothing, h=nothing,
                                 p=2, mode=SafeMode(), atol=nothing,
                                 target_eltype=nothing, exact_solve=true, verbose=false,
                                 kwargs...)

Construct spherical operators for the experimental non-diagonal SBP6 closure with the
same folded-grid API as `spherical_operators`, but using the `sbp6_exp` split-mass
solve. As in the standard diagonal path, construction is performed on a canonical
grid with `Δr = 1` and the final operator set is scaled afterward.
"""
function sbp6_exp_spherical_operators(source;
                                      N::Union{Nothing, Integer} = nothing,
                                      points::Union{Nothing, Integer} = nothing,
                                      R = nothing,
                                      h = nothing,
                                      p::Int = 2,
                                      mode = SafeMode(),
                                      atol = nothing,
                                      target_eltype::Union{Nothing, Type} = nothing,
                                      exact_solve::Bool = true,
                                      verbose::Bool = false,
                                      kwargs...)
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))

    points_int = _resolve_non_diagonal_points(N, points)
    R_use, h_use = _resolve_non_diagonal_grid(points_int, R, h)
    R_canonical = _non_diagonal_canonical_radius(points_int)

    solved = sbp6_exp_solve_accuracy_constraints(source;
                                                 accuracy_order = 6,
                                                 points = points_int,
                                                 h = 1,
                                                 R = R_canonical,
                                                 p = p,
                                                 mode = mode,
                                                 atol = atol,
                                                 exact_solve = exact_solve,
                                                 verbose = verbose,
                                                 kwargs...)

    ops_canonical = _build_non_diagonal_ops(source, solved)
    Tout = isnothing(target_eltype) ? _default_scale_eltype(R_use) : target_eltype
    Tout <: Real || throw(ArgumentError("`target_eltype` must be a subtype of `Real`."))

    return scale_spherical_operators(ops_canonical,
                                     R_use;
                                     target_eltype = Tout,
                                     atol = atol)
end
