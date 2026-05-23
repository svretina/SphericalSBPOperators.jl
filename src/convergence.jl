function _gundlach_output_eltype(u_h::AbstractVector,
                                 u_ref::AbstractVector,
                                 h::Real,
                                 s::Real,
                                 h0::Real)
    return promote_type(float(eltype(u_h)),
                        float(eltype(u_ref)),
                        float(typeof(h)),
                        float(typeof(s)),
                        float(typeof(h0)))
end

function _gundlach_norm_output_eltype(u_h::AbstractVector,
                                      u_ref::AbstractVector,
                                      H::AbstractMatrix,
                                      h::Real,
                                      s::Real,
                                      R::Real,
                                      h0::Real)
    return promote_type(_gundlach_output_eltype(u_h, u_ref, h, s, h0),
                        float(eltype(H)),
                        float(typeof(R)))
end

function _validate_gundlach_inputs(u_h::AbstractVector,
                                   u_ref::AbstractVector;
                                   h::Real,
                                   h0::Real)
    length(u_h) == length(u_ref) ||
        throw(DimensionMismatch("`u_h` and `u_ref` must have the same length."))
    isempty(u_h) && throw(ArgumentError("`u_h` must be non-empty."))
    h > 0 || throw(ArgumentError("`h` must be positive."))
    h0 > 0 || throw(ArgumentError("`h0` must be positive."))
    return nothing
end

function _validate_gundlach_norm_matrix(H::AbstractMatrix, n::Integer)
    size(H, 1) == size(H, 2) ||
        throw(DimensionMismatch("`H` must be square."))
    size(H, 1) == n ||
        throw(DimensionMismatch("`H` must have size $(n)x$(n), got $(size(H))."))
    return nothing
end

function _validate_increasing_grid(r::AbstractVector)
    length(r) >= 2 ||
        throw(ArgumentError("Need at least two grid points for grid alignment."))

    @inbounds for i in 2:length(r)
        r[i] > r[i - 1] ||
            throw(ArgumentError("`r` must be strictly increasing for grid alignment."))
    end

    return nothing
end

@inline function _extract_wave_solution_like(simulation)
    return hasproperty(simulation, :sol) ? getproperty(simulation, :sol) : simulation
end

function _canonical_gundlach_variable(variable::Symbol)
    if variable in (:Π, :Pi, :pi)
        return :Π
    elseif variable in (:Ψ, :Psi, :psi)
        return :Ψ
    end
    throw(ArgumentError("`variable` must be one of :Π/:Pi/:pi or :Ψ/:Psi/:psi."))
end

@inline _gundlach_variable_symbol(::Val{:Π}) = :Π
@inline _gundlach_variable_symbol(::Val{:Ψ}) = :Ψ

function _gundlach_variable_history(solution, variable::Symbol)
    variable_sym = _canonical_gundlach_variable(variable)
    return variable_sym === :Π ? getproperty(solution, :Π) : getproperty(solution, :Ψ)
end

function _gundlach_variable_history(solution, variable::Val)
    return _gundlach_variable_history(solution, _gundlach_variable_symbol(variable))
end

function _gundlach_snapshot(history::AbstractMatrix, tidx::Integer)
    1 <= tidx <= size(history, 2) ||
        throw(BoundsError(history, (:, tidx)))
    return @view history[:, tidx]
end

function _gundlach_grid_spacing(r::AbstractVector)
    r_vec = collect(r)
    atol = _resolve_atol(eltype(r_vec), nothing)
    return _uniform_spacing(r_vec; atol = atol)
end

function _align_reference_to_grid(target_r::AbstractVector,
                                  ref_r::AbstractVector,
                                  ref_u::AbstractVector)
    length(ref_r) == length(ref_u) ||
        throw(DimensionMismatch("`ref_r` and `ref_u` must have the same length."))
    isempty(target_r) && throw(ArgumentError("`target_r` must be non-empty."))
    isempty(ref_r) && throw(ArgumentError("`ref_r` must be non-empty."))
    if collect(target_r) == collect(ref_r)
        return collect(ref_u)
    end

    _validate_increasing_grid(ref_r)
    length(target_r) == 1 || _validate_increasing_grid(target_r)

    first_ref = ref_r[1]
    last_ref = ref_r[end]
    T = promote_type(float(eltype(target_r)), float(eltype(ref_r)), float(eltype(ref_u)))
    aligned = Vector{T}(undef, length(target_r))

    @inbounds for i in eachindex(target_r)
        ri = target_r[i]
        (first_ref <= ri <= last_ref) ||
            throw(ArgumentError("Target grid point r=$(ri) lies outside the reference grid interval [$first_ref, $last_ref]."))

        idx = searchsortedfirst(ref_r, ri)
        if idx <= length(ref_r) && ref_r[idx] == ri
            aligned[i] = T(ref_u[idx])
        else
            left = idx > 1 ? ref_r[idx - 1] : missing
            right = idx <= length(ref_r) ? ref_r[idx] : missing
            throw(ArgumentError("Reference grid does not contain the target node r=$(ri). " *
                                "Exact nested-grid alignment is required; nearest bracketing nodes are " *
                                "left=$(left), right=$(right)."))
        end
    end

    return aligned
end

function _gundlach_norm_matrix(ops::WaveOperators, ::Val{:Π})
    return _wave_scalar_mass(ops)
end

function _gundlach_norm_matrix(ops::WaveOperators, ::Val{:Ψ})
    return _wave_vector_mass(ops)
end

function _gundlach_norm_matrix(ops, ::Val{:Π})
    if hasproperty(ops, :S)
        return getproperty(ops, :S)
    elseif hasproperty(ops, :H)
        return getproperty(ops, :H)
    end
    throw(ArgumentError("Could not resolve a scalar SBP norm matrix for the supplied operator object."))
end

function _gundlach_norm_matrix(ops, ::Val{:Ψ})
    if hasproperty(ops, :V)
        return getproperty(ops, :V)
    elseif hasproperty(ops, :H)
        return getproperty(ops, :H)
    end
    throw(ArgumentError("Could not resolve a vector SBP norm matrix for the supplied operator object."))
end

function _gundlach_radius(ops::WaveOperators, r::AbstractVector)
    return Float64(ops.R)
end

function _gundlach_radius(ops, r::AbstractVector)
    if hasproperty(ops, :R)
        return Float64(getproperty(ops, :R))
    end
    return maximum(float.(r))
end

function _gundlach_snapshot_pair(sol_h,
                                 sol_ref,
                                 variable,
                                 tidx::Integer)
    coarse = _extract_wave_solution_like(sol_h)
    ref = _extract_wave_solution_like(sol_ref)

    coarse_history = _gundlach_variable_history(coarse, variable)
    ref_history = _gundlach_variable_history(ref, variable)
    u_h = _gundlach_snapshot(coarse_history, tidx)
    u_ref_raw = _gundlach_snapshot(ref_history, tidx)

    1 <= tidx <= length(coarse.t) || throw(BoundsError(coarse.t, tidx))
    1 <= tidx <= length(ref.t) || throw(BoundsError(ref.t, tidx))
    coarse.t[tidx] == ref.t[tidx] ||
        throw(ArgumentError("The selected snapshots do not occur at the same saved time: coarse=$(coarse.t[tidx]), ref=$(ref.t[tidx])."))

    r = collect(coarse.r)
    u_ref = _align_reference_to_grid(r, ref.r, u_ref_raw)
    return (u_h = collect(u_h), u_ref = u_ref, r = r)
end

"""
    gundlach_error(u_h, u_ref; h, k, h0=0.1)

Compute the discrete Gundlach-style Richardson/self-convergence profile

`(h / h0)^(-k) * (u_h - u_ref)`.

This discrete SBP version does not include any `r^(p/2)` factor. The spherical
weighting is represented by the SBP norm matrices used later in
`gundlach_error_norm`, not by pointwise rescaling of the error itself. Here `k`
is the expected convergence order being tested, while `h0` is only a fixed
normalization scale and is not the fine-grid spacing. The reference data `u_ref`
may come from a true fine reference solution or from the next finer resolution
after restriction onto the current grid.
"""
function gundlach_error(u_h::AbstractVector,
                        u_ref::AbstractVector;
                        h::Real,
                        s::Real,
                        h0::Real = 0.1)
    _validate_gundlach_inputs(u_h, u_ref; h = h, h0 = h0)

    T = _gundlach_output_eltype(u_h, u_ref, h, s, h0)
    scale = (T(h) / T(h0))^(-T(s))
    err = Vector{T}(undef, length(u_h))

    @inbounds for i in eachindex(u_h, u_ref)
        err[i] = scale * (T(u_h[i]) - T(u_ref[i]))
    end

    return err
end

"""
    gundlach_error_norm(u_h, u_ref, H; h, s, R, h0=0.1)

Compute the discrete Gundlach-style SBP norm

`sqrt((e' * H * e) / R)`

where `e = gundlach_error(u_h, u_ref; h, k, h0)`. This discrete version does
not include `r^(p/2)` in the error profile; the spherical weighting is
represented directly by the SBP norm matrix `H`. Use `S` for `Π` and `V` for
`Ψ`. This diagnostic is intended for Richardson/self-convergence checks in the
native SBP discrete norm.
"""
function gundlach_error_norm(u_h::AbstractVector,
                             u_ref::AbstractVector,
                             H::AbstractMatrix;
                             h::Real,
                             s::Real,
                             R::Real,
                             h0::Real = 0.1)
    _validate_gundlach_inputs(u_h, u_ref; h = h, h0 = h0)
    _validate_gundlach_norm_matrix(H, length(u_h))
    R > 0 || throw(ArgumentError("`R` must be positive."))

    T = _gundlach_norm_output_eltype(u_h, u_ref, H, h, s, R, h0)
    err = gundlach_error(u_h, u_ref; h = h, s = s, h0 = h0)
    quad = real(dot(err, H * err))
    return sqrt(T(quad) / T(R))
end

"""
    gundlach_error(sol_h, sol_ref, ops, ::Val{:Π}, tidx; k, h0=0.1)
    gundlach_error(sol_h, sol_ref, ops, ::Val{:Ψ}, tidx; k, h0=0.1)

Compute a Gundlach-style discrete error profile for a saved wave snapshot. The
coarse-grid spacing `h` is inferred from the current grid, while the reference
snapshot must match the coarse-grid nodes exactly after restriction onto the
coarse grid. This wrapper returns only
the raw rescaled difference; the spherical weighting enters later through the
SBP norm matrices.
"""
function gundlach_error(sol_h,
                        sol_ref,
                        ops,
                        variable::Val,
                        tidx::Integer;
                        s::Real,
                        h0::Real = 0.1)
    data = _gundlach_snapshot_pair(sol_h, sol_ref, variable, tidx)
    h = _gundlach_grid_spacing(data.r)
    return gundlach_error(data.u_h, data.u_ref; h = h, s = s, h0 = h0)
end

function gundlach_error(sol_h,
                        sol_ref,
                        ops,
                        variable::Symbol,
                        tidx::Integer;
                        s::Real,
                        h0::Real = 0.1)
    variable_sym = _canonical_gundlach_variable(variable)
    variable_val = variable_sym === :Π ? Val(:Π) : Val(:Ψ)
    return gundlach_error(sol_h, sol_ref, ops, variable_val, tidx;
                          s = s,
                          h0 = h0)
end

"""
    gundlach_error_norm(sol_h, sol_ref, ops, ::Val{:Π}, tidx; s, h0=0.1)
    gundlach_error_norm(sol_h, sol_ref, ops, ::Val{:Ψ}, tidx; s, h0=0.1)

Compute the discrete Gundlach-style SBP norm for a saved wave snapshot. This
wrapper uses the repository's native norm matrices: `S` for `Π` and `V` for
`Ψ`. The normalization radius is taken from `ops.R` when available, otherwise
from the coarse grid.
"""
function gundlach_error_norm(sol_h,
                             sol_ref,
                             ops,
                             variable::Val,
                             tidx::Integer;
                             s::Real,
                             h0::Real = 0.1)
    data = _gundlach_snapshot_pair(sol_h, sol_ref, variable, tidx)
    h = _gundlach_grid_spacing(data.r)
    H = _gundlach_norm_matrix(ops, variable)
    R = _gundlach_radius(ops, data.r)
    return gundlach_error_norm(data.u_h, data.u_ref, H; h = h, s = s, R = R, h0 = h0)
end

function gundlach_error_norm(sol_h,
                             sol_ref,
                             ops,
                             variable::Symbol,
                             tidx::Integer;
                             s::Real,
                             h0::Real = 0.1)
    variable_sym = _canonical_gundlach_variable(variable)
    variable_val = variable_sym === :Π ? Val(:Π) : Val(:Ψ)
    return gundlach_error_norm(sol_h, sol_ref, ops, variable_val, tidx;
                               s = s,
                               h0 = h0)
end
