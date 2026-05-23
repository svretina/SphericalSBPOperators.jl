function _extract_diagonal(H::SparseMatrixCSC{T, Ti}; atol::T) where {
        T <: Real, Ti <: Integer,
    }
    n, m = size(H)
    n == m || throw(DimensionMismatch("Mass matrix must be square."))
    diag_entries = fill(zero(T), n)
    max_offdiag = zero(T)
    I, J, V = findnz(H)
    for k in eachindex(V)
        i = I[k]
        j = J[k]
        v = V[k]
        if i == j
            diag_entries[i] = v
        else
            av = abs(v)
            if av > max_offdiag
                max_offdiag = av
            end
        end
    end

    if T <: AbstractFloat
        allowed = max(atol, T(128) * eps(T))
        max_offdiag <= allowed ||
            throw(ArgumentError("Mass matrix must be diagonal-norm; max off-diagonal is $max_offdiag."))
    else
        max_offdiag == zero(T) ||
            throw(ArgumentError("Exact arithmetic requires strictly diagonal mass matrix."))
    end
    return diag_entries
end

function _set_divergence_rows!(
        D::SparseMatrixCSC{T, Ti},
        RHS::SparseMatrixCSC{T, Ti},
        Hdiag::Vector{T}
    ) where {T <: Real, Ti <: Integer}
    n = size(D, 1)
    for i in 1:n
        hi = Hdiag[i]
        hi == zero(T) &&
            throw(ArgumentError("Encountered H[$i,$i]=0 in staggered divergence assembly."))
        for j in 1:n
            v = RHS[i, j]
            if v != zero(T)
                D[i, j] = v / hi
            end
        end
    end
    return D
end

function _uniform_spacing(r::Vector{T}; atol::T) where {T <: AbstractFloat}
    length(r) >= 2 ||
        throw(ArgumentError("At least two grid points are required to infer spacing."))
    Δr = r[2] - r[1]
    Δr > zero(T) || throw(ArgumentError("Grid spacing must be strictly positive."))

    tol = max(atol, T(512) * eps(T) * max(one(T), abs(Δr)))
    for i in 3:length(r)
        δ = r[i] - r[i - 1]
        abs(δ - Δr) <= tol ||
            throw(ArgumentError("Non-uniform staggered half-grid detected."))
    end
    return Δr
end

function _uniform_spacing(r::Vector{T}; atol::T) where {T <: Real}
    length(r) >= 2 ||
        throw(ArgumentError("At least two grid points are required to infer spacing."))
    Δr = r[2] - r[1]
    Δr > zero(T) || throw(ArgumentError("Grid spacing must be strictly positive."))
    for i in 3:length(r)
        δ = r[i] - r[i - 1]
        δ == Δr || throw(ArgumentError("Non-uniform staggered half-grid detected."))
    end
    return Δr
end

function _default_scale_eltype(R)
    if R isa AbstractFloat
        return typeof(R)
    end
    if R isa Rational
        return Rational{BigInt}
    end
    if R isa Integer
        return Rational{BigInt}
    end
    return Float64
end

function _scale_sparse_matrix(
        A::SparseMatrixCSC{Ta, Ti},
        factor,
        ::Type{Tb};
        snap_factor::Float64
    ) where {Ta <: Real, Ti <: Integer, Tb <: Real}
    n, m = size(A)
    I, J, V = findnz(A)
    f = convert(Tb, factor)
    W = Vector{Tb}(undef, length(V))
    for k in eachindex(V)
        W[k] = convert(Tb, V[k]) * f
    end
    B = sparse(I, J, W, n, m)
    snap_sparse!(B; snap_factor = snap_factor)
    return B
end

"""
    scale_spherical_operators(ops, R; target_eltype=nothing, atol=nothing)

Scale staggered operators from current spacing to new physical radius `R`.
"""
function scale_spherical_operators(
        ops::SphericalOperators,
        R;
        target_eltype::Union{Nothing, Type} = nothing,
        atol = nothing
    )
    ops.Nh >= 2 ||
        throw(ArgumentError("At least two staggered points are required for scaling."))

    Tout = isnothing(target_eltype) ? _default_scale_eltype(R) : target_eltype
    Tout <: Real || throw(ArgumentError("`target_eltype` must be a subtype of `Real`."))

    Nhalf = ops.Nh
    Rtarget = convert(Tout, R)
    # Staggered spacing with points at (k-1/2)Δr and k=1..Nhalf.
    Δr_target = (convert(Tout, 2) * Rtarget) / convert(Tout, 2 * Nhalf - 1)

    Tsrc = eltype(ops.r)
    atol_src = _resolve_atol(Tsrc, nothing)
    Δr_src = _uniform_spacing(ops.r; atol = atol_src)
    scale_ratio = Δr_target / convert(Tout, Δr_src)

    r_scaled = Vector{Tout}(undef, ops.Nh)
    for i in eachindex(ops.r)
        r_scaled[i] = convert(Tout, ops.r[i]) * scale_ratio
    end

    H_scaled = _scale_sparse_matrix(ops.H, scale_ratio^(ops.p + 1), Tout; snap_factor = ops.snap_factor)
    B_scaled = _scale_sparse_matrix(ops.B, scale_ratio^ops.p, Tout; snap_factor = ops.snap_factor)
    Geven_scaled = _scale_sparse_matrix(ops.Geven, inv(scale_ratio), Tout; snap_factor = ops.snap_factor)
    Godd_scaled = _scale_sparse_matrix(ops.Godd, inv(scale_ratio), Tout; snap_factor = ops.snap_factor)
    D_scaled = _scale_sparse_matrix(ops.D, inv(scale_ratio), Tout; snap_factor = ops.snap_factor)

    atol_scaled = _resolve_atol(Tout, atol)

    return SphericalOperators(
        r_scaled,
        H_scaled,
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
        ops.Nh
    )
end

"""
    spherical_operators(source; kwargs...)

Construct staggered folded spherical operators on `[0, R]` using an even number of
mirrored full-grid nodes (no node at the origin).
"""
function spherical_operators(
        source;
        accuracy_order,
        N,
        R,
        p::Int = 2,
        mode = FastMode(),
        atol = nothing,
        snap_factor::Float64 = 64.0,
        custom_stencil_cols::Union{Nothing, Vector{Int}} = nothing,
        return_canonical::Bool = false,
        target_eltype::Union{Nothing, Type} = nothing
    )
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    custom_stencil_cols === nothing ||
        throw(ArgumentError("`custom_stencil_cols` is not used in staggered mode."))

    Nint = Int(N)
    Nint > 0 || throw(ArgumentError("`N` must be positive."))

    # Canonical staggered grid: Δr=1, full node count=2N, and half-grid nodes are
    # 1/2, 3/2, ..., N-1/2.
    R_canonical = big(2 * Nint - 1) // 2
    Dfull, xfull,
        Gfull,
        Hfull = _build_full_grid_objects(
        source;
        accuracy_order = Int(accuracy_order),
        N = Nint,
        R = R_canonical,
        mode = mode
    )

    T = eltype(xfull)
    atol_construct = _resolve_atol(T, nothing)

    r, Rop, Eeven, Eodd = _build_folding_operators_staggered(xfull; atol = atol_construct)
    Nh = length(r)

    Geven = sparse(Rop * Gfull * Eeven)
    Godd = sparse(Rop * Gfull * Eodd)
    snap_sparse!(Geven; snap_factor = snap_factor)
    snap_sparse!(Godd; snap_factor = snap_factor)

    half_factor = convert(T, 1) / convert(T, 2)
    Hcart_half = sparse(half_factor * (transpose(Eeven) * Hfull * Eeven))
    metric = spdiagm(0 => r .^ p)
    H = sparse(Hcart_half * metric)
    snap_sparse!(H; snap_factor = snap_factor)

    B = spzeros(T, Nh, Nh)
    B[end, end] = r[end]^p
    snap_sparse!(B; snap_factor = snap_factor)

    Hdiag = _extract_diagonal(H; atol = atol_construct)
    all(!=(zero(T)), Hdiag) ||
        throw(ArgumentError("Staggered mass matrix contains zero diagonal entries."))

    RHS = sparse(B - transpose(Geven) * H)
    D = spzeros(T, Nh, Nh)
    _set_divergence_rows!(D, RHS, Hdiag)
    snap_sparse!(D; snap_factor = snap_factor)

    closure_pattern = _closure_diagnostics(Geven)
    closure_from_operator = _boundary_closure_width_from_operator(Dfull)
    closure_width = isnothing(closure_from_operator) ?
        closure_pattern.closure_width_right :
        max(closure_pattern.closure_width_right, closure_from_operator)

    ops_canonical = SphericalOperators(
        r,
        H,
        B,
        Geven,
        Godd,
        D,
        closure_width,
        Int(accuracy_order),
        p,
        convert(T, R_canonical),
        source,
        mode,
        atol_construct,
        snap_factor,
        length(xfull),
        Nh
    )

    return_canonical && return ops_canonical

    Tout = isnothing(target_eltype) ? _default_scale_eltype(R) : target_eltype
    Tout <: Real || throw(ArgumentError("`target_eltype` must be a subtype of `Real`."))

    return scale_spherical_operators(
        ops_canonical,
        R;
        target_eltype = Tout,
        atol = atol
    )
end

@inline apply_even_gradient(ops::SphericalOperators, phi) = ops.Geven * phi
@inline apply_odd_derivative(ops::SphericalOperators, u) = ops.Godd * u
@inline apply_divergence(ops::SphericalOperators, u) = ops.D * u

"""
    enforce_odd!(u)

Staggered grids do not include the origin node, so no direct odd-origin projection is
required. This function is a no-op for API compatibility.
"""
function enforce_odd!(u::AbstractVector)
    return u
end

"""
    check_odd(u; tol=...)

On staggered grids there is no origin point to check directly; this returns `true` for
API compatibility.
"""
function check_odd(u::AbstractVector; tol = zero(eltype(u)))
    _ = u
    _ = tol
    return true
end
