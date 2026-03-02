function _diagonal_entries(H::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
    n, m = size(H)
    n == m || throw(DimensionMismatch("Matrix must be square to extract diagonal entries."))
    diag_entries = fill(zero(T), n)
    I, J, V = findnz(H)
    @inbounds for k in eachindex(V)
        i = I[k]
        j = J[k]
        i == j && (diag_entries[i] = V[k])
    end
    return diag_entries
end

@inline function _split_mass_supported_order(accuracy_order::Int)
    return accuracy_order in (2, 4, 6, 8)
end

@inline function _split_mass_layout(layout::Symbol)
    layout in (:collocated, :staggered) ||
        throw(ArgumentError("`layout` must be :collocated or :staggered; got `$layout`."))
    return layout
end

@inline function _split_mass_moment_count(accuracy_order::Int)
    _split_mass_supported_order(accuracy_order) ||
        throw(ArgumentError("Only accuracy orders 2/4/6/8 are supported for split-mass construction; got $accuracy_order."))
    return fld(accuracy_order, 2)
end

@inline function _split_mass_origin_radius(accuracy_order::Int)
    Nmom = _split_mass_moment_count(accuracy_order)
    return max(0, Nmom - 1)
end

function _split_mass_allowed_v_pairs(accuracy_order::Int,
                                     Nh::Int,
                                     layout::Symbol;
                                     origin_bandwidth::Union{Nothing, Int} = nothing)
    Nh >= 1 || throw(ArgumentError("`Nh` must be positive."))
    layout_norm = _split_mass_layout(layout)
    radius_default = _split_mass_origin_radius(accuracy_order)
    radius = isnothing(origin_bandwidth) ? radius_default : Int(origin_bandwidth)
    radius >= 0 || throw(ArgumentError("`origin_bandwidth` must satisfy >= 0."))

    pairs = Tuple{Int, Int}[]
    if accuracy_order == 2 || radius == 0
        return pairs
    end

    if accuracy_order == 4
        if layout_norm === :collocated
            Nh >= 3 && push!(pairs, (2, 3))  # u_{3/2}
            Nh >= 4 && push!(pairs, (3, 4))  # u_{5/2}
        else
            Nh >= 3 && push!(pairs, (2, 3))  # u_1
        end
        return pairs
    end

    # Starter generalized pattern (orders 6/8):
    # compact origin block with half-bandwidth `radius`, yielding local
    # bandwidths 5 (radius=2) and 7 (radius=3).
    block_start = 2
    block_end = min(Nh, 2 + radius)
    if block_end <= block_start
        return pairs
    end
    @inbounds for i in block_start:(block_end - 1)
        jmax = min(block_end, i + radius)
        for j in (i + 1):jmax
            push!(pairs, (i, j))
        end
    end
    return pairs
end

@inline function _split_mass_pair_key(i::Int, j::Int)
    return i <= j ? (i, j) : (j, i)
end

function _split_mass_forbidden_offdiag_max(V::SparseMatrixCSC{T, Ti},
                                           allowed_pairs::Vector{Tuple{Int, Int}}) where {T <: Real, Ti <: Integer}
    allowed = Set(allowed_pairs)
    I, J, VV = findnz(V)
    max_forbidden = zero(T)
    @inbounds for k in eachindex(VV)
        i = I[k]
        j = J[k]
        i == j && continue
        key = _split_mass_pair_key(i, j)
        if !(key in allowed)
            av = abs(VV[k])
            if av > max_forbidden
                max_forbidden = av
            end
        end
    end
    return max_forbidden
end

function _split_mass_symmetry_defect_max(V::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
    I, J, VV = findnz(V)
    max_defect = zero(T)
    @inbounds for k in eachindex(VV)
        i = I[k]
        j = J[k]
        i > j && continue
        defect = abs(V[i, j] - V[j, i])
        if defect > max_defect
            max_defect = defect
        end
    end
    return max_defect
end

function is_spd(V; atol = 0.0)
    _ = atol
    n, m = size(V)
    n == m || return false
    issymmetric(V) || return false
    try
        cholesky(Symmetric(Matrix{Float64}(V)); check = true)
        return true
    catch
        return false
    end
end

function _split_mass_default_profile(r::Vector{T}, p::Int) where {T <: Real}
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    return r .^ p
end

function _split_mass_resolve_diagonal(r::Vector{T},
                                      p::Int,
                                      values::Union{Nothing, AbstractVector},
                                      logs::Union{Nothing, AbstractVector},
                                      name::Symbol) where {T <: Real}
    Nh = length(r)
    !(isnothing(values) && isnothing(logs)) || return _split_mass_default_profile(r, p)
    isnothing(values) || isnothing(logs) ||
        throw(ArgumentError("Provide either `$name` or `log_$(name)`, not both."))

    if !isnothing(values)
        length(values) == Nh ||
            throw(DimensionMismatch("`$name` must have length $Nh."))
        return convert(Vector{T}, values)
    end

    T <: AbstractFloat ||
        throw(ArgumentError("`log_$(name)` parameterization requires floating-point grid coordinates."))
    length(logs) == Nh ||
        throw(DimensionMismatch("`log_$(name)` must have length $Nh."))

    out = Vector{T}(undef, Nh)
    @inbounds for i in eachindex(logs)
        out[i] = exp(convert(T, logs[i]))
    end
    return out
end

function _split_mass_order2_consistency_error(sdiag, vdiag)
    m = zero(eltype(sdiag))
    @inbounds for i in eachindex(sdiag)
        err = abs(vdiag[i] - sdiag[i])
        if err > m
            m = err
        end
    end
    return m
end

@inline function _seed_band_scale_convert(::Type{T}, scale::Real) where {T <: AbstractFloat}
    return convert(T, scale)
end

@inline function _seed_band_scale_convert(::Type{T}, scale::Real) where {T <: Real}
    if scale isa Integer || scale isa Rational
        return convert(T, scale)
    end
    return convert(T, rationalize(BigInt, Float64(scale)))
end

function _split_mass_seed_offdiag(vdiag::Vector{T},
                                  accuracy_order::Int,
                                  layout::Symbol;
                                  origin_bandwidth::Union{Nothing, Int} = nothing,
                                  band_scale::Real = 1 // 10^12) where {T <: Real}
    Nh = length(vdiag)
    pairs = _split_mass_allowed_v_pairs(
                                       accuracy_order,
                                       Nh,
                                       layout;
                                       origin_bandwidth = origin_bandwidth
                                      )
    isempty(pairs) && return Vector{T}()

    scale_T = _seed_band_scale_convert(T, band_scale)
    vals = Vector{T}(undef, length(pairs))
    @inbounds for idx in eachindex(pairs)
        i, j = pairs[idx]
        base = min(abs(vdiag[i]), abs(vdiag[j]))
        vals[idx] = scale_T * base
    end
    return vals
end

@inline _as_big_rational(x::Rational{BigInt}) = x
@inline _as_big_rational(x::Integer) = big(x) // 1
@inline _as_big_rational(x::Rational{<:Integer}) = big(numerator(x)) // big(denominator(x))

function _as_big_rational(x::AbstractFloat)
    isfinite(x) ||
        throw(ArgumentError("Cannot convert non-finite floating-point value `$x` to Rational{BigInt}."))
    return rationalize(BigInt, x)
end

function _as_big_rational(x::Real)
    throw(ArgumentError("Could not convert value of type $(typeof(x)) to Rational{BigInt}."))
end

function _split_mass_origin_block_from_bidiag(origin_block_cholesky,
                                              ::Type{T}) where {T <: Real}
    length(origin_block_cholesky) == 5 ||
        throw(DimensionMismatch("`origin_block_cholesky` must contain five entries `(a,b,c,d,e)`."))
    a = convert(T, origin_block_cholesky[1])
    b = convert(T, origin_block_cholesky[2])
    c = convert(T, origin_block_cholesky[3])
    d = convert(T, origin_block_cholesky[4])
    e = convert(T, origin_block_cholesky[5])
    return (
            v2 = a * a,
            u32 = a * b,
            v3 = b * b + c * c,
            u52 = c * d,
            v4 = d * d + e * e
           )
end

"""
    construct_split_mass_matrices(r; kwargs...)

Construct split spherical masses `S` (even/scalar) and `V` (odd/vector) with
order-dependent structural constraints intended for optimization workflows.

Supported orders are `2, 4, 6, 8` and layouts are `:collocated` or `:staggered`.

Structural rules:
- order 2: `S` and `V` diagonal, with `V = S`;
- order 4:
  - collocated: only `V[2,3]` and `V[3,4]` may be nonzero off-diagonal;
  - staggered: only `V[2,3]` may be nonzero off-diagonal;
- orders 6/8: compact origin block with local half-bandwidth
  `origin_bandwidth = order/2 - 1` by default (5/7-band near-origin pattern).
"""
function construct_split_mass_matrices(r::Vector{T};
                                       accuracy_order::Int,
                                       p::Int,
                                       layout::Symbol = :collocated,
                                       s_diag::Union{Nothing, AbstractVector} = nothing,
                                       v_diag::Union{Nothing, AbstractVector} = nothing,
                                       log_s::Union{Nothing, AbstractVector} = nothing,
                                       log_v::Union{Nothing, AbstractVector} = nothing,
                                       v_offdiag::Union{Nothing, AbstractVector} = nothing,
                                       origin_block_cholesky::Union{Nothing, AbstractVector} = nothing,
                                       origin_bandwidth::Union{Nothing, Int} = nothing,
                                       strict::Bool = true,
                                       snap_factor::Float64 = 64.0) where {T <: Real}
    Nh = length(r)
    Nh >= 1 || throw(ArgumentError("`r` must be non-empty."))
    _split_mass_supported_order(accuracy_order) ||
        throw(ArgumentError("Only accuracy orders 2/4/6/8 are supported; got $accuracy_order."))
    layout_norm = _split_mass_layout(layout)

    sdiag = _split_mass_resolve_diagonal(r, p, s_diag, log_s, :s_diag)
    vdiag = _split_mass_resolve_diagonal(r, p, v_diag, log_v, :v_diag)

    if accuracy_order == 2
        !isnothing(v_offdiag) &&
            throw(ArgumentError("Order-2 split masses are diagonal; `v_offdiag` is not used."))
        !isnothing(origin_block_cholesky) &&
            throw(ArgumentError("`origin_block_cholesky` is only valid for order-4 collocated construction."))
        if strict
            v_eq_s = _split_mass_order2_consistency_error(sdiag, vdiag)
            tol = T <: AbstractFloat ? max(T(256) * eps(T), _resolve_atol(T, nothing)) : zero(T)
            v_eq_s <= tol ||
                throw(ArgumentError("Order-2 requires `V = S`; max diagonal mismatch = $v_eq_s."))
        end
        vdiag .= sdiag
    end

    allowed_pairs = _split_mass_allowed_v_pairs(
                                           accuracy_order,
                                           Nh,
                                           layout_norm;
                                           origin_bandwidth = origin_bandwidth
                                          )

    offdiag_values = if isnothing(origin_block_cholesky)
        if isnothing(v_offdiag)
            fill(zero(T), length(allowed_pairs))
        else
            length(v_offdiag) == length(allowed_pairs) ||
                throw(
                      DimensionMismatch(
                                        "`v_offdiag` must have length $(length(allowed_pairs)) " *
                                        "for order=$accuracy_order layout=$layout_norm."
                                       )
                     )
            convert(Vector{T}, v_offdiag)
        end
    else
        (accuracy_order == 4 && layout_norm === :collocated) ||
            throw(ArgumentError("`origin_block_cholesky` is only valid for order-4 collocated construction."))
        Nh >= 4 || throw(ArgumentError("Order-4 collocated origin block requires at least 4 half-grid points."))
        isnothing(v_offdiag) ||
            throw(ArgumentError("Provide either `v_offdiag` or `origin_block_cholesky`, not both."))

        block = _split_mass_origin_block_from_bidiag(origin_block_cholesky, T)
        vdiag[2] = block.v2
        vdiag[3] = block.v3
        vdiag[4] = block.v4
        vals = fill(zero(T), length(allowed_pairs))
        for idx in eachindex(allowed_pairs)
            pair = allowed_pairs[idx]
            if pair == (2, 3)
                vals[idx] = block.u32
            elseif pair == (3, 4)
                vals[idx] = block.u52
            end
        end
        vals
    end

    S = spdiagm(0 => sdiag)
    V = spdiagm(0 => vdiag)
    for idx in eachindex(allowed_pairs)
        i, j = allowed_pairs[idx]
        val = offdiag_values[idx]
        if val != zero(T)
            V[i, j] = val
            V[j, i] = val
        end
    end
    snap_sparse!(S; snap_factor = snap_factor)
    snap_sparse!(V; snap_factor = snap_factor)

    if strict
        tol = T <: AbstractFloat ? max(T(256) * eps(T), _resolve_atol(T, nothing)) : zero(T)
        forbidden = _split_mass_forbidden_offdiag_max(V, allowed_pairs)
        forbidden <= tol ||
            throw(ArgumentError("Detected forbidden V off-diagonal magnitude $forbidden for order=$accuracy_order layout=$layout_norm."))

        symmetry_defect = _split_mass_symmetry_defect_max(V)
        symmetry_defect <= tol ||
            throw(ArgumentError("V must be symmetric; max|V-V'|=$symmetry_defect."))
    end

    return (
            S = S,
            V = V,
            s_diag = sdiag,
            v_diag = vdiag,
            allowed_v_offdiagonals = allowed_pairs,
            layout = layout_norm,
            origin_bandwidth = isnothing(origin_bandwidth) ? _split_mass_origin_radius(accuracy_order) : Int(origin_bandwidth),
            accuracy_order = accuracy_order,
            p = p
           )
end

function construct_split_mass_matrices_seed(r::AbstractVector,
                                            G::AbstractMatrix;
                                            accuracy_order::Int,
                                            p::Int,
                                            layout::Symbol = :collocated,
                                            boundary_count::Union{Nothing, Int} = nothing,
                                            enforced_rows::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                                            normalization_index::Union{Nothing, Int} = nothing,
                                            farfield_points::Int = 2,
                                            origin_bandwidth::Union{Nothing, Int} = nothing,
                                            s_seed::Union{Nothing, AbstractVector} = nothing,
                                            v_seed::Union{Nothing, AbstractVector} = nothing,
                                            v_offdiag_seed::Union{Nothing, AbstractVector} = nothing,
                                            regularization::Real = 0.0,
                                            pd_epsilon::Real = 1.0e-12,
                                            strict::Bool = true,
                                            snap_factor::Float64 = 64.0)
    _split_mass_supported_order(accuracy_order) ||
        throw(ArgumentError("Only accuracy orders 2/4/6/8 are supported; got $accuracy_order."))
    layout_norm = _split_mass_layout(layout)
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))

    Nh = length(r)
    Nh >= 1 || throw(ArgumentError("`r` must be non-empty."))
    size(G, 1) == Nh == size(G, 2) ||
        throw(DimensionMismatch("`G` and `r` dimensions are incompatible."))

    T = promote_type(eltype(r), eltype(G))
    Tq = Rational{BigInt}
    r_q = Tq[_as_big_rational(ri) for ri in r]

    Gs = sparse(G)
    GI, GJ, GV = findnz(Gs)
    Gq = sparse(GI, GJ, Tq[_as_big_rational(vij) for vij in GV], Nh, Nh)

    s_seed_resolved = if isnothing(s_seed)
        _split_mass_default_profile(r_q, p)
    else
        Tq[_as_big_rational(si) for si in s_seed]
    end
    v_seed_resolved = if isnothing(v_seed)
        _split_mass_default_profile(r_q, p)
    else
        Tq[_as_big_rational(vi) for vi in v_seed]
    end
    length(s_seed_resolved) == Nh ||
        throw(DimensionMismatch("`s_seed` must have length $Nh."))
    length(v_seed_resolved) == Nh ||
        throw(DimensionMismatch("`v_seed` must have length $Nh."))

    if isnothing(enforced_rows)
        if isnothing(boundary_count)
            closures = _infer_fd_closure_from_gradient(Matrix{Tq}(Gq); atol = zero(Tq))
            b = max(closures.left, closures.right)
            row_start = b + 1
            row_end = Nh - b
            rows = row_start <= row_end ? collect(row_start:row_end) : Int[]
        else
            b = Int(boundary_count)
            b >= 0 || throw(ArgumentError("`boundary_count` must satisfy >= 0."))
            row_start = b + 1
            row_end = Nh - b
            rows = row_start <= row_end ? collect(row_start:row_end) : Int[]
        end
    else
        rows = copy(Int.(enforced_rows))
    end

    if !isempty(rows)
        any(i -> i < 1 || i > Nh, rows) &&
            throw(ArgumentError("`enforced_rows` must lie in 1:$Nh."))
    end

    allowed_pairs = _split_mass_allowed_v_pairs(
                                           accuracy_order,
                                           Nh,
                                           layout_norm;
                                           origin_bandwidth = origin_bandwidth
                                          )
    u_seed = if isnothing(v_offdiag_seed)
        fill(zero(Tq), length(allowed_pairs))
    else
        length(v_offdiag_seed) == length(allowed_pairs) ||
            throw(DimensionMismatch("`v_offdiag_seed` must have length $(length(allowed_pairs))."))
        Tq[_as_big_rational(ui) for ui in v_offdiag_seed]
    end

    n_s = Nh
    n_v = Nh
    n_u = length(allowed_pairs)
    n_unknowns = n_s + n_v + n_u
    idx_s(i) = i
    idx_v(i) = n_s + i
    idx_u(k) = n_s + n_v + k

    odd_degrees = _split_mass_monomial_degrees(accuracy_order)
    rp = r_q .^ p
    norm_idx = isnothing(normalization_index) ? Nh : clamp(Int(normalization_index), 1, Nh)
    ff_count = max(0, farfield_points)
    ff_start = max(1, Nh - ff_count + 1)
    pair_to_idx = Dict{Tuple{Int, Int}, Int}()
    @inbounds for (k, pair) in enumerate(allowed_pairs)
        pair_to_idx[pair] = k
    end

    A_rows = Vector{Vector{Tq}}()
    b_rows = Tq[]

    function push_eq!(row, rhs)
        push!(A_rows, row)
        push!(b_rows, rhs)
        return nothing
    end

    # Order-2 consistency constraint V=S.
    if accuracy_order == 2
        for i in 1:Nh
            row = zeros(Tq, n_unknowns)
            row[idx_v(i)] = one(Tq)
            row[idx_s(i)] = -one(Tq)
            push_eq!(row, zero(Tq))
        end
        if layout_norm === :collocated && Nh >= 2
            row = zeros(Tq, n_unknowns)
            row[idx_s(2)] = one(Tq)
            row[idx_s(1)] = -convert(Tq, p + 1)
            push_eq!(row, zero(Tq))
        end
    end

    if accuracy_order == 4 && layout_norm === :collocated && Nh >= 4
        has_u32 = haskey(pair_to_idx, (2, 3))
        has_u52 = haskey(pair_to_idx, (3, 4))
        has_u32 && has_u52 ||
            throw(ArgumentError("Order-4 centered constraints require V offdiagonals (2,3) and (3,4)."))
        u32 = idx_u(pair_to_idx[(2, 3)])
        u52 = idx_u(pair_to_idx[(3, 4)])

        # (p+1)s0 = v1 - (1/8)u_{3/2} + (5/8)u_{5/2}
        row1 = zeros(Tq, n_unknowns)
        row1[idx_s(1)] = convert(Tq, p + 1)
        row1[idx_v(2)] = -one(Tq)
        row1[u32] = one(Tq) // 8
        row1[u52] = -(5 // 8)
        push_eq!(row1, zero(Tq))

        # v2 = v1 + (63/8)u_{3/2} - (27/8)u_{5/2}
        row2 = zeros(Tq, n_unknowns)
        row2[idx_v(3)] = one(Tq)
        row2[idx_v(2)] = -one(Tq)
        row2[u32] = -(63 // 8)
        row2[u52] = 27 // 8
        push_eq!(row2, zero(Tq))
    end

    # Scale normalization and far-field anchoring.
    row_norm_s = zeros(Tq, n_unknowns)
    row_norm_v = zeros(Tq, n_unknowns)
    row_norm_s[idx_s(norm_idx)] = one(Tq)
    row_norm_v[idx_v(norm_idx)] = one(Tq)
    push_eq!(row_norm_s, rp[norm_idx])
    push_eq!(row_norm_v, rp[norm_idx])
    for i in ff_start:Nh
        row_s = zeros(Tq, n_unknowns)
        row_v = zeros(Tq, n_unknowns)
        row_s[idx_s(i)] = one(Tq)
        row_v[idx_v(i)] = one(Tq)
        rhs_i = rp[i]
        push_eq!(row_s, rhs_i)
        push_eq!(row_v, rhs_i)
    end

    # Odd-divergence exactness constraints:
    # -(G^T(Vψ))_i = s_i (p+k) r_i^(k-1), ψ=r^k.
    rf = r_q
    for k in odd_degrees
        rhs_coeff = convert(Tq, p + k)
        psi = rf .^ k
        for i in rows
            row = zeros(Tq, n_unknowns)
            row[idx_s(i)] = -rhs_coeff * rf[i]^(k - 1)

            # Diagonal V entries.
            for j in 1:Nh
                gij = Gq[j, i]
                gij == zero(Tq) && continue
                row[idx_v(j)] += -gij * psi[j]
            end

            # Allowed off-diagonals in V.
            for (pair_idx, (a, b)) in enumerate(allowed_pairs)
                coeff = -(Gq[a, i] * psi[b] + Gq[b, i] * psi[a])
                coeff == zero(Tq) && continue
                row[idx_u(pair_idx)] += coeff
            end

            push_eq!(row, zero(Tq))
        end
    end

    # Add alternating quadrature exactness constraints until we match unknown count:
    # scalar mass first (even q), then vector mass (odd q), and alternate.
    # This helps close the linear system for split-mass unknowns.
    Rdom = rf[end]
    q_scalar = 0
    q_vector = 1
    use_scalar = true
    quadrature_constraints = NamedTuple[]
    while length(A_rows) < n_unknowns
        if use_scalar
            q = q_scalar
            row = zeros(Tq, n_unknowns)
            @inbounds for i in 1:Nh
                row[idx_s(i)] = rf[i]^q
            end
            rhs = Rdom^(q + p + 1) / convert(Tq, q + p + 1)
            push_eq!(row, rhs)
            push!(quadrature_constraints, (family = :scalar, degree = q))
            q_scalar += 2
        else
            q = q_vector
            row = zeros(Tq, n_unknowns)
            @inbounds for i in 1:Nh
                row[idx_v(i)] = rf[i]^q
            end
            @inbounds for (pair_idx, (a, b)) in enumerate(allowed_pairs)
                row[idx_u(pair_idx)] += rf[a]^q + rf[b]^q
            end
            rhs = Rdom^(q + p + 1) / convert(Tq, q + p + 1)
            push_eq!(row, rhs)
            push!(quadrature_constraints, (family = :vector, degree = q))
            q_vector += 2
        end
        use_scalar = !use_scalar
    end

    m = length(A_rows)
    A = m == 0 ? zeros(Tq, 0, n_unknowns) : reduce(vcat, (reshape(row, 1, :) for row in A_rows))
    bvec = collect(b_rows)

    x0 = zeros(Tq, n_unknowns)
    @inbounds for i in 1:Nh
        x0[idx_s(i)] = s_seed_resolved[i]
        x0[idx_v(i)] = v_seed_resolved[i]
    end
    @inbounds for i in 1:n_u
        x0[idx_u(i)] = u_seed[i]
    end

    λ = _as_big_rational(regularization)
    λ == zero(Tq) || throw(ArgumentError("Exact rational seed solve requires `regularization = 0`."))
    x = if m == 0
        copy(x0)
    else
        _solve_exact_linear_system(A, bvec)
    end

    sdiag = Vector{Tq}(undef, Nh)
    vdiag = Vector{Tq}(undef, Nh)
    uvals = Vector{Tq}(undef, n_u)
    @inbounds for i in 1:Nh
        sdiag[i] = x[idx_s(i)]
        vdiag[i] = x[idx_v(i)]
    end
    @inbounds for i in 1:n_u
        uvals[i] = x[idx_u(i)]
    end

    pd_eps = _as_big_rational(pd_epsilon)
    pd_eps < zero(Tq) && throw(ArgumentError("`pd_epsilon` must satisfy >= 0."))

    # Enforce strict SPD for S on the full diagonal.
    min_s = minimum(sdiag)
    if min_s <= pd_eps
        shift_s = (pd_eps - min_s) + (1 // 10^12)
        @inbounds for i in 1:Nh
            sdiag[i] += shift_s
        end
    end

    # Enforce strict SPD for V on the full matrix by diagonal shifting,
    # using Cholesky-based SPD checks.
    Vfull = zeros(Tq, Nh, Nh)
    @inbounds for i in 1:Nh
        Vfull[i, i] = vdiag[i]
    end
    @inbounds for (k, (a, b)) in enumerate(allowed_pairs)
        val = uvals[k]
        Vfull[a, b] = val
        Vfull[b, a] = val
    end
    if !is_spd(Matrix{Float64}(Vfull))
        shift_v = max(pd_eps, 1 // 10^12)
        spd_found = false
        for _ in 1:64
            Vtrial = copy(Vfull)
            @inbounds for i in 1:Nh
                Vtrial[i, i] += shift_v
            end
            if is_spd(Matrix{Float64}(Vtrial))
                @inbounds for i in 1:Nh
                    vdiag[i] += shift_v
                end
                Vfull = Vtrial
                spd_found = true
                break
            end
            shift_v *= 2
        end
        spd_found || throw(ArgumentError("Could not enforce SPD for V with diagonal shifts in seed split-mass solve."))
    end

    base = construct_split_mass_matrices(
                                         convert(Vector{T}, r);
                                         accuracy_order = accuracy_order,
                                         p = p,
                                         layout = layout_norm,
                                         s_diag = [convert(T, si) for si in sdiag],
                                         v_diag = [convert(T, vi) for vi in vdiag],
                                         v_offdiag = [convert(T, ui) for ui in uvals],
                                         origin_bandwidth = origin_bandwidth,
                                         strict = strict,
                                         snap_factor = snap_factor
                                        )

    residual_max = if m == 0
        zero(Tq)
    else
        _split_mass_maxabs(abs.(A * x .- bvec))
    end

    s_pos_min = minimum(sdiag)
    v_active_min_eig = minimum(eigen(Symmetric(Matrix{Float64}(base.V))).values)
    v_is_spd = is_spd(Matrix{Float64}(base.V))

    return (
            base...,
            solver = :seed_linearized,
            enforced_rows = rows,
            odd_degrees = odd_degrees,
            quadrature_constraints = quadrature_constraints,
            regularization = λ,
            pd_epsilon = pd_eps,
            linear_residual_max = residual_max,
            s_pos_min = s_pos_min,
            v_active_min_eigenvalue = v_active_min_eig,
            v_is_spd = v_is_spd
           )
end

function construct_split_mass_matrices_optimization(r::AbstractVector,
                                                    G::AbstractMatrix;
                                                    accuracy_order::Int,
                                                    p::Int,
                                                    layout::Symbol = :collocated,
                                                    boundary_count::Union{Nothing, Int} = nothing,
                                                    enforced_rows::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                                                    normalization_index::Union{Nothing, Int} = nothing,
                                                    farfield_points::Int = 2,
                                                    origin_bandwidth::Union{Nothing, Int} = nothing,
                                                    s_seed::Union{Nothing, AbstractVector} = nothing,
                                                    v_seed::Union{Nothing, AbstractVector} = nothing,
                                                    v_offdiag_seed::Union{Nothing, AbstractVector} = nothing,
                                                    regularization::Real = 0.0,
                                                    pd_epsilon::Real = 1.0e-12,
                                                    strict::Bool = true,
                                                    snap_factor::Float64 = 64.0,
                                                    max_iter::Int = 5_000,
                                                    tolerance::Real = 1.0e-10,
                                                    print_level::Int = 0)
    _split_mass_supported_order(accuracy_order) ||
        throw(ArgumentError("Only accuracy orders 2/4/6/8 are supported; got $accuracy_order."))
    layout_norm = _split_mass_layout(layout)
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))

    Nh = length(r)
    Nh >= 1 || throw(ArgumentError("`r` must be non-empty."))
    size(G, 1) == Nh == size(G, 2) ||
        throw(DimensionMismatch("`G` and `r` dimensions are incompatible."))

    T = promote_type(eltype(r), eltype(G))
    Tf = Float64
    rf = Tf[convert(Tf, ri) for ri in r]

    Gs = sparse(Tf.(G))

    s_seed_resolved = if isnothing(s_seed)
        _split_mass_default_profile(rf, p)
    else
        Tf[convert(Tf, si) for si in s_seed]
    end
    v_seed_resolved = if isnothing(v_seed)
        _split_mass_default_profile(rf, p)
    else
        Tf[convert(Tf, vi) for vi in v_seed]
    end
    length(s_seed_resolved) == Nh ||
        throw(DimensionMismatch("`s_seed` must have length $Nh."))
    length(v_seed_resolved) == Nh ||
        throw(DimensionMismatch("`v_seed` must have length $Nh."))

    if isnothing(enforced_rows)
        if isnothing(boundary_count)
            closures = _infer_fd_closure_from_gradient(Matrix{Tf}(Gs); atol = zero(Tf))
            b = max(closures.left, closures.right)
            row_start = b + 1
            row_end = Nh - b
            rows = row_start <= row_end ? collect(row_start:row_end) : Int[]
        else
            b = Int(boundary_count)
            b >= 0 || throw(ArgumentError("`boundary_count` must satisfy >= 0."))
            row_start = b + 1
            row_end = Nh - b
            rows = row_start <= row_end ? collect(row_start:row_end) : Int[]
        end
    else
        rows = copy(Int.(enforced_rows))
    end

    if !isempty(rows)
        any(i -> i < 1 || i > Nh, rows) &&
            throw(ArgumentError("`enforced_rows` must lie in 1:$Nh."))
    end

    allowed_pairs = _split_mass_allowed_v_pairs(
                                                accuracy_order,
                                                Nh,
                                                layout_norm;
                                                origin_bandwidth = origin_bandwidth
                                               )
    u_seed = if isnothing(v_offdiag_seed)
        fill(zero(Tf), length(allowed_pairs))
    else
        length(v_offdiag_seed) == length(allowed_pairs) ||
            throw(DimensionMismatch("`v_offdiag_seed` must have length $(length(allowed_pairs))."))
        Tf[convert(Tf, ui) for ui in v_offdiag_seed]
    end

    n_s = Nh
    n_v = Nh
    n_u = length(allowed_pairs)
    n_unknowns = n_s + n_v + n_u
    idx_s(i) = i
    idx_v(i) = n_s + i
    idx_u(k) = n_s + n_v + k

    odd_degrees = _split_mass_monomial_degrees(accuracy_order)
    rp = rf .^ p
    norm_idx = isnothing(normalization_index) ? Nh : clamp(Int(normalization_index), 1, Nh)
    ff_count = max(0, farfield_points)
    ff_start = max(1, Nh - ff_count + 1)
    pair_to_idx = Dict{Tuple{Int, Int}, Int}()
    @inbounds for (k, pair) in enumerate(allowed_pairs)
        pair_to_idx[pair] = k
    end

    A_rows = Vector{Vector{Tf}}()
    b_rows = Tf[]

    function push_eq!(row, rhs)
        push!(A_rows, row)
        push!(b_rows, rhs)
        return nothing
    end

    if accuracy_order == 2
        for i in 1:Nh
            row = zeros(Tf, n_unknowns)
            row[idx_v(i)] = one(Tf)
            row[idx_s(i)] = -one(Tf)
            push_eq!(row, zero(Tf))
        end
        if layout_norm === :collocated && Nh >= 2
            row = zeros(Tf, n_unknowns)
            row[idx_s(2)] = one(Tf)
            row[idx_s(1)] = -convert(Tf, p + 1)
            push_eq!(row, zero(Tf))
        end
    end

    if accuracy_order == 4 && layout_norm === :collocated && Nh >= 4
        has_u32 = haskey(pair_to_idx, (2, 3))
        has_u52 = haskey(pair_to_idx, (3, 4))
        has_u32 && has_u52 ||
            throw(ArgumentError("Order-4 centered constraints require V offdiagonals (2,3) and (3,4)."))
        u32 = idx_u(pair_to_idx[(2, 3)])
        u52 = idx_u(pair_to_idx[(3, 4)])

        row1 = zeros(Tf, n_unknowns)
        row1[idx_s(1)] = convert(Tf, p + 1)
        row1[idx_v(2)] = -one(Tf)
        row1[u32] = one(Tf) / 8
        row1[u52] = -(5 / 8)
        push_eq!(row1, zero(Tf))

        row2 = zeros(Tf, n_unknowns)
        row2[idx_v(3)] = one(Tf)
        row2[idx_v(2)] = -one(Tf)
        row2[u32] = -(63 / 8)
        row2[u52] = 27 / 8
        push_eq!(row2, zero(Tf))
    end

    row_norm_s = zeros(Tf, n_unknowns)
    row_norm_v = zeros(Tf, n_unknowns)
    row_norm_s[idx_s(norm_idx)] = one(Tf)
    row_norm_v[idx_v(norm_idx)] = one(Tf)
    push_eq!(row_norm_s, rp[norm_idx])
    push_eq!(row_norm_v, rp[norm_idx])
    for i in ff_start:Nh
        row_s = zeros(Tf, n_unknowns)
        row_v = zeros(Tf, n_unknowns)
        row_s[idx_s(i)] = one(Tf)
        row_v[idx_v(i)] = one(Tf)
        rhs_i = rp[i]
        push_eq!(row_s, rhs_i)
        push_eq!(row_v, rhs_i)
    end

    for k in odd_degrees
        rhs_coeff = convert(Tf, p + k)
        psi = rf .^ k
        for i in rows
            row = zeros(Tf, n_unknowns)
            row[idx_s(i)] = -rhs_coeff * rf[i]^(k - 1)

            for j in 1:Nh
                gij = Gs[j, i]
                gij == zero(Tf) && continue
                row[idx_v(j)] += -gij * psi[j]
            end

            for (pair_idx, (a, b)) in enumerate(allowed_pairs)
                coeff = -(Gs[a, i] * psi[b] + Gs[b, i] * psi[a])
                coeff == zero(Tf) && continue
                row[idx_u(pair_idx)] += coeff
            end

            push_eq!(row, zero(Tf))
        end
    end

    Rdom = rf[end]
    q_scalar = 0
    q_vector = 1
    use_scalar = true
    quadrature_constraints = NamedTuple[]
    while length(A_rows) < n_unknowns
        if use_scalar
            q = q_scalar
            row = zeros(Tf, n_unknowns)
            @inbounds for i in 1:Nh
                row[idx_s(i)] = rf[i]^q
            end
            rhs = Rdom^(q + p + 1) / convert(Tf, q + p + 1)
            push_eq!(row, rhs)
            push!(quadrature_constraints, (family = :scalar, degree = q))
            q_scalar += 2
        else
            q = q_vector
            row = zeros(Tf, n_unknowns)
            @inbounds for i in 1:Nh
                row[idx_v(i)] = rf[i]^q
            end
            @inbounds for (pair_idx, (a, b)) in enumerate(allowed_pairs)
                row[idx_u(pair_idx)] += rf[a]^q + rf[b]^q
            end
            rhs = Rdom^(q + p + 1) / convert(Tf, q + p + 1)
            push_eq!(row, rhs)
            push!(quadrature_constraints, (family = :vector, degree = q))
            q_vector += 2
        end
        use_scalar = !use_scalar
    end

    m = length(A_rows)
    A = m == 0 ? zeros(Tf, 0, n_unknowns) : reduce(vcat, (reshape(row, 1, :) for row in A_rows))
    bvec = collect(b_rows)

    x0 = zeros(Tf, n_unknowns)
    @inbounds for i in 1:Nh
        x0[idx_s(i)] = s_seed_resolved[i]
        x0[idx_v(i)] = v_seed_resolved[i]
    end
    @inbounds for i in 1:n_u
        x0[idx_u(i)] = u_seed[i]
    end

    λ = convert(Tf, regularization)
    λ >= zero(Tf) || throw(ArgumentError("`regularization` must satisfy >= 0."))
    pd_eps = convert(Tf, pd_epsilon)
    pd_eps >= zero(Tf) || throw(ArgumentError("`pd_epsilon` must satisfy >= 0."))
    tolerance_f = convert(Tf, tolerance)
    tolerance_f > zero(Tf) || throw(ArgumentError("`tolerance` must be > 0."))
    max_iter >= 1 || throw(ArgumentError("`max_iter` must satisfy >= 1."))

    model = GenericModel{Tf}(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol", tolerance_f)
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "print_level", print_level)

    @variable(model, x[1:n_unknowns])
    for j in 1:n_unknowns
        set_start_value(x[j], x0[j])
    end
    if m > 0
        @variable(model, resid[1:m])
        for eq_idx in 1:m
            @constraint(model, sum(A[eq_idx, j] * x[j] for j in 1:n_unknowns) - bvec[eq_idx] == resid[eq_idx])
        end
        @objective(
                   model,
                   Min,
                   sum(resid[eq_idx]^2 for eq_idx in 1:m) +
                   sum((x[j] - x0[j])^2 for j in 1:n_unknowns) +
                   λ * sum((x[idx_u(j)])^2 for j in 1:n_u)
                  )
    else
        @objective(
                   model,
                   Min,
                   sum((x[j] - x0[j])^2 for j in 1:n_unknowns) +
                   λ * sum((x[idx_u(j)])^2 for j in 1:n_u)
                  )
    end

    optimize!(model)
    term = termination_status(model)
    term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED, MOI.FEASIBLE_POINT) ||
        throw(ArgumentError("JuMP split-mass optimization failed with termination status `$term`."))

    xsol = [convert(Tf, value(x[j])) for j in 1:n_unknowns]

    sdiag = Vector{Tf}(undef, Nh)
    vdiag = Vector{Tf}(undef, Nh)
    uvals = Vector{Tf}(undef, n_u)
    @inbounds for i in 1:Nh
        sdiag[i] = xsol[idx_s(i)]
        vdiag[i] = xsol[idx_v(i)]
    end
    @inbounds for i in 1:n_u
        uvals[i] = xsol[idx_u(i)]
    end

    min_s = minimum(sdiag)
    if min_s <= pd_eps
        shift_s = (pd_eps - min_s) + 1.0e-12
        @inbounds for i in 1:Nh
            sdiag[i] += shift_s
        end
    end

    Vfull = zeros(Tf, Nh, Nh)
    @inbounds for i in 1:Nh
        Vfull[i, i] = vdiag[i]
    end
    @inbounds for (k, (a, b)) in enumerate(allowed_pairs)
        val = uvals[k]
        Vfull[a, b] = val
        Vfull[b, a] = val
    end
    if !is_spd(Vfull)
        shift_v = max(pd_eps, 1.0e-12)
        spd_found = false
        for _ in 1:64
            Vtrial = copy(Vfull)
            @inbounds for i in 1:Nh
                Vtrial[i, i] += shift_v
            end
            if is_spd(Vtrial)
                @inbounds for i in 1:Nh
                    vdiag[i] += shift_v
                end
                Vfull = Vtrial
                spd_found = true
                break
            end
            shift_v *= 2
        end
        spd_found || throw(ArgumentError("Could not enforce SPD for V with diagonal shifts in optimization split-mass solve."))
    end

    base = construct_split_mass_matrices(
                                         convert(Vector{T}, r);
                                         accuracy_order = accuracy_order,
                                         p = p,
                                         layout = layout_norm,
                                         s_diag = [convert(T, si) for si in sdiag],
                                         v_diag = [convert(T, vi) for vi in vdiag],
                                         v_offdiag = [convert(T, ui) for ui in uvals],
                                         origin_bandwidth = origin_bandwidth,
                                         strict = strict,
                                         snap_factor = snap_factor
                                        )

    residual_max = if m == 0
        zero(Tf)
    else
        _split_mass_maxabs(abs.(A * xsol .- bvec))
    end

    s_pos_min = minimum(sdiag)
    v_active_min_eig = minimum(eigen(Symmetric(Matrix{Float64}(base.V))).values)
    v_is_spd = is_spd(Matrix{Float64}(base.V))

    return (
            base...,
            solver = :jump_optimization,
            termination_status = term,
            enforced_rows = rows,
            odd_degrees = odd_degrees,
            quadrature_constraints = quadrature_constraints,
            regularization = λ,
            pd_epsilon = pd_eps,
            linear_residual_max = residual_max,
            s_pos_min = s_pos_min,
            v_active_min_eigenvalue = v_active_min_eig,
            v_is_spd = v_is_spd
           )
end

"""
    construct_split_mass_matrices_cddlib(r, G; kwargs...)

Construct split masses `S` and `V` by solving linear constraints with
`CDDLib.Optimizer{Rational{BigInt}}` on a rational grid.

This path enforces diagonal `S` and supports order/layout-dependent near-origin
off-diagonal couplings in `V` through linear variables. Odd-monomial divergence
exactness is imposed on interior rows selected by `boundary_count`/`enforced_rows`.
"""
function construct_split_mass_matrices_cddlib(r::AbstractVector,
                                              G::AbstractMatrix;
                                              accuracy_order::Int,
                                              p::Int,
                                              layout::Symbol = :collocated,
                                              boundary_count::Union{Nothing, Int} = nothing,
                                              enforced_rows::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                                              normalization_index::Union{Nothing, Int} = nothing,
                                              farfield_points::Int = 2,
                                              eps_positive::Real = 1 // 10^12,
                                              force_diagonal_v::Bool = false,
                                              origin_bandwidth::Union{Nothing, Int} = nothing,
                                              enforce_v_offdiag_bounds::Bool = true,
                                              add_slack::Bool = true)
    _split_mass_supported_order(accuracy_order) ||
        throw(ArgumentError("Only accuracy orders 2/4/6/8 are supported; got $accuracy_order."))
    layout_norm = _split_mass_layout(layout)
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))

    Nh = length(r)
    Nh >= 1 || throw(ArgumentError("`r` must be non-empty."))
    size(G, 1) == Nh == size(G, 2) ||
        throw(DimensionMismatch("`G` and `r` dimensions are incompatible."))

    Tq = Rational{BigInt}
    eps_q = _as_big_rational(eps_positive)
    r_q = Tq[_as_big_rational(ri) for ri in r]

    Gs = sparse(G)
    I, J, V = findnz(Gs)
    Vq = Tq[_as_big_rational(vij) for vij in V]
    Gq = sparse(I, J, Vq, Nh, Nh)

    left_closure = 0
    right_closure = 0
    if isnothing(boundary_count)
        closures = _infer_fd_closure_from_gradient(Matrix{Tq}(Gq); atol = zero(Tq))
        b = max(closures.left, closures.right)
        left_closure = b
        right_closure = b
    else
        b = Int(boundary_count)
        b >= 0 || throw(ArgumentError("`boundary_count` must satisfy >= 0."))
        left_closure = b
        right_closure = b
    end

    rows = if isnothing(enforced_rows)
        start_row = left_closure + 1
        end_row = Nh - right_closure
        start_row <= end_row ? collect(start_row:end_row) : Int[]
    else
        copy(Int.(enforced_rows))
    end
    if !isempty(rows)
        any(i -> i < 1 || i > Nh, rows) &&
            throw(ArgumentError("`enforced_rows` must lie in 1:$Nh."))
    end

    rp = r_q .^ p
    norm_idx = isnothing(normalization_index) ? Nh : clamp(Int(normalization_index), 1, Nh)
    max_ff = max(0, farfield_points)
    ff_start = max(1, Nh - max_ff + 1)
    odd_degrees = _split_mass_monomial_degrees(accuracy_order)
    allowed_pairs = force_diagonal_v ?
                    Tuple{Int, Int}[] :
                    _split_mass_allowed_v_pairs(
                                                accuracy_order,
                                                Nh,
                                                layout_norm;
                                                origin_bandwidth = origin_bandwidth
                                               )

    model = GenericModel{Tq}(CDDLib.Optimizer{Tq})

    @variable(model, s[1:Nh])
    @variable(model, v[1:Nh])
    if !isempty(allowed_pairs)
        @variable(model, u[1:length(allowed_pairs)])
        if enforce_v_offdiag_bounds
            for idx in eachindex(allowed_pairs)
                a, b = allowed_pairs[idx]
                @constraint(model, u[idx] <= v[a])
                @constraint(model, -u[idx] <= v[a])
                @constraint(model, u[idx] <= v[b])
                @constraint(model, -u[idx] <= v[b])
            end
        end
    end

    @constraint(model, s[1] == zero(Tq))
    @constraint(model, v[1] == zero(Tq))
    for i in 2:Nh
        @constraint(model, s[i] >= eps_q)
        @constraint(model, v[i] >= eps_q)
    end

    if accuracy_order == 2
        for i in 1:Nh
            @constraint(model, v[i] == s[i])
        end
    end

    @constraint(model, s[norm_idx] == rp[norm_idx])
    @constraint(model, v[norm_idx] == rp[norm_idx])
    for i in ff_start:Nh
        @constraint(model, s[i] == rp[i])
        @constraint(model, v[i] == rp[i])
    end

    n_eq = length(rows) * length(odd_degrees)
    slack = nothing
    if add_slack
        @variable(model, slack_var[1:n_eq] >= zero(Tq))
        slack = slack_var
        eq_idx = 0
        for k in odd_degrees
            for i in rows
                eq_idx += 1
                rhs_coeff = convert(Tq, p + k) * (r_q[i]^(k - 1))
                expr_diag = -sum(
                            Gq.nzval[ptr] * (r_q[Gq.rowval[ptr]]^k) * v[Gq.rowval[ptr]]
                            for ptr in Gq.colptr[i]:(Gq.colptr[i + 1] - 1)
                           )
                expr_offdiag = if isempty(allowed_pairs)
                    zero(Tq)
                else
                    -sum(
                         (Gq[a, i] * (r_q[b]^k) + Gq[b, i] * (r_q[a]^k)) * u[idx]
                         for (idx, (a, b)) in enumerate(allowed_pairs)
                        )
                end
                expr = expr_diag + expr_offdiag - rhs_coeff * s[i]
                @constraint(model, expr <= slack_var[eq_idx])
                @constraint(model, -expr <= slack_var[eq_idx])
            end
        end
        @objective(model, Min, sum(slack_var))
    else
        for k in odd_degrees
            for i in rows
                rhs_coeff = convert(Tq, p + k) * (r_q[i]^(k - 1))
                expr_offdiag = if isempty(allowed_pairs)
                    zero(Tq)
                else
                    -sum(
                         (Gq[a, i] * (r_q[b]^k) + Gq[b, i] * (r_q[a]^k)) * u[idx]
                         for (idx, (a, b)) in enumerate(allowed_pairs)
                        )
                end
                @constraint(
                            model,
                            -sum(
                                 Gq.nzval[ptr] * (r_q[Gq.rowval[ptr]]^k) * v[Gq.rowval[ptr]]
                                 for ptr in Gq.colptr[i]:(Gq.colptr[i + 1] - 1)
                                ) + expr_offdiag == rhs_coeff * s[i]
                           )
            end
        end
        @objective(model, Min, zero(Tq))
    end

    optimize!(model)
    term = termination_status(model)
    term in (MOI.OPTIMAL, MOI.FEASIBLE_POINT) ||
        throw(ArgumentError("CDDLib split-mass solve failed with termination status `$term`."))

    sdiag = Tq[value(s[i]) for i in 1:Nh]
    vdiag = Tq[value(v[i]) for i in 1:Nh]
    S = sparse(spdiagm(0 => sdiag))
    Vdiag = sparse(spdiagm(0 => vdiag))
    if !isempty(allowed_pairs)
        @inbounds for idx in eachindex(allowed_pairs)
            a, b = allowed_pairs[idx]
            uval = convert(Tq, value(u[idx]))
            if uval != 0 // 1
                Vdiag[a, b] = uval
                Vdiag[b, a] = uval
            end
        end
    end
    slack_max = if add_slack && n_eq > 0
        maximum(Tq[value(slack[i]) for i in 1:n_eq])
    else
        zero(Tq)
    end

    return (
            S = S,
            V = Vdiag,
            s_diag = sdiag,
            v_diag = vdiag,
            allowed_v_offdiagonals = allowed_pairs,
            layout = layout_norm,
            origin_bandwidth = isnothing(origin_bandwidth) ? _split_mass_origin_radius(accuracy_order) : Int(origin_bandwidth),
            accuracy_order = accuracy_order,
            p = p,
            solver = :CDDLib,
            termination_status = term,
            enforced_rows = rows,
            odd_degrees = odd_degrees,
            add_slack = add_slack,
            slack_max = slack_max
           )
end

@inline function _safe_inverse_diagonal_entry(si::T; atol::T) where {T <: AbstractFloat}
    return abs(si) <= atol ? zero(T) : inv(si)
end

@inline function _safe_inverse_diagonal_entry(si::T; atol::T) where {T <: Real}
    return si == zero(T) ? zero(T) : inv(si)
end

function _safe_inverse_diagonal(sdiag::Vector{T}; atol::T) where {T <: Real}
    invdiag = similar(sdiag)
    @inbounds for i in eachindex(sdiag)
        invdiag[i] = _safe_inverse_diagonal_entry(sdiag[i]; atol = atol)
    end
    return invdiag
end

@inline function _split_mass_maxabs(vec::AbstractVector{T}) where {T <: Real}
    isempty(vec) && return zero(T)
    m = zero(T)
    @inbounds for x in vec
        ax = abs(x)
        if ax > m
            m = ax
        end
    end
    return m
end

function _split_mass_monomial_degrees(accuracy_order::Int)
    Nmom = _split_mass_moment_count(accuracy_order)
    return collect(1:2:(2 * Nmom - 1))
end

function _infer_fd_closure_from_gradient(G::AbstractMatrix{T}; atol::T) where {T <: Real}
    n, m = size(G)
    n == m || throw(DimensionMismatch("Expected square gradient matrix for closure inference."))
    n == 0 && return (left = 0, right = 0)

    ref = clamp(fld(n, 2), 1, n)
    function row_pattern(i::Int)
        idx = Int[]
        @inbounds for j in 1:n
            abs(G[i, j]) > atol && push!(idx, j - i)
        end
        return idx
    end

    pref = row_pattern(ref)
    left = 0
    @inbounds for i in 1:n
        row_pattern(i) == pref || (left += 1; continue)
        break
    end

    right = 0
    @inbounds for i in n:-1:1
        row_pattern(i) == pref || (right += 1; continue)
        break
    end

    maxc = max(0, n - 1)
    return (left = min(left, maxc), right = min(right, maxc))
end

function _split_mass_origin_constraint_residuals(S::SparseMatrixCSC{T, Ti},
                                                 V::SparseMatrixCSC{T, Ti},
                                                 p::Int,
                                                 accuracy_order::Int,
                                                 layout::Symbol) where {T <: Real, Ti <: Integer}
    Nh = size(S, 1)
    rows = NamedTuple[]
    layout_norm = _split_mass_layout(layout)

    if layout_norm === :collocated
        if accuracy_order == 2 && Nh >= 2
            s0 = S[1, 1]
            s1 = S[2, 2]
            resid = s1 - convert(T, p + 1) * s0
            push!(rows, (name = :sbp2_centered_origin, residual = resid))
        elseif accuracy_order == 4 && Nh >= 4
            s0 = S[1, 1]
            v1 = V[2, 2]
            v2 = V[3, 3]
            u32 = V[2, 3]
            u52 = V[3, 4]
            c8 = convert(T, 8)
            c1 = convert(T, 1) / c8
            c5 = convert(T, 5) / c8
            c27 = convert(T, 27) / c8
            c63 = convert(T, 63) / c8
            r1 = convert(T, p + 1) * s0 - (v1 - c1 * u32 + c5 * u52)
            r2 = v2 - (v1 + c63 * u32 - c27 * u52)
            push!(rows, (name = :sbp4_centered_origin_eq1, residual = r1))
            push!(rows, (name = :sbp4_centered_origin_eq2, residual = r2))
        end
    end

    return rows
end

"""
    split_mass_constraint_report(S, V, G, r; kwargs...)

Evaluate optimization-oriented mass constraints and diagnostics for split masses.

Returns:
- `D_div = -S^{-1}G^TV` using safe diagonal inversion of `S`;
- monomial exactness residuals for odd degrees `k=1,3,...,2N-1` where `N=order/2`;
- origin constraints (for centered order 2/4);
- normalization/far-field residuals;
- structural and SPD checks.
"""
function split_mass_constraint_report(S::SparseMatrixCSC{TS, TiS},
                                      V::SparseMatrixCSC{TV, TiV},
                                      G::AbstractMatrix,
                                      r::Vector{TR};
                                      accuracy_order::Int,
                                      p::Int,
                                      layout::Symbol = :collocated,
                                      enforced_rows::Union{Nothing, Vector{Int}} = nothing,
                                      boundary_count::Union{Nothing, Int} = nothing,
                                      normalization_index::Union{Nothing, Int} = nothing,
                                      farfield_points::Int = 2,
                                      origin_bandwidth::Union{Nothing, Int} = nothing,
                                      atol = nothing) where {TS <: Real, TiS <: Integer, TV <: Real, TiV <: Integer, TR <: Real}
    Nh = length(r)
    size(S, 1) == Nh == size(S, 2) ||
        throw(DimensionMismatch("`S` and `r` dimensions are incompatible."))
    size(V, 1) == Nh == size(V, 2) ||
        throw(DimensionMismatch("`V` and `r` dimensions are incompatible."))
    size(G, 1) == Nh == size(G, 2) ||
        throw(DimensionMismatch("`G` and `r` dimensions are incompatible."))
    _split_mass_supported_order(accuracy_order) ||
        throw(ArgumentError("Only accuracy orders 2/4/6/8 are supported; got $accuracy_order."))

    T = promote_type(TS, TV, TR, eltype(G))
    atol_local = _resolve_atol(T, atol)
    layout_norm = _split_mass_layout(layout)

    Sdiag = _extract_diagonal(convert(SparseMatrixCSC{T, Int}, S); atol = atol_local)
    Vdense = convert(Matrix{T}, V)
    Vdiag = Vector{T}(undef, Nh)
    @inbounds for i in 1:Nh
        Vdiag[i] = Vdense[i, i]
    end
    Gmat = convert(Matrix{T}, G)
    Vmat = convert(Matrix{T}, V)

    left_closure = 0
    right_closure = 0
    if isnothing(boundary_count)
        closures = _infer_fd_closure_from_gradient(Gmat; atol = atol_local)
        b = max(closures.left, closures.right)
        left_closure = b
        right_closure = b
    else
        b = Int(boundary_count)
        b >= 0 || throw(ArgumentError("`boundary_count` must satisfy >= 0."))
        left_closure = b
        right_closure = b
    end

    if isnothing(enforced_rows)
        start_row = left_closure + 1
        end_row = Nh - right_closure
        rows = start_row <= end_row ? collect(start_row:end_row) : Int[]
    else
        rows = copy(enforced_rows)
    end

    if !isempty(rows)
        any(i -> i < 1 || i > Nh, rows) &&
            throw(ArgumentError("`enforced_rows` must lie in 1:$Nh."))
    else
        rows = Int[]
    end

    Sinv = _safe_inverse_diagonal(Sdiag; atol = atol_local)
    D_div = sparse(-spdiagm(0 => Sinv) * transpose(Gmat) * Vmat)

    odd_degrees = _split_mass_monomial_degrees(accuracy_order)
    monomial = NamedTuple[]
    for k in odd_degrees
        psi = r .^ k
        lhs = -(transpose(Gmat) * (Vmat * psi))
        rhs = Sdiag .* (convert(T, p + k) .* (r .^ (k - 1)))
        resid = lhs .- rhs
        max_enforced = isempty(rows) ? zero(T) : _split_mass_maxabs(view(resid, rows))
        push!(
              monomial,
              (
               degree = k,
               residual = resid,
               max_abs_full = _split_mass_maxabs(resid),
               max_abs_enforced = max_enforced
              )
             )
    end

    origin_constraints = _split_mass_origin_constraint_residuals(
                                                         convert(SparseMatrixCSC{T, Int}, S),
                                                         convert(SparseMatrixCSC{T, Int}, V),
                                                         p,
                                                         accuracy_order,
                                                         layout_norm
                                                        )

    norm_idx = isnothing(normalization_index) ? Nh : clamp(Int(normalization_index), 1, Nh)
    rp = r .^ p
    norm_rows = NamedTuple[]
    push!(
          norm_rows,
          (name = :normalization_s, index = norm_idx, residual = Sdiag[norm_idx] - rp[norm_idx])
         )
    max_ff = max(0, farfield_points)
    ff_start = max(1, Nh - max_ff + 1)
    for i in ff_start:Nh
        push!(norm_rows, (name = :farfield_s, index = i, residual = Sdiag[i] - rp[i]))
        push!(norm_rows, (name = :farfield_v, index = i, residual = Vdiag[i] - rp[i]))
    end

    allowed_pairs = _split_mass_allowed_v_pairs(
                                           accuracy_order,
                                           Nh,
                                           layout_norm;
                                           origin_bandwidth = origin_bandwidth
                                          )
    forbidden_offdiag_max = _split_mass_forbidden_offdiag_max(
                                                          convert(SparseMatrixCSC{T, Int}, V),
                                                          allowed_pairs
                                                         )
    symmetry_defect = _split_mass_symmetry_defect_max(convert(SparseMatrixCSC{T, Int}, V))

    s_pos_min = minimum(Sdiag)
    s_positive = s_pos_min > atol_local

    active_idx = collect(1:Nh)
    min_eig_active = if isempty(active_idx)
        convert(Float64, NaN)
    else
        Vactive = Matrix{Float64}(Vmat[active_idx, active_idx])
        minimum(eigen(Symmetric(Vactive)).values)
    end
    v_spd_active = if isempty(active_idx)
        false
    else
        is_spd(Matrix{Float64}(Vmat[active_idx, active_idx]))
    end

    sbp_identity_dense = Matrix(convert(SparseMatrixCSC{T, Int}, S) * D_div) + Matrix(transpose(Vmat * Gmat))
    sbp_identity = sparse(sbp_identity_dense)

    return (
            D_div = D_div,
            odd_degrees = odd_degrees,
            enforced_rows = rows,
            interior_region = (
                               left_closure = left_closure,
                               right_closure = right_closure,
                               row_start = isempty(rows) ? 0 : first(rows),
                               row_end = isempty(rows) ? 0 : last(rows)
                              ),
            monomial_exactness = monomial,
            origin_constraints = origin_constraints,
            normalization_constraints = norm_rows,
            structural = (
                          allowed_v_offdiagonals = allowed_pairs,
                          forbidden_offdiag_max = forbidden_offdiag_max,
                          symmetry_defect_max = symmetry_defect,
                          order2_v_minus_s_max = accuracy_order == 2 ? _split_mass_order2_consistency_error(Sdiag, Vdiag) : zero(T)
                         ),
            spd = (
                   s_positive = s_positive,
                   s_pos_min = s_pos_min,
                   v_active_indices = active_idx,
                   v_active_min_eigenvalue = min_eig_active,
                   v_spd_active = v_spd_active
                  ),
            sbp_identity = (
                            residual = sbp_identity,
                            max_abs = _split_mass_maxabs(abs.(findnz(sbp_identity)[3]))
                           )
           )
end
