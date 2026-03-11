"""
    sbp6_v_offdiag_pairs(N)

Return the ordered list of near-origin off-diagonal index pairs `(i,j)` used by the
SBP6 split vector-mass pattern.

Pattern (when indices exist):
- row 2: `(2,3)`, `(2,4)`
- row 3: `(3,4)`, `(3,5)`
- row 4: `(4,5)`, `(4,6)`
- row 5: `(5,6)`, `(5,7)`
- row 6: `(6,7)`
"""
function sbp6_v_offdiag_pairs(N::Integer)
    Nint = Int(N)
    Nint >= 1 || throw(ArgumentError("`N` must be positive."))

    pairs = Tuple{Int,Int}[]
    # Rows 2:5 carry distance-1 and distance-2 couplings.
    @inbounds for i in 2:min(5, Nint - 1)
        j1 = i + 1
        j1 <= Nint && push!(pairs, (i, j1))
        j2 = i + 2
        j2 <= Nint && push!(pairs, (i, j2))
    end
    # Row 6 carries only distance-1 coupling (6,7).
    (Nint >= 7) && push!(pairs, (6, 7))
    return pairs
end

"""
    sbp6_vector_mass(; N=10, v_symbol_prefix="v", s_symbol_prefix="s")

Construct a symbolic SBP6 vector-mass template with the requested sparsity pattern,
without providing numerical values.

Returns a `NamedTuple` with:
- `S`: dense diagonal `Matrix{Any}` template with entries `s_1,...,s_N`,
- `V`: dense `Matrix{Any}` template (`0`, `1`, and `Symbol` entries),
- `free_symbols`: unknowns in the order `[s_diag; v_diag; v_offdiag]`,
- `symbol_to_index`: map from symbol to unknown-vector index,
- `s_diag_symbols`: scalar-mass symbols `[s_1, ..., s_N]`,
- `v_diag_symbols`: diagonal unknowns `[v_2, ..., v_min(10,N)]`,
- `v_offdiag_symbols`: off-diagonal unknowns in `sbp6_v_offdiag_pairs(N)` order,
- `v_offdiag_pairs`: off-diagonal pair list,
- `s_tail_symbols`: tail symbols `[s_11, ..., s_N]` reused on `V` diagonal entries
  where `V[i,i] = S[i,i]`.
"""
function sbp6_vector_mass(; N::Integer=10,
    v_symbol_prefix::AbstractString="v",
    s_symbol_prefix::AbstractString="s")
    Nint = Int(N)
    Nint >= 1 || throw(ArgumentError("`N` must be positive."))

    # Scalar mass template (fully diagonal).
    s_diag_symbols = Symbol[]
    S = Matrix{Any}(undef, Nint, Nint)
    fill!(S, 0)
    @inbounds for i in 1:Nint
        sym = Symbol(s_symbol_prefix, "_", i)
        S[i, i] = sym
        push!(s_diag_symbols, sym)
    end

    # Vector mass template (SBP6 split structure near origin).
    V = Matrix{Any}(undef, Nint, Nint)
    fill!(V, 0)
    V[1, 1] = 1

    v_diag_symbols = Symbol[]
    block_end = min(10, Nint)
    @inbounds for i in 2:block_end
        sym = Symbol(v_symbol_prefix, "_", i)
        V[i, i] = sym
        push!(v_diag_symbols, sym)
    end

    v_offdiag_pairs = sbp6_v_offdiag_pairs(Nint)
    v_offdiag_symbols = Symbol[]
    @inbounds for (i, j) in v_offdiag_pairs
        sym = Symbol(v_symbol_prefix, "_", i, "_", j)
        V[i, j] = sym
        V[j, i] = sym
        push!(v_offdiag_symbols, sym)
    end

    s_tail_symbols = Symbol[]
    @inbounds for i in 11:Nint
        sym = s_diag_symbols[i]
        V[i, i] = sym
        push!(s_tail_symbols, sym)
    end

    free_symbols = vcat(s_diag_symbols, v_diag_symbols, v_offdiag_symbols)
    symbol_to_index = Dict{Symbol,Int}(sym => idx for (idx, sym) in enumerate(free_symbols))

    return (
        S=S,
        V=V,
        free_symbols=free_symbols,
        symbol_to_index=symbol_to_index,
        s_diag_symbols=s_diag_symbols,
        v_diag_symbols=v_diag_symbols,
        v_offdiag_symbols=v_offdiag_symbols,
        v_offdiag_pairs=v_offdiag_pairs,
        s_tail_symbols=s_tail_symbols,
    )
end

function _sbp6_extract_diagonal(S::SparseMatrixCSC{T,Ti}) where {T<:Real,Ti<:Integer}
    n, m = size(S)
    n == m || throw(DimensionMismatch("`S` must be square."))

    sdiag = fill(zero(T), n)
    max_offdiag = zero(T)
    I, J, V = findnz(S)
    @inbounds for k in eachindex(V)
        i = I[k]
        j = J[k]
        v = V[k]
        if i == j
            sdiag[i] = v
        else
            av = abs(v)
            if av > max_offdiag
                max_offdiag = av
            end
        end
    end

    if T <: AbstractFloat
        atol = _resolve_atol(T, nothing)
        tol = max(atol, T(256) * eps(T))
        max_offdiag <= tol ||
            throw(ArgumentError("`S` must be diagonal (max off-diagonal magnitude = $max_offdiag)."))
    else
        max_offdiag == zero(T) ||
            throw(ArgumentError("`S` must be exactly diagonal for non-floating arithmetic."))
    end

    return sdiag
end

"""
    sbp6_vector_mass(s_diag; v11=one(eltype(s_diag)),
                     v_diag_block=nothing, v_offdiag_block=nothing)

Construct the SBP6 split vector-mass matrix `V` from scalar diagonal entries `s_diag`.

Structure:
- `V[1,1] = v11`
- diagonal indices `2:min(10,N)` are set by `v_diag_block` when provided, otherwise
  they default to `s_diag[i]`
- diagonal indices `11:N` are always `s_diag[i]`
- off-diagonals are only on `sbp6_v_offdiag_pairs(N)` and are set by
  `v_offdiag_block` (defaults to zeros)

The order of `v_diag_block` is `[v[2], v[3], ..., v[min(10,N)]]`.
The order of `v_offdiag_block` matches `sbp6_v_offdiag_pairs(N)`.
"""
function sbp6_vector_mass(s_diag::AbstractVector{T};
    v11=one(T),
    v_diag_block::Union{Nothing,AbstractVector}=nothing,
    v_offdiag_block::Union{Nothing,AbstractVector}=nothing) where {T<:Real}
    N = length(s_diag)
    N >= 1 || throw(ArgumentError("`s_diag` must be non-empty."))

    vdiag = Vector{T}(undef, N)
    @inbounds for i in 1:N
        vdiag[i] = convert(T, s_diag[i])
    end
    vdiag[1] = convert(T, v11)

    block_end = min(10, N)
    block_len = max(0, block_end - 1)
    if !isnothing(v_diag_block)
        length(v_diag_block) == block_len ||
            throw(DimensionMismatch("`v_diag_block` must have length $block_len for N=$N."))
        @inbounds for k in 1:block_len
            vdiag[k+1] = convert(T, v_diag_block[k])
        end
    end

    pairs = sbp6_v_offdiag_pairs(N)
    offvals = if isnothing(v_offdiag_block)
        fill(zero(T), length(pairs))
    else
        length(v_offdiag_block) == length(pairs) ||
            throw(DimensionMismatch("`v_offdiag_block` must have length $(length(pairs)) for N=$N."))
        [convert(T, v_offdiag_block[k]) for k in eachindex(v_offdiag_block)]
    end

    V = spdiagm(0 => vdiag)
    @inbounds for (k, (i, j)) in enumerate(pairs)
        v = offvals[k]
        if v != zero(T)
            V[i, j] = v
            V[j, i] = v
        end
    end

    return V
end

"""
    sbp6_vector_mass(S::SparseMatrixCSC; kwargs...)

Convenience overload that extracts `s_diag` from a diagonal scalar-mass matrix `S`
and delegates to `sbp6_vector_mass(s_diag; kwargs...)`.
"""
function sbp6_vector_mass(S::SparseMatrixCSC{T,Ti};
    v11=one(T),
    v_diag_block::Union{Nothing,AbstractVector}=nothing,
    v_offdiag_block::Union{Nothing,AbstractVector}=nothing) where {T<:Real,Ti<:Integer}
    sdiag = _sbp6_extract_diagonal(S)
    return sbp6_vector_mass(
        sdiag;
        v11=v11,
        v_diag_block=v_diag_block,
        v_offdiag_block=v_offdiag_block
    )
end

"""
    sbp6_scalar_mass_gradient(source; accuracy_order=6, points=21, h=1,
                              N=points-1, R=h*(points-1), p=2, mode=SafeMode(),
                              build_matrix=:probe, atol=nothing)

Construct the folded half-grid gradient and diagonal scalar mass using the same
pipeline as the diagonal-mass method:
1. build Cartesian SBP derivative on `[-R, R]`,
2. fold to `[0, R]` using even parity,
3. form `S = Hcart_half * diag(r^p)`.

Default construction uses `h=1` and `points=21` on `[0,R]`, i.e. `N=20`, `R=20`.
The domain radius is converted to `Rational{BigInt}` for exact arithmetic.

Returns a `NamedTuple` containing at least `r`, `Geven`, `Godd`, `S`, and `Sdiag`.
"""
function sbp6_scalar_mass_gradient(source;
    accuracy_order::Int=6,
    points::Int=21,
    h::Real=1,
    N=points - 1,
    R=h * (points - 1),
    p::Int=2,
    mode=SafeMode(),
    build_matrix::Symbol=:probe,
    atol=nothing)
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    Nint = Int(N)
    Nint > 0 || throw(ArgumentError("`N` must be positive."))
    accuracy_order > 0 || throw(ArgumentError("`accuracy_order` must be positive."))
    points > 1 || throw(ArgumentError("`points` must be > 1."))

    Rq = _sbp6_as_big_rational(R)
    Dfull, xfull, Gfull, Hfull = _build_full_grid_objects(
        source;
        accuracy_order=accuracy_order,
        N=Nint,
        R=Rq,
        mode=mode,
        build_matrix=build_matrix
    )

    T = eltype(xfull)
    atol_use = _resolve_atol(T, atol)
    r, Rop, Eeven, Eodd = _build_folding_operators(xfull; atol=atol_use)

    Geven = sparse(Rop * Gfull * Eeven)
    Godd = sparse(Rop * Gfull * Eodd)

    half_factor = convert(T, 1) / convert(T, 2)
    Hcart_half = sparse(half_factor * (transpose(Eeven) * Hfull * Eeven))
    metric = spdiagm(0 => r .^ p)
    S = sparse(Hcart_half * metric)
    Sdiag = _sbp6_extract_diagonal(S)

    return (
        r=r,
        Geven=Geven,
        Godd=Godd,
        S=S,
        Sdiag=Sdiag,
        Dfull=Dfull,
        xfull=xfull,
        Gfull=Gfull,
        Hfull=Hfull,
        Hcart_half=Hcart_half,
        Rop=Rop,
        Eeven=Eeven,
        Eodd=Eodd,
        p=p,
        accuracy_order=accuracy_order,
        R=Rq,
        mode=mode,
        atol=atol_use,
        build_matrix=build_matrix
    )
end

@inline _sbp6_as_big_rational(x::Rational{BigInt}) = x
@inline _sbp6_as_big_rational(x::Integer) = big(x) // 1
@inline _sbp6_as_big_rational(x::Rational{<:Integer}) = big(numerator(x)) // big(denominator(x))

function _sbp6_as_big_rational(x::AbstractFloat)
    isfinite(x) || throw(ArgumentError("Cannot convert non-finite floating-point value `$x` to Rational{BigInt}."))
    return rationalize(BigInt, x)
end

function _sbp6_as_big_rational(x::Real)
    throw(ArgumentError("Could not convert value of type $(typeof(x)) to Rational{BigInt}."))
end

function _sbp6_maxabs(values)
    isempty(values) && return zero(eltype(values))
    m = zero(eltype(values))
    @inbounds for v in values
        av = abs(v)
        if av > m
            m = av
        end
    end
    return m
end

function _sbp6_is_symmetric(A::SparseMatrixCSC{T,Ti}; tol::Float64=1e-12) where {T<:Real,Ti<:Integer}
    if T <: AbstractFloat
        return norm(Matrix(A - transpose(A)), Inf) <= tol
    end
    return Matrix(A) == transpose(Matrix(A))
end

function _sbp6_is_pd(A::SparseMatrixCSC{T,Ti}) where {T<:Real,Ti<:Integer}
    n, m = size(A)
    n == m || return false
    try
        LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Matrix{Float64}(A)); check=true)
        return true
    catch
        return false
    end
end

"""
    sbp6_construct_divergence(S, V, Geven, r; p=2)

Construct divergence operator via

`D = S^{-1}(B - G^T V)`,

where `B[end,end] = r[end]^p`.
"""
function sbp6_construct_divergence(S::SparseMatrixCSC{T,Ti},
    V::SparseMatrixCSC{T,Ti},
    Geven::SparseMatrixCSC{T,Ti},
    r::AbstractVector;
    p::Int=2) where {T<:Real,Ti<:Integer}
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    N = length(r)
    size(S) == (N, N) || throw(DimensionMismatch("`S` must be $(N)x$(N)."))
    size(V) == (N, N) || throw(DimensionMismatch("`V` must be $(N)x$(N)."))
    size(Geven) == (N, N) || throw(DimensionMismatch("`Geven` must be $(N)x$(N)."))

    Sdiag = _sbp6_extract_diagonal(S)
    B = spzeros(T, N, N)
    B[end, end] = convert(T, r[end]^p)

    RHS = sparse(B - transpose(Geven) * V)
    I, J, RV = findnz(RHS)
    DV = Vector{T}(undef, length(RV))
    @inbounds for k in eachindex(RV)
        i = I[k]
        si = Sdiag[i]
        si == zero(T) && throw(ArgumentError("Encountered zero scalar mass entry S[$i,$i] while forming D = S^{-1}(B-G^T V)."))
        DV[k] = RV[k] / si
    end

    D = sparse(I, J, DV, N, N)
    return (D=D, B=B, RHS=RHS, Sdiag=Sdiag)
end

function _sbp6_constraint_rows(N::Int, closure_right::Int, first_rows_r5::Int)
    closure_clamped = clamp(closure_right, 0, N)
    rows_r = collect(1:N)
    rows_r3_end = max(0, N - closure_clamped)
    rows_r3 = rows_r3_end == 0 ? Int[] : collect(1:rows_r3_end)
    rows_r5 = collect(1:min(max(0, first_rows_r5), N))
    return (rows_r=rows_r, rows_r3=rows_r3, rows_r5=rows_r5, closure_right=closure_clamped)
end

function _sbp6_constraint_error(D::SparseMatrixCSC, r::AbstractVector, p::Int, degree::Int, rows::Vector{Int})
    isempty(rows) && return 0.0
    u = r .^ degree
    exact = (p + degree) .* (r .^ (degree - 1))
    num = D * u
    return maximum(abs.(Float64.(num[rows] .- exact[rows])))
end

"""
    sbp6_solve_accuracy_constraints(setup; boundary_closure=nothing, first_rows_r5=5,
                                    exact_solve=true, verbose=true)

Solve SBP6 split-mass linear constraints for unknown `S` (diagonal) and structured `V`
using:

- `D = S^{-1}(B - G^T V)`
- `D*r = 3` (everywhere, for `p=2`)
- `D*r^3 = 5r^2` (rows excluding right boundary closure)
- `D*r^5 = 7r^4` (first `first_rows_r5` rows)
- `ones' * S * ones = R^3/3` (for `p=2`)

`setup` is the output of `sbp6_scalar_mass_gradient`.
"""
function sbp6_solve_accuracy_constraints(setup::NamedTuple;
    boundary_closure::Union{Nothing,Int}=nothing,
    first_rows_r5::Int=5,
    exact_solve::Bool=true,
    verbose::Bool=true)
    hasproperty(setup, :r) || throw(ArgumentError("`setup` must include `r`."))
    hasproperty(setup, :Geven) || throw(ArgumentError("`setup` must include `Geven`."))
    hasproperty(setup, :p) || throw(ArgumentError("`setup` must include `p`."))
    hasproperty(setup, :R) || throw(ArgumentError("`setup` must include `R`."))

    r = setup.r
    Geven = setup.Geven
    p = Int(setup.p)
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    N = length(r)
    size(Geven) == (N, N) || throw(DimensionMismatch("`Geven` must be $(N)x$(N)."))

    closure_auto = _closure_diagnostics(Geven).closure_width_right
    closure_right = isnothing(boundary_closure) ? closure_auto : Int(boundary_closure)
    rows = _sbp6_constraint_rows(N, closure_right, first_rows_r5)

    tmpl = sbp6_vector_mass(N=N)
    n_s = length(tmpl.s_diag_symbols)
    n_vdiag = length(tmpl.v_diag_symbols)
    n_voff = length(tmpl.v_offdiag_symbols)
    n_unknowns = n_s + n_vdiag + n_voff

    idx_s(i) = i
    idx_vdiag(j) = n_s + (j - 1)     # j in 2:block_end
    idx_voff(k) = n_s + n_vdiag + k

    Tq = Rational{BigInt}
    rq = Tq[_sbp6_as_big_rational(ri) for ri in r]
    I, J, GV = findnz(sparse(Geven))
    Gq = sparse(I, J, Tq[_sbp6_as_big_rational(gij) for gij in GV], N, N)
    Rq = _sbp6_as_big_rational(setup.R)

    A_rows = Vector{Vector{Tq}}()
    b_rows = Tq[]
    tags = NamedTuple[]

    function push_eq!(row, rhs, family::Symbol, degree::Int, row_index::Int)
        push!(A_rows, row)
        push!(b_rows, rhs)
        push!(tags, (family=family, degree=degree, row=row_index))
        return nothing
    end

    pairs = tmpl.v_offdiag_pairs
    block_end = min(10, N)
    BNN = rq[end]^p

    function add_degree_constraints!(degree::Int, rows_idx::Vector{Int}, family::Symbol)
        isempty(rows_idx) && return nothing
        u = rq .^ degree
        coeff = convert(Tq, p + degree)
        @inbounds for i in rows_idx
            row = zeros(Tq, n_unknowns)
            exact_i = coeff * rq[i]^(degree - 1)

            # S contribution on RHS term: -s_i * exact_i
            row[idx_s(i)] -= exact_i

            # V tail entries that are tied to S diagonals (i >= 11): -G[j,i]*u[j] * s_j
            for j in 11:N
                gij = Gq[j, i]
                gij == 0 // 1 && continue
                row[idx_s(j)] += -(gij * u[j])
            end

            # V diagonal free block entries (j = 2..min(10,N))
            for j in 2:block_end
                gij = Gq[j, i]
                gij == 0 // 1 && continue
                row[idx_vdiag(j)] += -(gij * u[j])
            end

            # V off-diagonal free entries
            for (kpair, (a, b)) in enumerate(pairs)
                coeff_pair = -(Gq[a, i] * u[b] + Gq[b, i] * u[a])
                coeff_pair == 0 // 1 && continue
                row[idx_voff(kpair)] += coeff_pair
            end

            # Constant contribution from fixed V[1,1]=1 and boundary matrix B.
            constant_term = (i == N ? (BNN * u[i]) : (0 // 1)) - Gq[1, i] * u[1]
            rhs = -constant_term
            push_eq!(row, rhs, family, degree, i)
        end
        return nothing
    end

    # D*r = (p+1)
    add_degree_constraints!(1, rows.rows_r, :divergence)
    # D*r^3 = (p+3)r^2 excluding right boundary closure
    add_degree_constraints!(3, rows.rows_r3, :divergence)
    # D*r^5 = (p+5)r^4 on first rows
    add_degree_constraints!(5, rows.rows_r5, :divergence)

    # Quadrature constraint: ones' * S * ones = ∫ r^p dr = R^(p+1)/(p+1)
    row_q = zeros(Tq, n_unknowns)
    @inbounds for i in 1:N
        row_q[idx_s(i)] = 1 // 1
    end
    rhs_q = Rq^(p + 1) / convert(Tq, p + 1)
    push_eq!(row_q, rhs_q, :quadrature, 0, 0)

    m = length(A_rows)
    A = m == 0 ? zeros(Tq, 0, n_unknowns) : reduce(vcat, (reshape(row, 1, :) for row in A_rows))
    bvec = collect(b_rows)

    exact_solve || throw(ArgumentError("Only exact rational solve is supported; set `exact_solve=true`."))

    x_exact = _solve_exact_linear_system(A, bvec)
    residual_exact = m == 0 ? 0 // 1 : _sbp6_maxabs(abs.(A * x_exact .- bvec))

    Tm = Rational{BigInt}
    r_use = rq
    G_use = Gq
    sdiag = Tm[x_exact[idx_s(i)] for i in 1:N]
    v_diag_block = Tm[x_exact[idx_vdiag(j)] for j in 2:block_end]
    v_offdiag_block = Tm[x_exact[idx_voff(k)] for k in 1:n_voff]

    S = spdiagm(0 => sdiag)
    V = sbp6_vector_mass(
        sdiag;
        v11=one(Tm),
        v_diag_block=v_diag_block,
        v_offdiag_block=v_offdiag_block
    )
    div_data = sbp6_construct_divergence(
        S,
        V,
        G_use,
        r_use;
        p=p
    )
    D = div_data.D
    B = div_data.B

    err_r = _sbp6_constraint_error(D, r_use, p, 1, rows.rows_r)
    err_r3 = _sbp6_constraint_error(D, r_use, p, 3, rows.rows_r3)
    err_r5 = _sbp6_constraint_error(D, r_use, p, 5, rows.rows_r5)
    quad_val = sum(sdiag)
    quad_target = Rq^(p + 1) / convert(Tq, p + 1)
    quad_err = Float64(abs(quad_val - quad_target))

    s_sym = _sbp6_is_symmetric(S)
    v_sym = _sbp6_is_symmetric(V)
    s_pd = all(si -> si > 0 // 1, sdiag)
    v_pd = _sbp6_is_pd(V)

    if verbose
        println("SBP6 solve mode: exact")
        println("  equations = ", m, ", unknowns = ", n_unknowns, ", exact residual = ", residual_exact)
        println("  boundary closure (right) = ", rows.closure_right)
        println("  constraint max errors: D*r = ", err_r,
            ", D*r^3 = ", err_r3, ", D*r^5 = ", err_r5,
            ", quadrature = ", quad_err)
        println("  SPD checks: S symmetric = ", s_sym, ", S PD = ", s_pd,
            ", V symmetric = ", v_sym, ", V PD = ", v_pd)
    end

    return (
        mode=:exact,
        setup=setup,
        closure_right=rows.closure_right,
        row_sets=rows,
        template=tmpl,
        A=A,
        b=bvec,
        tags=tags,
        x=x_exact,
        residual=residual_exact,
        S=S,
        V=V,
        D=D,
        B=B,
        errors=(
            Dr=err_r,
            Dr3=err_r3,
            Dr5=err_r5,
            quadrature=quad_err
        ),
        spd=(
            S_symmetric=s_sym,
            S_positive_definite=s_pd,
            V_symmetric=v_sym,
            V_positive_definite=v_pd
        )
    )
end

"""
    sbp6_solve_accuracy_constraints(source; kwargs...)

Convenience overload that first builds folded SBP6 gradient/scalar-mass data using
`sbp6_scalar_mass_gradient` and then solves the linear split-mass constraints.
"""
function sbp6_solve_accuracy_constraints(source;
    accuracy_order::Int=6,
    points::Int=21,
    h::Real=1,
    N=points - 1,
    R=h * (points - 1),
    p::Int=2,
    mode=SafeMode(),
    build_matrix::Symbol=:probe,
    atol=nothing,
    boundary_closure::Union{Nothing,Int}=nothing,
    first_rows_r5::Int=5,
    exact_solve::Bool=true,
    verbose::Bool=true)
    setup = sbp6_scalar_mass_gradient(
        source;
        accuracy_order=accuracy_order,
        points=points,
        h=h,
        N=N,
        R=R,
        p=p,
        mode=mode,
        build_matrix=build_matrix,
        atol=atol
    )

    return sbp6_solve_accuracy_constraints(
        setup;
        boundary_closure=boundary_closure,
        first_rows_r5=first_rows_r5,
        exact_solve=exact_solve,
        verbose=verbose
    )
end

"""
    sbp6_operators(source, points; kwargs...)

Build non-diagonal-mass SBP6 operators by internally solving the split-mass linear
system, then return the operator set

`(D, G, S, V, B)`.

`G` is the folded even-parity gradient (`Geven`).
"""
function sbp6_operators(source,
    points::Integer;
    h::Real=1,
    accuracy_order::Int=6,
    p::Int=2,
    mode=SafeMode(),
    build_matrix::Symbol=:probe,
    atol=nothing,
    boundary_closure::Union{Nothing,Int}=nothing,
    first_rows_r5::Int=5,
    exact_solve::Bool=true,
    verbose::Bool=false)
    points_int = Int(points)
    points_int > 1 || throw(ArgumentError("`points` must be > 1."))

    solved = sbp6_solve_accuracy_constraints(
        source;
        accuracy_order=accuracy_order,
        points=points_int,
        h=h,
        p=p,
        mode=mode,
        build_matrix=build_matrix,
        atol=atol,
        boundary_closure=boundary_closure,
        first_rows_r5=first_rows_r5,
        exact_solve=exact_solve,
        verbose=verbose
    )

    return (
        D=solved.D,
        G=solved.setup.Geven,
        S=solved.S,
        V=solved.V,
        B=solved.B
    )
end
