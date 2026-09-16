using JuMP
import Ipopt

const _SBP8_MOI = JuMP.MOI

"""
    sbp8_v_offdiag_pairs(N; boundary_rows=16, boundary_bandwidth=7)

Return the ordered list of near-origin off-diagonal index pairs `(i,j)` used by the
SBP8 split vector-mass boundary block.

- `boundary_rows` selects the leading block size where off-diagonals may be nonzero.
- `boundary_bandwidth` must be odd (e.g. 5 for pentadiagonal, 7 for heptadiagonal).
"""
function sbp8_v_offdiag_pairs(N::Integer;
                              boundary_rows::Integer = 16,
                              boundary_bandwidth::Integer = 7)
    Nint = Int(N)
    Nint >= 1 || throw(ArgumentError("`N` must be positive."))

    bw = Int(boundary_bandwidth)
    bw >= 3 && isodd(bw) ||
        throw(ArgumentError("`boundary_bandwidth` must be an odd integer >= 3 (got $boundary_bandwidth)."))
    half_band = (bw - 1) ÷ 2

    nb = min(Int(boundary_rows), Nint)
    nb >= 1 || throw(ArgumentError("`boundary_rows` must be >= 1."))

    pairs = Tuple{Int, Int}[]
    @inbounds for i in 1:nb
        jmax = min(nb, i + half_band)
        for j in (i + 1):jmax
            push!(pairs, (i, j))
        end
    end
    return pairs
end

"""
    sbp8_vector_mass(; N=20, boundary_rows=16, boundary_bandwidth=7,
                     v_symbol_prefix="v", s_symbol_prefix="s")

Construct a symbolic SBP8 vector-mass template.

Structure:
- `S` is fully diagonal with symbols `s_1, ..., s_N`.
- `V` is symmetric, with variable diagonal and banded off-diagonal entries only on the
  leading `boundary_rows x boundary_rows` block.
- For `i > boundary_rows`, `V[i,i] = s_i`.
"""
function sbp8_vector_mass(;
                          N::Integer = 20,
                          boundary_rows::Integer = 16,
                          boundary_bandwidth::Integer = 7,
                          v_symbol_prefix::AbstractString = "v",
                          s_symbol_prefix::AbstractString = "s")
    Nint = Int(N)
    Nint >= 1 || throw(ArgumentError("`N` must be positive."))

    nb = min(Int(boundary_rows), Nint)
    nb >= 1 || throw(ArgumentError("`boundary_rows` must be >= 1."))

    S = Matrix{Any}(undef, Nint, Nint)
    fill!(S, 0)
    s_diag_symbols = Symbol[]
    @inbounds for i in 1:Nint
        sym = Symbol(s_symbol_prefix, "_", i)
        S[i, i] = sym
        push!(s_diag_symbols, sym)
    end

    V = Matrix{Any}(undef, Nint, Nint)
    fill!(V, 0)

    v_diag_symbols = Symbol[]
    @inbounds for i in 1:nb
        sym = Symbol(v_symbol_prefix, "_", i)
        V[i, i] = sym
        push!(v_diag_symbols, sym)
    end

    v_offdiag_pairs = sbp8_v_offdiag_pairs(Nint;
                                           boundary_rows = nb,
                                           boundary_bandwidth = boundary_bandwidth)
    v_offdiag_symbols = Symbol[]
    @inbounds for (i, j) in v_offdiag_pairs
        sym = Symbol(v_symbol_prefix, "_", i, "_", j)
        V[i, j] = sym
        V[j, i] = sym
        push!(v_offdiag_symbols, sym)
    end

    s_tail_symbols = Symbol[]
    if nb < Nint
        @inbounds for i in (nb + 1):Nint
            sym = s_diag_symbols[i]
            V[i, i] = sym
            push!(s_tail_symbols, sym)
        end
    end

    free_symbols = vcat(s_diag_symbols, v_diag_symbols, v_offdiag_symbols)
    symbol_to_index = Dict{Symbol, Int}(sym => idx
                                        for (idx, sym) in enumerate(free_symbols))

    return (S = S,
            V = V,
            free_symbols = free_symbols,
            symbol_to_index = symbol_to_index,
            s_diag_symbols = s_diag_symbols,
            v_diag_symbols = v_diag_symbols,
            v_offdiag_symbols = v_offdiag_symbols,
            v_offdiag_pairs = v_offdiag_pairs,
            s_tail_symbols = s_tail_symbols,
            boundary_rows = nb,
            boundary_bandwidth = Int(boundary_bandwidth))
end

function _sbp8_extract_diagonal(S::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
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
    sbp8_vector_mass(s_diag; boundary_rows=16, boundary_bandwidth=7,
                     v_boundary_diag=nothing, v_boundary_offdiag=nothing)

Construct numeric SBP8 vector-mass matrix `V` from scalar diagonal entries `s_diag`.

- `V[i,i] = v_boundary_diag[i]` for `i <= boundary_rows` (defaults to `s_diag[i]`).
- `V[i,i] = s_diag[i]` for `i > boundary_rows`.
- Off-diagonals are only on `sbp8_v_offdiag_pairs(...)`, ordered exactly as in that function.
"""
function sbp8_vector_mass(s_diag::AbstractVector{T};
                          boundary_rows::Integer = 16,
                          boundary_bandwidth::Integer = 7,
                          v_boundary_diag::Union{Nothing, AbstractVector} = nothing,
                          v_boundary_offdiag::Union{Nothing, AbstractVector} = nothing) where {T <:
                                                                                               Real}
    N = length(s_diag)
    N >= 1 || throw(ArgumentError("`s_diag` must be non-empty."))

    nb = min(Int(boundary_rows), N)
    nb >= 1 || throw(ArgumentError("`boundary_rows` must be >= 1."))

    vdiag = Vector{T}(undef, N)
    @inbounds for i in 1:N
        vdiag[i] = convert(T, s_diag[i])
    end

    if isnothing(v_boundary_diag)
        @inbounds for i in 1:nb
            vdiag[i] = convert(T, s_diag[i])
        end
    else
        length(v_boundary_diag) == nb ||
            throw(DimensionMismatch("`v_boundary_diag` must have length $nb for N=$N."))
        @inbounds for i in 1:nb
            vdiag[i] = convert(T, v_boundary_diag[i])
        end
    end

    pairs = sbp8_v_offdiag_pairs(N;
                                 boundary_rows = nb,
                                 boundary_bandwidth = boundary_bandwidth)

    offvals = if isnothing(v_boundary_offdiag)
        fill(zero(T), length(pairs))
    else
        length(v_boundary_offdiag) == length(pairs) ||
            throw(DimensionMismatch("`v_boundary_offdiag` must have length $(length(pairs)) for N=$N."))
        [convert(T, v_boundary_offdiag[k]) for k in eachindex(v_boundary_offdiag)]
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
    sbp8_vector_mass(S::SparseMatrixCSC; kwargs...)

Convenience overload that extracts `s_diag` from diagonal scalar mass `S` and
delegates to `sbp8_vector_mass(s_diag; kwargs...)`.
"""
function sbp8_vector_mass(S::SparseMatrixCSC{T, Ti};
                          boundary_rows::Integer = 16,
                          boundary_bandwidth::Integer = 7,
                          v_boundary_diag::Union{Nothing, AbstractVector} = nothing,
                          v_boundary_offdiag::Union{Nothing, AbstractVector} = nothing) where {
                                                                                               T <:
                                                                                               Real,
                                                                                               Ti <:
                                                                                               Integer
                                                                                               }
    sdiag = _sbp8_extract_diagonal(S)
    return sbp8_vector_mass(sdiag;
                            boundary_rows = boundary_rows,
                            boundary_bandwidth = boundary_bandwidth,
                            v_boundary_diag = v_boundary_diag,
                            v_boundary_offdiag = v_boundary_offdiag)
end

"""
    sbp8_scalar_mass_gradient(source; accuracy_order=8, points=21, h=1,
                              N=points-1, R=h*(points-1), p=2, mode=SafeMode(),
                              atol=nothing)

Construct folded half-grid gradient/mass setup for SBP8.

Returns a `NamedTuple` with at least `r`, `Geven`, `Godd`, `S`, `Sdiag`, `Hcart_half`.
"""
function sbp8_scalar_mass_gradient(source;
                                   accuracy_order::Int = 8,
                                   points::Int = 21,
                                   h::Real = 1,
                                   N = points - 1,
                                   R = h * (points - 1),
                                   p::Int = 2,
                                   mode = SafeMode(),
                                   atol = nothing)
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    Nint = Int(N)
    Nint > 0 || throw(ArgumentError("`N` must be positive."))
    accuracy_order > 0 || throw(ArgumentError("`accuracy_order` must be positive."))
    points > 1 || throw(ArgumentError("`points` must be > 1."))

    Rq = _sbp8_as_big_rational(R)
    Dfull, xfull, Gfull, Hfull = _build_full_grid_objects(source;
                                                          accuracy_order = accuracy_order,
                                                          N = Nint,
                                                          R = Rq,
                                                          mode = mode)

    T = eltype(xfull)
    atol_use = _resolve_atol(T, atol)
    r, Rop, Eeven, Eodd = _build_folding_operators(xfull; atol = atol_use)

    Geven = sparse(Rop * Gfull * Eeven)
    Godd = sparse(Rop * Gfull * Eodd)

    half_factor = convert(T, 1) / convert(T, 2)
    Hcart_half = sparse(half_factor * (transpose(Eeven) * Hfull * Eeven))
    metric = spdiagm(0 => r .^ p)
    S = sparse(Hcart_half * metric)
    Sdiag = _sbp8_extract_diagonal(S)

    return (r = r,
            Geven = Geven,
            Godd = Godd,
            S = S,
            Sdiag = Sdiag,
            Dfull = Dfull,
            xfull = xfull,
            Gfull = Gfull,
            Hfull = Hfull,
            Hcart_half = Hcart_half,
            Rop = Rop,
            Eeven = Eeven,
            Eodd = Eodd,
            p = p,
            accuracy_order = accuracy_order,
            R = Rq,
            mode = mode,
            atol = atol_use)
end

@inline _sbp8_as_big_rational(x::Rational{BigInt}) = x
@inline _sbp8_as_big_rational(x::Integer) = big(x) // 1
@inline _sbp8_as_big_rational(x::Rational{<:Integer}) = big(numerator(x)) //
                                                        big(denominator(x))

function _sbp8_as_big_rational(x::AbstractFloat)
    isfinite(x) ||
        throw(ArgumentError("Cannot convert non-finite floating-point value `$x` to Rational{BigInt}."))
    return rationalize(BigInt, x)
end

function _sbp8_as_big_rational(x::Real)
    throw(ArgumentError("Could not convert value of type $(typeof(x)) to Rational{BigInt}."))
end

function _sbp8_sparse_convert(A::SparseMatrixCSC{Ta, Ti},
                              ::Type{Tb}) where {Ta <: Real, Ti <: Integer, Tb <: Real}
    n, m = size(A)
    I, J, V = findnz(A)
    W = Vector{Tb}(undef, length(V))
    @inbounds for k in eachindex(V)
        W[k] = convert(Tb, V[k])
    end
    return sparse(I, J, W, n, m)
end

function _sbp8_maxabs(values)
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

function _sbp8_is_symmetric(A::SparseMatrixCSC{T, Ti};
                            tol::Float64 = 1.0e-12) where {T <: Real, Ti <: Integer}
    if T <: AbstractFloat
        return norm(Matrix(A - transpose(A)), Inf) <= tol
    end
    return Matrix(A) == transpose(Matrix(A))
end

function _sbp8_is_pd(A::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
    n, m = size(A)
    n == m || return false
    try
        LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Matrix{Float64}(A)); check = true)
        return true
    catch
        return false
    end
end

"""
    sbp8_construct_divergence(S, V, Geven, r; p=2)

Construct divergence operator via

`D = S^{-1}(B - G^T V)`,

where `B[end,end] = r[end]^p`.
"""
function sbp8_construct_divergence(S::SparseMatrixCSC{T, Ti},
                                   V::SparseMatrixCSC{T, Ti},
                                   Geven::SparseMatrixCSC{T, Ti},
                                   r::AbstractVector;
                                   p::Int = 2) where {T <: Real, Ti <: Integer}
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    N = length(r)
    size(S) == (N, N) || throw(DimensionMismatch("`S` must be $(N)x$(N)."))
    size(V) == (N, N) || throw(DimensionMismatch("`V` must be $(N)x$(N)."))
    size(Geven) == (N, N) || throw(DimensionMismatch("`Geven` must be $(N)x$(N)."))

    Sdiag = _sbp8_extract_diagonal(S)
    B = spzeros(T, N, N)
    B[end, end] = convert(T, r[end]^p)

    RHS = sparse(B - transpose(Geven) * V)
    I, J, RV = findnz(RHS)
    DV = Vector{T}(undef, length(RV))
    @inbounds for k in eachindex(RV)
        i = I[k]
        si = Sdiag[i]
        si == zero(T) &&
            throw(ArgumentError("Encountered zero scalar mass entry S[$i,$i] while forming D = S^{-1}(B-G^T V)."))
        DV[k] = RV[k] / si
    end

    D = sparse(I, J, DV, N, N)
    return (D = D, B = B, RHS = RHS, Sdiag = Sdiag)
end

function _sbp8_infer_right_boundary_closure(setup::NamedTuple, Geven::SparseMatrixCSC)
    candidates = Int[]

    if hasproperty(setup, :Dfull)
        inferred = _boundary_closure_width_from_operator(setup.Dfull)
        if !isnothing(inferred)
            push!(candidates, Int(inferred))
        end
    end

    pattern = _closure_diagnostics(Geven).closure_width_right
    push!(candidates, Int(pattern))

    isempty(candidates) &&
        throw(ArgumentError("Could not infer right-boundary closure width."))
    return max(0, maximum(candidates))
end

function _sbp8_constraint_rows(N::Int, closure_right::Int, first_rows_r7::Int)
    closure_clamped = clamp(closure_right, 0, N)
    rows_r = collect(1:N)
    rows_poly_end = max(0, N - closure_clamped)
    rows_r3 = rows_poly_end == 0 ? Int[] : collect(1:rows_poly_end)
    rows_r5 = rows_poly_end == 0 ? Int[] : collect(1:rows_poly_end)
    rows_r7 = collect(1:min(max(0, first_rows_r7), N))
    return (rows_r = rows_r,
            rows_r3 = rows_r3,
            rows_r5 = rows_r5,
            rows_r7 = rows_r7,
            closure_right = closure_clamped)
end

function _sbp8_constraint_error_metrics(D::SparseMatrixCSC, r::AbstractVector, p::Int,
                                        degree::Int, rows::Vector{Int})
    isempty(rows) && return (abs = 0.0, rel = 0.0)
    u = r .^ degree
    exact = (p + degree) .* (r .^ (degree - 1))
    num = D * u
    err = Float64.(num[rows] .- exact[rows])
    absmax = maximum(abs.(err))
    scale = max(1.0, maximum(abs.(Float64.(exact[rows]))))
    relmax = absmax / scale
    return (abs = absmax, rel = relmax)
end

function _sbp8_build_linear_constraint_system(r::Vector{Float64},
                                              G::Matrix{Float64},
                                              s_target::Vector{Float64},
                                              p::Int,
                                              rows,
                                              nb::Int,
                                              pairs::Vector{Tuple{Int, Int}},
                                              quad_target::Float64)
    N = length(r)
    npairs = length(pairs)
    nvar = 2 * nb + npairs

    idx_s(i) = i
    idx_vd(i) = nb + i
    idx_vo(k) = 2 * nb + k

    Arows = Vector{Vector{Float64}}()
    bvec = Float64[]

    Bnn = r[end]^p

    function push_row!(coeff::Vector{Float64}, rhs::Float64)
        push!(Arows, coeff)
        push!(bvec, rhs)
        return nothing
    end

    function add_degree_rows!(degree::Int, rows_idx::Vector{Int})
        isempty(rows_idx) && return nothing
        u = r .^ degree
        exact = (p + degree) .* (r .^ (degree - 1))

        @inbounds for i in rows_idx
            coeff = zeros(Float64, nvar)
            rhs = 0.0

            # Constant SBP boundary term.
            rhs -= (i == N ? Bnn * u[i] : 0.0)

            # Diagonal V terms.
            for a in 1:N
                g = G[a, i]
                g == 0.0 && continue
                c = -(g * u[a])
                if a <= nb
                    coeff[idx_vd(a)] += c
                else
                    rhs -= c * s_target[a]
                end
            end

            # Boundary off-diagonal V terms.
            for (kpair, (a, b)) in enumerate(pairs)
                c = -(G[a, i] * u[b] + G[b, i] * u[a])
                c == 0.0 && continue
                coeff[idx_vo(kpair)] += c
            end

            # Scalar-mass terms.
            csi = -exact[i]
            if i <= nb
                coeff[idx_s(i)] += csi
            else
                rhs -= csi * s_target[i]
            end

            push_row!(coeff, rhs)
        end
        return nothing
    end

    add_degree_rows!(1, rows.rows_r)
    add_degree_rows!(3, rows.rows_r3)
    add_degree_rows!(5, rows.rows_r5)
    add_degree_rows!(7, rows.rows_r7)

    # Quadrature constraint.
    coeff_q = zeros(Float64, nvar)
    @inbounds for i in 1:nb
        coeff_q[idx_s(i)] = 1.0
    end
    rhs_q = quad_target - (nb < N ? sum(s_target[(nb + 1):N]) : 0.0)
    push_row!(coeff_q, rhs_q)

    m = length(Arows)
    A = zeros(Float64, m, nvar)
    @inbounds for i in 1:m
        A[i, :] .= Arows[i]
    end
    return (A = A, b = bvec)
end

function _sbp8_polish_linear_constraints(x0::Vector{Float64},
                                         A::Matrix{Float64},
                                         b::Vector{Float64})
    isempty(b) && return (x = x0, residual = 0.0, changed = false, success = true)
    r = A * x0 .- b
    inf0 = maximum(abs.(r))
    inf0 == 0.0 && return (x = x0, residual = 0.0, changed = false, success = true)

    x = copy(x0)
    best_x = copy(x0)
    best_res = inf0

    # Rectangular QR solve gives a stable minimum-norm correction for A*δ = r.
    for _ in 1:3
        rcur = A * x .- b
        if maximum(abs.(rcur)) <= max(1.0e-15, eps(Float64))
            break
        end
        delta = try
            A \ rcur
        catch
            # Fallback if factorization fails.
            transpose(A) * (LinearAlgebra.pinv(A * transpose(A)) * rcur)
        end
        x .-= delta
        rnew = A * x .- b
        infn = maximum(abs.(rnew))
        if infn < best_res
            best_res = infn
            best_x .= x
        end
    end

    return (x = best_x, residual = best_res, changed = true,
            success = best_res <= inf0 + 1.0e-14)
end

function _sbp8_exact_default_s1_target(rq::Vector{Rational{BigInt}};
                                       factor::Rational{BigInt} = 11 // 20)
    length(rq) >= 2 ||
        throw(ArgumentError("Need at least two grid points to infer spacing for default S[1,1] target."))
    h = rq[2] - rq[1]
    h > 0 // 1 || throw(ArgumentError("Grid spacing must be positive."))
    return factor * h^3
end

function _sbp8_build_exact_constraint_system(setup::NamedTuple,
                                             rows;
                                             diag_free_indices::Vector{Int},
                                             v_offdiag_pairs::Vector{Tuple{Int, Int}},
                                             s1_target::Union{Nothing, Rational{BigInt}} = nothing)
    hasproperty(setup, :r) || throw(ArgumentError("`setup` must include `r`."))
    hasproperty(setup, :Geven) || throw(ArgumentError("`setup` must include `Geven`."))
    hasproperty(setup, :p) || throw(ArgumentError("`setup` must include `p`."))
    hasproperty(setup, :R) || throw(ArgumentError("`setup` must include `R`."))

    Tq = Rational{BigInt}
    rq = Tq[_sbp8_as_big_rational(ri) for ri in setup.r]
    N = length(rq)
    p = Int(setup.p)

    I, J, GV = findnz(sparse(setup.Geven))
    Gq = sparse(I, J, Tq[_sbp8_as_big_rational(gij) for gij in GV], N, N)
    Rq = _sbp8_as_big_rational(setup.R)

    n_s = N
    n_vdiag = length(diag_free_indices)
    n_voff = length(v_offdiag_pairs)
    n_unknowns = n_s + n_vdiag + n_voff

    idx_s(i) = i
    vdiag_to_col = Dict{Int, Int}(i => (n_s + k) for (k, i) in enumerate(diag_free_indices))
    idx_voff(k) = n_s + n_vdiag + k

    A_rows = Vector{Vector{Tq}}()
    b_rows = Tq[]
    tags = NamedTuple[]

    function push_eq!(row, rhs, family::Symbol, degree::Int, row_index::Int)
        push!(A_rows, row)
        push!(b_rows, rhs)
        push!(tags, (family = family, degree = degree, row = row_index))
        return nothing
    end

    BNN = rq[end]^p
    function add_degree_constraints!(degree::Int, rows_idx::Vector{Int}, family::Symbol)
        isempty(rows_idx) && return nothing
        u = rq .^ degree
        exact = convert(Tq, p + degree) .* (rq .^ (degree - 1))

        @inbounds for i in rows_idx
            row = zeros(Tq, n_unknowns)

            # S contribution from -s_i * exact_i.
            row[idx_s(i)] -= exact[i]

            # V diagonal contribution.
            for j in 1:N
                gij = Gq[j, i]
                gij == 0 // 1 && continue
                c = -(gij * u[j])
                if haskey(vdiag_to_col, j)
                    row[vdiag_to_col[j]] += c
                else
                    row[idx_s(j)] += c
                end
            end

            # V off-diagonal free entries.
            for (kpair, (a, b)) in enumerate(v_offdiag_pairs)
                c = -(Gq[a, i] * u[b] + Gq[b, i] * u[a])
                c == 0 // 1 && continue
                row[idx_voff(kpair)] += c
            end

            # Constant contribution from B.
            rhs = -((i == N) ? (BNN * u[i]) : (0 // 1))
            push_eq!(row, rhs, family, degree, i)
        end
        return nothing
    end

    add_degree_constraints!(1, rows.rows_r, :divergence)
    add_degree_constraints!(3, rows.rows_r3, :divergence)
    add_degree_constraints!(5, rows.rows_r5, :divergence)
    add_degree_constraints!(7, rows.rows_r7, :divergence)

    # Quadrature constraint: sum(s_i) = R^(p+1)/(p+1).
    row_q = zeros(Tq, n_unknowns)
    @inbounds for i in 1:N
        row_q[idx_s(i)] = 1 // 1
    end
    rhs_q = Rq^(p + 1) / convert(Tq, p + 1)
    push_eq!(row_q, rhs_q, :quadrature, 0, 0)

    # Optional anchor to avoid singular S[1,1]=0 exact solutions.
    if !isnothing(s1_target)
        row_s1 = zeros(Tq, n_unknowns)
        row_s1[idx_s(1)] = 1 // 1
        push_eq!(row_s1, s1_target, :anchor, 0, 1)
    end

    m = length(A_rows)
    A = m == 0 ? zeros(Tq, 0, n_unknowns) :
        reduce(vcat, (reshape(row, 1, :) for row in A_rows))
    b = collect(b_rows)
    return (A = A,
            b = b,
            tags = tags,
            r = rq,
            G = Gq,
            N = N,
            p = p,
            diag_free_indices = diag_free_indices,
            v_offdiag_pairs = v_offdiag_pairs,
            indexing = (n_s = n_s,
                        n_vdiag = n_vdiag,
                        n_voff = n_voff,
                        idx_s = idx_s,
                        vdiag_to_col = vdiag_to_col,
                        idx_voff = idx_voff))
end

function _sbp8_solve_accuracy_constraints_exact(setup::NamedTuple;
                                                boundary_rows::Union{Nothing, Int} = nothing,
                                                boundary_bandwidth::Int = 7,
                                                boundary_closure::Union{Nothing, Int} = nothing,
                                                first_rows_r7::Int = 6,
                                                v_diag_free_end::Union{Nothing, Int} = nothing,
                                                s1_target::Union{Nothing, Real} = nothing,
                                                s1_target_factor::Float64 = 0.55,
                                                s_min::Float64 = 1.0e-12,
                                                enforce_fp_accuracy::Bool = true,
                                                floating_point_factor::Float64 = 5.0e4,
                                                enforce_real_negative_spectrum::Bool = true,
                                                spectrum_imag_tol::Float64 = 5.0e-6,
                                                spectrum_nonpositive_tol::Float64 = 1.0e-8,
                                                stability_check::Bool = true,
                                                verbose::Bool = true)
    hasproperty(setup, :r) || throw(ArgumentError("`setup` must include `r`."))
    hasproperty(setup, :Geven) || throw(ArgumentError("`setup` must include `Geven`."))
    hasproperty(setup, :p) || throw(ArgumentError("`setup` must include `p`."))
    s_min > 0 ||
        throw(ArgumentError("`s_min` must be strictly positive to keep S invertible."))

    Geven = sparse(setup.Geven)
    closure_auto = _sbp8_infer_right_boundary_closure(setup, Geven)
    closure_right = isnothing(boundary_closure) ? closure_auto : Int(boundary_closure)

    N = length(setup.r)
    rows = _sbp8_constraint_rows(N, closure_right, first_rows_r7)

    pair_rows_auto = min(N, 2 * closure_right)
    pair_rows = isnothing(boundary_rows) ? pair_rows_auto : clamp(Int(boundary_rows), 1, N)
    v_diag_end = isnothing(v_diag_free_end) ? min(N, max(pair_rows + 4, 20)) :
                 clamp(Int(v_diag_free_end), 1, N)
    diag_free_indices = collect(1:v_diag_end)

    pairs = sbp8_v_offdiag_pairs(N;
                                 boundary_rows = pair_rows,
                                 boundary_bandwidth = boundary_bandwidth)

    rq = Rational{BigInt}[_sbp8_as_big_rational(ri) for ri in setup.r]
    s1_q = if isnothing(s1_target)
        factor_q = _sbp8_as_big_rational(s1_target_factor)
        _sbp8_exact_default_s1_target(rq; factor = factor_q)
    else
        _sbp8_as_big_rational(s1_target)
    end

    sys = _sbp8_build_exact_constraint_system(setup,
                                              rows;
                                              diag_free_indices = diag_free_indices,
                                              v_offdiag_pairs = pairs,
                                              s1_target = s1_q)
    A = sys.A
    b = sys.b
    x_exact = _solve_exact_linear_system(A, b)
    residual_exact = isempty(b) ? (0 // 1) : _sbp8_maxabs(abs.(A * x_exact .- b))

    idx = sys.indexing
    n_s = idx.n_s
    n_vdiag = idx.n_vdiag
    n_voff = idx.n_voff

    sdiag_q = [x_exact[i] for i in 1:n_s]
    min_s = minimum(Float64.(sdiag_q))
    min_s > 0 ||
        throw(ArgumentError("Exact SBP8 solution violates strict positivity on S diagonal (min = $min_s)."))
    min_s > s_min ||
        throw(ArgumentError("Exact SBP8 solution violates `s_min` on S diagonal (min = $min_s, s_min = $s_min)."))
    s1_exact = Float64(sdiag_q[1])
    s1_exact > 0 ||
        throw(ArgumentError("Exact SBP8 solution violates strict positivity at origin: S[1,1] = $s1_exact."))
    s1_exact > s_min ||
        throw(ArgumentError("Exact SBP8 solution violates invertibility at origin: S[1,1] = $s1_exact <= s_min = $s_min."))

    vdiag_q = Vector{Rational{BigInt}}(undef, N)
    @inbounds for i in 1:N
        col = get(idx.vdiag_to_col, i, 0)
        if col > 0
            vdiag_q[i] = x_exact[col]
        else
            vdiag_q[i] = sdiag_q[i]
        end
    end
    voff_q = n_voff == 0 ? Rational{BigInt}[] : [x_exact[idx.idx_voff(k)] for k in 1:n_voff]

    S = spdiagm(0 => sdiag_q)
    V = spdiagm(0 => vdiag_q)
    @inbounds for (k, (i, j)) in enumerate(pairs)
        v = voff_q[k]
        if v != 0 // 1
            V[i, j] = v
            V[j, i] = v
        end
    end

    div_data = sbp8_construct_divergence(S, V, sys.G, sys.r; p = sys.p)
    D = div_data.D
    B = div_data.B

    err_r = _sbp8_constraint_error_metrics(D, sys.r, sys.p, 1, rows.rows_r)
    err_r3 = _sbp8_constraint_error_metrics(D, sys.r, sys.p, 3, rows.rows_r3)
    err_r5 = _sbp8_constraint_error_metrics(D, sys.r, sys.p, 5, rows.rows_r5)
    err_r7 = _sbp8_constraint_error_metrics(D, sys.r, sys.p, 7, rows.rows_r7)

    quad_target = _sbp8_as_big_rational(setup.R)^(sys.p + 1) / (sys.p + 1)
    quad_val = sum(sdiag_q)
    quad_err = Float64(abs(quad_val - quad_target))
    quad_scale = max(1.0, abs(Float64(quad_target)))
    quad_rel = quad_err / quad_scale

    s_sym = _sbp8_is_symmetric(S)
    v_sym = _sbp8_is_symmetric(V)
    s_pd = all(si -> si > 0 // 1, sdiag_q)

    Vb = Matrix{Float64}(V[1:pair_rows, 1:pair_rows])
    v_boundary_pd = try
        LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Vb); check = true)
        true
    catch
        false
    end
    v_pd = _sbp8_is_pd(V)

    L = sparse(D * sys.G)
    eigvals_L = stability_check || enforce_real_negative_spectrum ?
                _high_precision_schur_values(Matrix(L)) : Complex{Float64x4}[]
    max_real_L = isempty(eigvals_L) ? NaN : maximum(Float64.(real.(eigvals_L)))
    max_abs_imag_L = isempty(eigvals_L) ? NaN : maximum(Float64.(abs.(imag.(eigvals_L))))
    spectrum_ok = isempty(eigvals_L) ? true :
                  (max_abs_imag_L <= spectrum_imag_tol &&
                   max_real_L <= spectrum_nonpositive_tol)
    if enforce_real_negative_spectrum && !spectrum_ok
        throw(ArgumentError("SBP8 spectrum check failed (exact mode): max_abs_imag_eig=$(max_abs_imag_L), " *
                            "max_real_eig=$(max_real_L), thresholds=(imag<=$(spectrum_imag_tol), real<=$(spectrum_nonpositive_tol))."))
    end

    fp_tol = floating_point_factor * eps(Float64)
    fp_ok = (err_r.rel <= fp_tol) &&
            (err_r3.rel <= fp_tol) &&
            (err_r5.rel <= fp_tol) &&
            (err_r7.rel <= fp_tol) &&
            (quad_rel <= fp_tol)
    if enforce_fp_accuracy && !fp_ok
        throw(ArgumentError("SBP8 exact constraints are not at floating-point level after conversion. " *
                            "rel-errors: Dr=$(err_r.rel), Dr3=$(err_r3.rel), Dr5=$(err_r5.rel), " *
                            "Dr7=$(err_r7.rel), quadrature=$(quad_rel); threshold=$(fp_tol)."))
    end

    interface_jump = (pair_rows < N) ?
                     abs(Float64(vdiag_q[pair_rows] - sdiag_q[pair_rows + 1])) : 0.0

    constructor_rows = v_diag_end
    constructor_pairs = sbp8_v_offdiag_pairs(N;
                                             boundary_rows = constructor_rows,
                                             boundary_bandwidth = boundary_bandwidth)
    pair_value = Dict{Tuple{Int, Int}, Float64}()
    @inbounds for (k, pr) in enumerate(pairs)
        pair_value[pr] = Float64(voff_q[k])
    end
    v_boundary_offdiag_constructor = [get(pair_value, pr, 0.0) for pr in constructor_pairs]
    v_boundary_diag_constructor = Float64.(vdiag_q[1:constructor_rows])

    if verbose
        println("SBP8 solve mode: exact")
        println("  equations = ", size(A, 1), ", unknowns = ", size(A, 2),
                ", exact residual = ", residual_exact)
        println("  right closure = ", rows.closure_right,
                ", pair rows = ", pair_rows,
                ", vdiag free end = ", v_diag_end,
                ", boundary bandwidth = ", boundary_bandwidth,
                ", offdiag vars = ", n_voff)
        println("  S[1,1] target = ", s1_q, ", solved S[1,1] = ", sdiag_q[1])
        println("  constraint max errors (abs): D*r = ", err_r.abs,
                ", D*r^3 = ", err_r3.abs,
                ", D*r^5 = ", err_r5.abs,
                ", D*r^7 = ", err_r7.abs,
                ", quadrature = ", quad_err)
        println("  SPD checks: S symmetric = ", s_sym,
                ", S positive = ", s_pd,
                ", V symmetric = ", v_sym,
                ", V boundary PD = ", v_boundary_pd,
                ", V full PD = ", v_pd)
        if stability_check
            println("  stability: max real eig(D*G) = ", max_real_L,
                    ", max |imag eig(D*G)| = ", max_abs_imag_L,
                    ", spectral_ok = ", spectrum_ok)
        end
        println("  boundary/interface smoothness: |V[$pair_rows,$pair_rows] - S[$(pair_rows + 1),$(pair_rows + 1)]| = ",
                interface_jump)
    end

    return (mode = :exact,
            setup = setup,
            closure_right = rows.closure_right,
            boundary_rows = pair_rows,
            boundary_bandwidth = boundary_bandwidth,
            row_sets = rows,
            termination_status = :EXACT,
            primal_status = :EXACT,
            dual_status = :EXACT,
            locally_optimal = true,
            A = A,
            b = b,
            tags = sys.tags,
            x = x_exact,
            residual = residual_exact,
            S = S,
            V = V,
            D = D,
            B = B,
            L = L,
            max_real_eig_L = max_real_L,
            max_abs_imag_eig_L = max_abs_imag_L,
            spectrum = (enforce_real_negative_spectrum = enforce_real_negative_spectrum,
                        imag_tolerance = spectrum_imag_tol,
                        nonpositive_tolerance = spectrum_nonpositive_tol,
                        satisfied = spectrum_ok),
            errors = (Dr = err_r.abs,
                      Dr3 = err_r3.abs,
                      Dr5 = err_r5.abs,
                      Dr7 = err_r7.abs,
                      Dr_rel = err_r.rel,
                      Dr3_rel = err_r3.rel,
                      Dr5_rel = err_r5.rel,
                      Dr7_rel = err_r7.rel,
                      quadrature = quad_err,
                      quadrature_rel = quad_rel,
                      interface_jump = interface_jump),
            floating_point_accuracy = (factor = floating_point_factor,
                                       tolerance = fp_tol,
                                       satisfied = fp_ok),
            polish = (applied = false, accepted = false, residual_before = NaN,
                      residual_after = NaN),
            spd = (S_symmetric = s_sym,
                   S_positive_definite = s_pd,
                   V_symmetric = v_sym,
                   V_boundary_positive_definite = v_boundary_pd,
                   V_positive_definite = v_pd),
            coefficients = (s_boundary = Float64.(sdiag_q[1:min(pair_rows, N)]),
                            v_boundary_diag = Float64.(vdiag_q[1:min(pair_rows, N)]),
                            v_boundary_offdiag = Float64.(voff_q),
                            v_offdiag_pairs = pairs,
                            v_diag_free_indices = diag_free_indices,
                            constructor_kwargs = (boundary_rows = constructor_rows,
                                                  boundary_bandwidth = boundary_bandwidth,
                                                  v_boundary_diag = v_boundary_diag_constructor,
                                                  v_boundary_offdiag = v_boundary_offdiag_constructor)))
end

"""
    sbp8_solve_accuracy_constraints(setup; kwargs...)

Solve SBP8 split-mass constraints with a JuMP/Ipopt nonlinear program.

Core structure:
- scalar mass `S` diagonal with unknown boundary entries and fixed interior tail,
- vector mass `V` symmetric with boundary band block (5- or 7-diagonal),
- SBP-derived divergence polynomial constraints on `r`, `r^3`, `r^5`, `r^7`,
- quadrature constraint and positivity/SPD surrogates,
- regularized objective preserving folded Cartesian morphology.
"""
function sbp8_solve_accuracy_constraints(setup::NamedTuple;
                                         boundary_rows::Union{Nothing, Int} = nothing,
                                         boundary_bandwidth::Int = 7,
                                         boundary_closure::Union{Nothing, Int} = nothing,
                                         first_rows_r7::Int = 6,
                                         exact_solve::Bool = true,
                                         v_diag_free_end::Union{Nothing, Int} = nothing,
                                         s1_target::Union{Nothing, Real} = nothing,
                                         s1_target_factor::Float64 = 0.55,
                                         lambda_s::Float64 = 1.0e-3,
                                         lambda_vs::Float64 = 1.0e-4,
                                         s_min::Float64 = 1.0e-12,
                                         spd_margin::Float64 = 1.0e-12,
                                         ipopt_max_iter::Int = 20000,
                                         ipopt_tol::Float64 = 1.0e-8,
                                         ipopt_acceptable_tol::Float64 = 1.0e-4,
                                         ipopt_constr_viol_tol::Float64 = 1.0e-8,
                                         ipopt_print_level::Int = 0,
                                         post_polish::Bool = true,
                                         post_polish_tol::Float64 = 1.0e-12,
                                         enforce_fp_accuracy::Bool = true,
                                         floating_point_factor::Float64 = 5.0e4,
                                         enforce_real_negative_spectrum::Bool = true,
                                         spectrum_imag_tol::Float64 = 5.0e-6,
                                         spectrum_nonpositive_tol::Float64 = 1.0e-8,
                                         stability_check::Bool = true,
                                         verbose::Bool = true)
    hasproperty(setup, :r) || throw(ArgumentError("`setup` must include `r`."))
    hasproperty(setup, :Geven) || throw(ArgumentError("`setup` must include `Geven`."))
    hasproperty(setup, :Sdiag) || throw(ArgumentError("`setup` must include `Sdiag`."))
    hasproperty(setup, :p) || throw(ArgumentError("`setup` must include `p`."))
    hasproperty(setup, :R) || throw(ArgumentError("`setup` must include `R`."))
    s_min > 0 ||
        throw(ArgumentError("`s_min` must be strictly positive to keep S invertible."))

    if exact_solve
        try
            return _sbp8_solve_accuracy_constraints_exact(setup;
                                                          boundary_rows = boundary_rows,
                                                          boundary_bandwidth = boundary_bandwidth,
                                                          boundary_closure = boundary_closure,
                                                          first_rows_r7 = first_rows_r7,
                                                          v_diag_free_end = v_diag_free_end,
                                                          s1_target = s1_target,
                                                          s1_target_factor = s1_target_factor,
                                                          s_min = s_min,
                                                          enforce_fp_accuracy = enforce_fp_accuracy,
                                                          floating_point_factor = floating_point_factor,
                                                          enforce_real_negative_spectrum = enforce_real_negative_spectrum,
                                                          spectrum_imag_tol = spectrum_imag_tol,
                                                          spectrum_nonpositive_tol = spectrum_nonpositive_tol,
                                                          stability_check = stability_check,
                                                          verbose = verbose)
        catch err
            if verbose
                println("SBP8 exact solve failed (", typeof(err), "): ", err)
                println("Falling back to JuMP/Ipopt solve.")
            end
        end
    end

    r = Float64.(setup.r)
    Geven = _sbp8_sparse_convert(sparse(setup.Geven), Float64)
    s_target = Float64.(setup.Sdiag)
    p = Int(setup.p)

    N = length(r)
    size(Geven) == (N, N) || throw(DimensionMismatch("`Geven` must be $(N)x$(N)."))
    length(s_target) == N || throw(DimensionMismatch("`Sdiag` must have length $N."))

    closure_auto = _sbp8_infer_right_boundary_closure(setup, Geven)
    closure_right = isnothing(boundary_closure) ? closure_auto : Int(boundary_closure)
    rows = _sbp8_constraint_rows(N, closure_right, first_rows_r7)

    boundary_rows_auto = min(N, 2 * closure_auto)
    nb = isnothing(boundary_rows) ? boundary_rows_auto : clamp(Int(boundary_rows), 1, N)

    pairs = sbp8_v_offdiag_pairs(N;
                                 boundary_rows = nb,
                                 boundary_bandwidth = boundary_bandwidth)
    npairs = length(pairs)

    G = Matrix(Geven)
    Bnn = r[end]^p
    Rval = Float64(setup.R)
    quad_target = Rval^(p + 1) / (p + 1)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_optimizer_attribute(model, "tol", ipopt_tol)
    JuMP.set_optimizer_attribute(model, "acceptable_tol", ipopt_acceptable_tol)
    JuMP.set_optimizer_attribute(model, "constr_viol_tol", ipopt_constr_viol_tol)
    JuMP.set_optimizer_attribute(model, "max_iter", ipopt_max_iter)
    JuMP.set_optimizer_attribute(model, "print_level", ipopt_print_level)

    JuMP.@variable(model, s[1:nb]>=s_min)
    JuMP.@variable(model, vdiag[1:nb]>=s_min)
    if npairs > 0
        JuMP.@variable(model, voff[1:npairs])
        @inbounds for k in 1:npairs
            JuMP.set_start_value(voff[k], 0.0)
        end
    end
    @inbounds for i in 1:nb
        JuMP.set_start_value(s[i], s_target[i])
        JuMP.set_start_value(vdiag[i], s_target[i])
    end

    # Cholesky factorization constraints enforce SPD of the boundary block:
    # V_boundary = L * L', with off-band entries constrained to zero.
    JuMP.@variable(model, L[1:nb, 1:nb])
    for i in 1:nb
        for j in (i + 1):nb
            JuMP.fix(L[i, j], 0.0; force = true)
        end
        JuMP.set_lower_bound(L[i, i], sqrt(spd_margin))
        JuMP.set_start_value(L[i, i], sqrt(max(spd_margin, s_target[i])))
        for j in 1:(i - 1)
            JuMP.set_start_value(L[i, j], 0.0)
        end
    end

    pair_to_idx = Dict{Tuple{Int, Int}, Int}()
    if npairs > 0
        @inbounds for (k, (i, j)) in enumerate(pairs)
            pair_to_idx[(i, j)] = k
        end
    end

    for i in 1:nb
        for j in 1:i
            if i == j
                JuMP.@NLconstraint(model, vdiag[i]==sum(L[i, k]^2 for k in 1:i))
            else
                idx = get(pair_to_idx, (j, i), 0)
                if idx > 0
                    JuMP.@NLconstraint(model,
                                       voff[idx]==sum(L[i, k] * L[j, k] for k in 1:j))
                else
                    JuMP.@NLconstraint(model, sum(L[i, k] * L[j, k] for k in 1:j)==0.0)
                end
            end
        end
    end

    if npairs == 0
        for i in 1:nb
            JuMP.@constraint(model, vdiag[i]>=spd_margin)
        end
    end

    function add_degree_constraints!(degree::Int, rows_idx::Vector{Int})
        isempty(rows_idx) && return nothing

        u = r .^ degree
        exact = (p + degree) .* (r .^ (degree - 1))

        @inbounds for i in rows_idx
            constant = i == N ? Bnn * u[i] : 0.0
            expr = JuMP.AffExpr(constant)

            # Diagonal V contribution.
            for a in 1:N
                g = G[a, i]
                g == 0.0 && continue
                coeff_diag = -(g * u[a])
                if a <= nb
                    JuMP.add_to_expression!(expr, coeff_diag, vdiag[a])
                else
                    constant += coeff_diag * s_target[a]
                end
            end

            # Boundary off-diagonal V contribution.
            for (kpair, (a, b)) in enumerate(pairs)
                coeff_pair = -(G[a, i] * u[b] + G[b, i] * u[a])
                coeff_pair == 0.0 && continue
                JuMP.add_to_expression!(expr, coeff_pair, voff[kpair])
            end

            # Scalar mass contribution.
            coeff_s = -exact[i]
            if i <= nb
                JuMP.add_to_expression!(expr, coeff_s, s[i])
            else
                constant += coeff_s * s_target[i]
            end

            expr.constant = constant
            JuMP.@constraint(model, expr==0.0)
        end
        return nothing
    end

    add_degree_constraints!(1, rows.rows_r)
    add_degree_constraints!(3, rows.rows_r3)
    add_degree_constraints!(5, rows.rows_r5)
    add_degree_constraints!(7, rows.rows_r7)

    const_tail = nb < N ? sum(s_target[(nb + 1):N]) : 0.0
    JuMP.@constraint(model, sum(s[i] for i in 1:nb) + const_tail==quad_target)

    if npairs > 0
        JuMP.@objective(model, Min,
                        lambda_s *
                        sum((s[i] - s_target[i])^2 for i in 1:nb)+
                        lambda_vs * (sum((vdiag[i] - s[i])^2 for i in 1:nb) +
                         2.0 * sum(voff[k]^2 for k in 1:npairs)))
    else
        JuMP.@objective(model, Min,
                        lambda_s *
                        sum((s[i] - s_target[i])^2 for i in 1:nb)+
                        lambda_vs * sum((vdiag[i] - s[i])^2 for i in 1:nb))
    end

    JuMP.optimize!(model)

    term = JuMP.termination_status(model)
    primal = JuMP.primal_status(model)
    dual = JuMP.dual_status(model)
    locally_optimal = term == _SBP8_MOI.LOCALLY_SOLVED ||
                      term == _SBP8_MOI.ALMOST_LOCALLY_SOLVED

    JuMP.has_values(model) ||
        throw(ArgumentError("SBP8 JuMP model has no primal solution values (termination_status=$term, primal_status=$primal)."))

    s_boundary = Float64.(JuMP.value.(s))
    s_boundary[1] > 0 ||
        throw(ArgumentError("JuMP SBP8 solution violates strict positivity at origin: S[1,1] = $(s_boundary[1])."))
    s_boundary[1] > s_min ||
        throw(ArgumentError("JuMP SBP8 solution violates invertibility at origin: S[1,1] = $(s_boundary[1]) <= s_min = $s_min."))
    v_boundary_diag = Float64.(JuMP.value.(vdiag))
    v_boundary_offdiag = npairs > 0 ? Float64.(JuMP.value.(voff)) : Float64[]

    # Optional exact linear-polish on (s_boundary, v_boundary_diag, v_boundary_offdiag)
    # to drive enforced accuracy/quadrature constraints to machine precision.
    polished = (applied = false, accepted = false, residual_before = NaN,
                residual_after = NaN)
    if post_polish
        nvar = 2 * nb + npairs
        x0 = zeros(Float64, nvar)
        @inbounds for i in 1:nb
            x0[i] = s_boundary[i]
            x0[nb + i] = v_boundary_diag[i]
        end
        @inbounds for k in 1:npairs
            x0[2 * nb + k] = v_boundary_offdiag[k]
        end

        sys = _sbp8_build_linear_constraint_system(r, G, s_target, p, rows, nb, pairs,
                                                   quad_target)
        res0 = isempty(sys.b) ? 0.0 : maximum(abs.(sys.A * x0 .- sys.b))
        pol = _sbp8_polish_linear_constraints(x0, sys.A, sys.b)
        polished = (applied = true,
                    accepted = false,
                    residual_before = res0,
                    residual_after = pol.residual)

        if pol.success
            x = pol.x
            s_try = x[1:nb]
            vd_try = x[(nb + 1):(2 * nb)]
            vo_try = npairs > 0 ? x[(2 * nb + 1):end] : Float64[]

            # Keep polished values only if positivity and boundary SPD are preserved.
            s_ok = all(si -> si > s_min, s_try)
            Vtry = sbp8_vector_mass(vcat(s_try, s_target[(nb + 1):end]);
                                    boundary_rows = nb,
                                    boundary_bandwidth = boundary_bandwidth,
                                    v_boundary_diag = vd_try,
                                    v_boundary_offdiag = vo_try)
            Vb_try = Matrix{Float64}(Vtry[1:nb, 1:nb])
            v_ok = try
                LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Vb_try); check = true)
                true
            catch
                false
            end

            if s_ok && v_ok && pol.residual <= post_polish_tol
                s_boundary = collect(s_try)
                s_boundary[1] > 0 ||
                    throw(ArgumentError("Post-polish SBP8 solution violates strict positivity at origin: S[1,1] = $(s_boundary[1])."))
                s_boundary[1] > s_min ||
                    throw(ArgumentError("Post-polish SBP8 solution violates invertibility at origin: S[1,1] = $(s_boundary[1]) <= s_min = $s_min."))
                v_boundary_diag = collect(vd_try)
                v_boundary_offdiag = collect(vo_try)
                polished = (applied = true,
                            accepted = true,
                            residual_before = res0,
                            residual_after = pol.residual)
            end
        end
    end

    sdiag = copy(s_target)
    @inbounds for i in 1:nb
        sdiag[i] = s_boundary[i]
    end

    S = spdiagm(0 => sdiag)
    V = sbp8_vector_mass(sdiag;
                         boundary_rows = nb,
                         boundary_bandwidth = boundary_bandwidth,
                         v_boundary_diag = v_boundary_diag,
                         v_boundary_offdiag = v_boundary_offdiag)

    div_data = sbp8_construct_divergence(S, V, Geven, r; p = p)
    D = div_data.D
    B = div_data.B

    err_r = _sbp8_constraint_error_metrics(D, r, p, 1, rows.rows_r)
    err_r3 = _sbp8_constraint_error_metrics(D, r, p, 3, rows.rows_r3)
    err_r5 = _sbp8_constraint_error_metrics(D, r, p, 5, rows.rows_r5)
    err_r7 = _sbp8_constraint_error_metrics(D, r, p, 7, rows.rows_r7)

    quad_val = sum(sdiag)
    quad_err = abs(quad_val - quad_target)
    quad_scale = max(1.0, abs(quad_target))
    quad_rel = quad_err / quad_scale

    s_sym = _sbp8_is_symmetric(S)
    v_sym = _sbp8_is_symmetric(V)
    s_pd = all(si -> si > 0.0, sdiag)

    Vb = Matrix{Float64}(V[1:nb, 1:nb])
    v_boundary_pd = try
        LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Vb); check = true)
        true
    catch
        false
    end
    v_pd = _sbp8_is_pd(V)

    L = sparse(D * Geven)
    eigvals_L = stability_check || enforce_real_negative_spectrum ?
                _high_precision_schur_values(Matrix(L)) : Complex{Float64x4}[]
    max_real_L = isempty(eigvals_L) ? NaN : maximum(Float64.(real.(eigvals_L)))
    max_abs_imag_L = isempty(eigvals_L) ? NaN : maximum(Float64.(abs.(imag.(eigvals_L))))
    spectrum_ok = isempty(eigvals_L) ? true :
                  (max_abs_imag_L <= spectrum_imag_tol &&
                   max_real_L <= spectrum_nonpositive_tol)
    if enforce_real_negative_spectrum && !spectrum_ok
        throw(ArgumentError("SBP8 spectrum check failed: max_abs_imag_eig=$(max_abs_imag_L), " *
                            "max_real_eig=$(max_real_L), thresholds=(imag<=$(spectrum_imag_tol), real<=$(spectrum_nonpositive_tol))."))
    end

    interface_jump = nb < N ? abs(v_boundary_diag[end] - s_target[nb + 1]) : 0.0

    fp_tol = floating_point_factor * eps(Float64)
    fp_ok = (err_r.rel <= fp_tol) &&
            (err_r3.rel <= fp_tol) &&
            (err_r5.rel <= fp_tol) &&
            (err_r7.rel <= fp_tol) &&
            (quad_rel <= fp_tol)
    if enforce_fp_accuracy && !fp_ok
        throw(ArgumentError("SBP8 accuracy constraints are not at floating-point level. " *
                            "rel-errors: Dr=$(err_r.rel), Dr3=$(err_r3.rel), Dr5=$(err_r5.rel), " *
                            "Dr7=$(err_r7.rel), quadrature=$(quad_rel); threshold=$(fp_tol)."))
    end

    if verbose
        println("SBP8 JuMP solve:")
        println("  term = ", term, ", primal = ", primal, ", dual = ", dual,
                ", locally_optimal = ", locally_optimal)
        println("  right closure = ", rows.closure_right,
                ", boundary rows = ", nb,
                ", boundary bandwidth = ", boundary_bandwidth,
                ", offdiag vars = ", npairs)
        println("  constraint max errors (abs): D*r = ", err_r.abs,
                ", D*r^3 = ", err_r3.abs,
                ", D*r^5 = ", err_r5.abs,
                ", D*r^7 = ", err_r7.abs,
                ", quadrature = ", quad_err)
        println("  constraint max errors (rel): D*r = ", err_r.rel,
                ", D*r^3 = ", err_r3.rel,
                ", D*r^5 = ", err_r5.rel,
                ", D*r^7 = ", err_r7.rel,
                ", quadrature = ", quad_rel,
                ", fp_tol = ", fp_tol,
                ", fp_ok = ", fp_ok)
        if post_polish
            println("  post-polish: applied = ", polished.applied,
                    ", accepted = ", polished.accepted,
                    ", residual(before) = ", polished.residual_before,
                    ", residual(after) = ", polished.residual_after)
        end
        println("  SPD checks: S symmetric = ", s_sym,
                ", S positive = ", s_pd,
                ", V symmetric = ", v_sym,
                ", V boundary PD = ", v_boundary_pd,
                ", V full PD = ", v_pd)
        if stability_check
            println("  stability: max real eig(D*G) = ", max_real_L,
                    ", max |imag eig(D*G)| = ", max_abs_imag_L,
                    ", spectral_ok = ", spectrum_ok)
        end
        println("  boundary/interface smoothness: |V[$nb,$nb] - S[$(nb + 1),$(nb + 1)]| = ",
                interface_jump)
    end

    return (mode = :jump_ipopt,
            setup = setup,
            closure_right = rows.closure_right,
            boundary_rows = nb,
            boundary_bandwidth = boundary_bandwidth,
            row_sets = rows,
            termination_status = term,
            primal_status = primal,
            dual_status = dual,
            locally_optimal = locally_optimal,
            S = S,
            V = V,
            D = D,
            B = B,
            L = L,
            max_real_eig_L = max_real_L,
            max_abs_imag_eig_L = max_abs_imag_L,
            spectrum = (enforce_real_negative_spectrum = enforce_real_negative_spectrum,
                        imag_tolerance = spectrum_imag_tol,
                        nonpositive_tolerance = spectrum_nonpositive_tol,
                        satisfied = spectrum_ok),
            errors = (Dr = err_r.abs,
                      Dr3 = err_r3.abs,
                      Dr5 = err_r5.abs,
                      Dr7 = err_r7.abs,
                      Dr_rel = err_r.rel,
                      Dr3_rel = err_r3.rel,
                      Dr5_rel = err_r5.rel,
                      Dr7_rel = err_r7.rel,
                      quadrature = quad_err,
                      quadrature_rel = quad_rel,
                      interface_jump = interface_jump),
            floating_point_accuracy = (factor = floating_point_factor,
                                       tolerance = fp_tol,
                                       satisfied = fp_ok),
            polish = polished,
            spd = (S_symmetric = s_sym,
                   S_positive_definite = s_pd,
                   V_symmetric = v_sym,
                   V_boundary_positive_definite = v_boundary_pd,
                   V_positive_definite = v_pd),
            coefficients = (s_boundary = s_boundary,
                            v_boundary_diag = v_boundary_diag,
                            v_boundary_offdiag = v_boundary_offdiag,
                            v_offdiag_pairs = pairs,
                            constructor_kwargs = (boundary_rows = nb,
                                                  boundary_bandwidth = boundary_bandwidth,
                                                  v_boundary_diag = v_boundary_diag,
                                                  v_boundary_offdiag = v_boundary_offdiag)))
end

"""
    sbp8_solve_accuracy_constraints(source; kwargs...)

Convenience overload: build folded SBP8 setup via `sbp8_scalar_mass_gradient`, then
solve for boundary split-mass coefficients with JuMP/Ipopt.
"""
function sbp8_solve_accuracy_constraints(source;
                                         accuracy_order::Int = 8,
                                         points::Int = 21,
                                         h::Real = 1,
                                         N = points - 1,
                                         R = h * (points - 1),
                                         p::Int = 2,
                                         mode = SafeMode(),
                                         atol = nothing,
                                         boundary_rows::Union{Nothing, Int} = nothing,
                                         boundary_bandwidth::Int = 7,
                                         boundary_closure::Union{Nothing, Int} = nothing,
                                         first_rows_r7::Int = 6,
                                         exact_solve::Bool = true,
                                         v_diag_free_end::Union{Nothing, Int} = nothing,
                                         s1_target::Union{Nothing, Real} = nothing,
                                         s1_target_factor::Float64 = 0.55,
                                         lambda_s::Float64 = 1.0e-3,
                                         lambda_vs::Float64 = 1.0e-4,
                                         s_min::Float64 = 1.0e-12,
                                         spd_margin::Float64 = 1.0e-12,
                                         ipopt_max_iter::Int = 20000,
                                         ipopt_tol::Float64 = 1.0e-8,
                                         ipopt_acceptable_tol::Float64 = 1.0e-4,
                                         ipopt_constr_viol_tol::Float64 = 1.0e-8,
                                         ipopt_print_level::Int = 0,
                                         post_polish::Bool = true,
                                         post_polish_tol::Float64 = 1.0e-12,
                                         enforce_fp_accuracy::Bool = true,
                                         floating_point_factor::Float64 = 5.0e4,
                                         enforce_real_negative_spectrum::Bool = true,
                                         spectrum_imag_tol::Float64 = 5.0e-6,
                                         spectrum_nonpositive_tol::Float64 = 1.0e-8,
                                         stability_check::Bool = true,
                                         verbose::Bool = true)
    setup = sbp8_scalar_mass_gradient(source;
                                      accuracy_order = accuracy_order,
                                      points = points,
                                      h = h,
                                      N = N,
                                      R = R,
                                      p = p,
                                      mode = mode,
                                      atol = atol)

    return sbp8_solve_accuracy_constraints(setup;
                                           boundary_rows = boundary_rows,
                                           boundary_bandwidth = boundary_bandwidth,
                                           boundary_closure = boundary_closure,
                                           first_rows_r7 = first_rows_r7,
                                           exact_solve = exact_solve,
                                           v_diag_free_end = v_diag_free_end,
                                           s1_target = s1_target,
                                           s1_target_factor = s1_target_factor,
                                           lambda_s = lambda_s,
                                           lambda_vs = lambda_vs,
                                           s_min = s_min,
                                           spd_margin = spd_margin,
                                           ipopt_max_iter = ipopt_max_iter,
                                           ipopt_tol = ipopt_tol,
                                           ipopt_acceptable_tol = ipopt_acceptable_tol,
                                           ipopt_constr_viol_tol = ipopt_constr_viol_tol,
                                           ipopt_print_level = ipopt_print_level,
                                           post_polish = post_polish,
                                           post_polish_tol = post_polish_tol,
                                           enforce_fp_accuracy = enforce_fp_accuracy,
                                           floating_point_factor = floating_point_factor,
                                           enforce_real_negative_spectrum = enforce_real_negative_spectrum,
                                           spectrum_imag_tol = spectrum_imag_tol,
                                           spectrum_nonpositive_tol = spectrum_nonpositive_tol,
                                           stability_check = stability_check,
                                           verbose = verbose)
end

"""
    sbp8_operators(source, points; kwargs...)

Build non-diagonal-mass SBP8 operators by solving the split-mass optimization,
then return `(D, G, S, V, B)`.
"""
function sbp8_operators(source,
                        points::Integer;
                        h::Real = 1,
                        accuracy_order::Int = 8,
                        p::Int = 2,
                        mode = SafeMode(),
                        atol = nothing,
                        boundary_rows::Union{Nothing, Int} = nothing,
                        boundary_bandwidth::Int = 7,
                        boundary_closure::Union{Nothing, Int} = nothing,
                        first_rows_r7::Int = 6,
                        exact_solve::Bool = true,
                        v_diag_free_end::Union{Nothing, Int} = nothing,
                        s1_target::Union{Nothing, Real} = nothing,
                        s1_target_factor::Float64 = 0.55,
                        lambda_s::Float64 = 1.0e-3,
                        lambda_vs::Float64 = 1.0e-4,
                        s_min::Float64 = 1.0e-12,
                        spd_margin::Float64 = 1.0e-12,
                        ipopt_max_iter::Int = 20000,
                        ipopt_tol::Float64 = 1.0e-8,
                        ipopt_acceptable_tol::Float64 = 1.0e-4,
                        ipopt_constr_viol_tol::Float64 = 1.0e-8,
                        ipopt_print_level::Int = 0,
                        post_polish::Bool = true,
                        post_polish_tol::Float64 = 1.0e-12,
                        enforce_fp_accuracy::Bool = true,
                        floating_point_factor::Float64 = 5.0e4,
                        enforce_real_negative_spectrum::Bool = true,
                        spectrum_imag_tol::Float64 = 5.0e-6,
                        spectrum_nonpositive_tol::Float64 = 1.0e-8,
                        stability_check::Bool = true,
                        verbose::Bool = false)
    points_int = Int(points)
    points_int > 1 || throw(ArgumentError("`points` must be > 1."))

    solved = sbp8_solve_accuracy_constraints(source;
                                             accuracy_order = accuracy_order,
                                             points = points_int,
                                             h = h,
                                             p = p,
                                             mode = mode,
                                             atol = atol,
                                             boundary_rows = boundary_rows,
                                             boundary_bandwidth = boundary_bandwidth,
                                             boundary_closure = boundary_closure,
                                             first_rows_r7 = first_rows_r7,
                                             exact_solve = exact_solve,
                                             v_diag_free_end = v_diag_free_end,
                                             s1_target = s1_target,
                                             s1_target_factor = s1_target_factor,
                                             lambda_s = lambda_s,
                                             lambda_vs = lambda_vs,
                                             s_min = s_min,
                                             spd_margin = spd_margin,
                                             ipopt_max_iter = ipopt_max_iter,
                                             ipopt_tol = ipopt_tol,
                                             ipopt_acceptable_tol = ipopt_acceptable_tol,
                                             ipopt_constr_viol_tol = ipopt_constr_viol_tol,
                                             ipopt_print_level = ipopt_print_level,
                                             post_polish = post_polish,
                                             post_polish_tol = post_polish_tol,
                                             enforce_fp_accuracy = enforce_fp_accuracy,
                                             floating_point_factor = floating_point_factor,
                                             enforce_real_negative_spectrum = enforce_real_negative_spectrum,
                                             spectrum_imag_tol = spectrum_imag_tol,
                                             spectrum_nonpositive_tol = spectrum_nonpositive_tol,
                                             stability_check = stability_check,
                                             verbose = verbose)

    return (D = solved.D,
            G = solved.setup.Geven,
            S = solved.S,
            V = solved.V,
            B = solved.B)
end
