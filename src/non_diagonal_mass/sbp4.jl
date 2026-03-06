"""
    sbp4_v_offdiag_pairs(N)

SBP4 near-origin off-diagonal pattern for `V`.
Allowed pairs are `(2,3)`, `(3,4)`, and `(4,5)` when those indices exist.
"""
function sbp4_v_offdiag_pairs(N::Integer)
    Nint = Int(N)
    Nint >= 1 || throw(ArgumentError("`N` must be positive."))
    pairs = Tuple{Int, Int}[]
    Nint >= 3 && push!(pairs, (2, 3))
    Nint >= 4 && push!(pairs, (3, 4))
    Nint >= 5 && push!(pairs, (4, 5))
    return pairs
end

"""
    sbp4_vector_mass(; N=10, v_symbol_prefix="v", s_symbol_prefix="s")

Symbolic SBP4 split-mass template.

Structure:
- `V[1,1] = 1`
- `V[2,2] = v_2`
- `V[3,3] = v_3`
- `V[4,4] = v_4`
- `V[5,5] = v_5`
- `V[2,3] = V[3,2] = v_2_3`
- `V[3,4] = V[4,3] = v_3_4`
- `V[4,5] = V[5,4] = v_4_5`
- `V[i,i] = s_i` for `i >= 6`
"""
function sbp4_vector_mass(; N::Integer = 10,
                          v_symbol_prefix::AbstractString = "v",
                          s_symbol_prefix::AbstractString = "s")
    Nint = Int(N)
    Nint >= 1 || throw(ArgumentError("`N` must be positive."))

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

    v2_symbol = Nint >= 2 ? Symbol(v_symbol_prefix, "_2") : Symbol[]
    v3_symbol = Nint >= 3 ? Symbol(v_symbol_prefix, "_3") : Symbol[]
    v4_symbol = Nint >= 4 ? Symbol(v_symbol_prefix, "_4") : Symbol[]
    v5_symbol = Nint >= 5 ? Symbol(v_symbol_prefix, "_5") : Symbol[]
    v23_symbol = Nint >= 3 ? Symbol(v_symbol_prefix, "_2_3") : Symbol[]
    v34_symbol = Nint >= 4 ? Symbol(v_symbol_prefix, "_3_4") : Symbol[]
    v45_symbol = Nint >= 5 ? Symbol(v_symbol_prefix, "_4_5") : Symbol[]

    if Nint >= 1
        V[1, 1] = 1
    end
    if Nint >= 2
        V[2, 2] = v2_symbol
    end
    if Nint >= 3
        V[3, 3] = v3_symbol
        V[2, 3] = v23_symbol
        V[3, 2] = v23_symbol
    end
    if Nint >= 4
        V[4, 4] = v4_symbol
        V[3, 4] = v34_symbol
        V[4, 3] = v34_symbol
    end
    if Nint >= 5
        V[5, 5] = v5_symbol
        V[4, 5] = v45_symbol
        V[5, 4] = v45_symbol
    end
    if Nint >= 6
        @inbounds for i in 6:Nint
            V[i, i] = s_diag_symbols[i]
        end
    end

    v_diag_symbols = Symbol[]
    Nint >= 2 && push!(v_diag_symbols, v2_symbol)
    Nint >= 3 && push!(v_diag_symbols, v3_symbol)
    Nint >= 4 && push!(v_diag_symbols, v4_symbol)
    Nint >= 5 && push!(v_diag_symbols, v5_symbol)
    v_offdiag_symbols = Symbol[]
    Nint >= 3 && push!(v_offdiag_symbols, v23_symbol)
    Nint >= 4 && push!(v_offdiag_symbols, v34_symbol)
    Nint >= 5 && push!(v_offdiag_symbols, v45_symbol)

    free_symbols = vcat(s_diag_symbols, v_diag_symbols, v_offdiag_symbols)
    symbol_to_index = Dict{Symbol, Int}(sym => idx for (idx, sym) in enumerate(free_symbols))

    return (
            S = S,
            V = V,
            free_symbols = free_symbols,
            symbol_to_index = symbol_to_index,
            s_diag_symbols = s_diag_symbols,
            v_diag_symbols = v_diag_symbols,
            v_offdiag_symbols = v_offdiag_symbols,
            v_offdiag_pairs = sbp4_v_offdiag_pairs(Nint),
           )
end

function _sbp4_extract_diagonal(S::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
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
    sbp4_vector_mass(s_diag; v2=s_diag[2], v3=s_diag[3], v4=s_diag[4], v5=s_diag[5],
                     v23=zero(T), v34=zero(T), v45=zero(T))

Numeric SBP4 `V` construction with the fixed pattern:
- only off-diagonals `(2,3)`/`(3,2)`, `(3,4)`/`(4,3)`, `(4,5)`/`(5,4)` may be nonzero;
- `V[1,1] = 1`;
- diagonal entries from `i >= 6` are copied from `s_diag[i]`.
"""
function sbp4_vector_mass(s_diag::AbstractVector{T};
                          v2 = (length(s_diag) >= 2 ? s_diag[2] : zero(T)),
                          v3 = (length(s_diag) >= 3 ? s_diag[3] : zero(T)),
                          v4 = (length(s_diag) >= 4 ? s_diag[4] : zero(T)),
                          v5 = (length(s_diag) >= 5 ? s_diag[5] : zero(T)),
                          v23 = zero(T),
                          v34 = zero(T),
                          v45 = zero(T)) where {T <: Real}
    N = length(s_diag)
    N >= 1 || throw(ArgumentError("`s_diag` must be non-empty."))

    vdiag = [convert(T, si) for si in s_diag]
    vdiag[1] = one(T)
    if N >= 2
        vdiag[2] = convert(T, v2)
    end
    if N >= 3
        vdiag[3] = convert(T, v3)
    end
    if N >= 4
        vdiag[4] = convert(T, v4)
    end
    if N >= 5
        vdiag[5] = convert(T, v5)
    end

    V = spdiagm(0 => vdiag)
    if N >= 3
        v = convert(T, v23)
        if v != zero(T)
            V[2, 3] = v
            V[3, 2] = v
        end
    end
    if N >= 4
        v = convert(T, v34)
        if v != zero(T)
            V[3, 4] = v
            V[4, 3] = v
        end
    end
    if N >= 5
        v = convert(T, v45)
        if v != zero(T)
            V[4, 5] = v
            V[5, 4] = v
        end
    end
    return V
end

function sbp4_vector_mass(S::SparseMatrixCSC{T, Ti};
                          v2 = nothing,
                          v3 = nothing,
                          v4 = nothing,
                          v5 = nothing,
                          v23 = zero(T),
                          v34 = zero(T),
                          v45 = zero(T)) where {T <: Real, Ti <: Integer}
    sdiag = _sbp4_extract_diagonal(S)
    v2_use = isnothing(v2) ? (length(sdiag) >= 2 ? sdiag[2] : zero(T)) : convert(T, v2)
    v3_use = isnothing(v3) ? (length(sdiag) >= 3 ? sdiag[3] : zero(T)) : convert(T, v3)
    v4_use = isnothing(v4) ? (length(sdiag) >= 4 ? sdiag[4] : zero(T)) : convert(T, v4)
    v5_use = isnothing(v5) ? (length(sdiag) >= 5 ? sdiag[5] : zero(T)) : convert(T, v5)
    return sbp4_vector_mass(sdiag; v2 = v2_use, v3 = v3_use, v4 = v4_use, v5 = v5_use,
                            v23 = v23, v34 = v34, v45 = v45)
end

@inline _sbp4_as_big_rational(x::Rational{BigInt}) = x
@inline _sbp4_as_big_rational(x::Integer) = big(x) // 1
@inline _sbp4_as_big_rational(x::Rational{<:Integer}) = big(numerator(x)) // big(denominator(x))

function _sbp4_as_big_rational(x::AbstractFloat)
    isfinite(x) || throw(ArgumentError("Cannot convert non-finite floating-point value `$x` to Rational{BigInt}."))
    return rationalize(BigInt, x)
end

function _sbp4_as_big_rational(x::Real)
    throw(ArgumentError("Could not convert value of type $(typeof(x)) to Rational{BigInt}."))
end

function _sbp4_maxabs(values)
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

function _sbp4_is_symmetric(A::SparseMatrixCSC{T, Ti}; tol::Float64 = 1e-12) where {T <: Real, Ti <: Integer}
    if T <: AbstractFloat
        return norm(Matrix(A - transpose(A)), Inf) <= tol
    end
    return Matrix(A) == transpose(Matrix(A))
end

function _sbp4_is_pd(A::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
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
    sbp4_scalar_mass_gradient(source; accuracy_order=4, points=21, h=1,
                              N=points-1, R=h*(points-1), p=2, mode=SafeMode(),
                              build_matrix=:probe, atol=nothing)

Build folded half-grid gradient and diagonal scalar mass in exact rational arithmetic.
"""
function sbp4_scalar_mass_gradient(source;
                                   accuracy_order::Int = 4,
                                   points::Int = 21,
                                   h::Real = 1,
                                   N = points - 1,
                                   R = h * (points - 1),
                                   p::Int = 2,
                                   mode = SafeMode(),
                                   build_matrix::Symbol = :probe,
                                   atol = nothing)
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    Nint = Int(N)
    Nint > 0 || throw(ArgumentError("`N` must be positive."))
    accuracy_order > 0 || throw(ArgumentError("`accuracy_order` must be positive."))
    points > 1 || throw(ArgumentError("`points` must be > 1."))

    Rq = _sbp4_as_big_rational(R)
    Dfull, xfull, Gfull, Hfull = _build_full_grid_objects(
                                                           source;
                                                           accuracy_order = accuracy_order,
                                                           N = Nint,
                                                           R = Rq,
                                                           mode = mode,
                                                           build_matrix = build_matrix
                                                          )

    T = eltype(xfull)
    atol_use = _resolve_atol(T, atol)
    r, Rop, Eeven, Eodd = _build_folding_operators(xfull; atol = atol_use)

    Geven = sparse(Rop * Gfull * Eeven)
    Godd = sparse(Rop * Gfull * Eodd)

    half_factor = convert(T, 1) / convert(T, 2)
    Hcart_half = sparse(half_factor * (transpose(Eeven) * Hfull * Eeven))
    metric = spdiagm(0 => r .^ p)
    S = sparse(Hcart_half * metric)
    Sdiag = _sbp4_extract_diagonal(S)

    return (
            r = r,
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
            atol = atol_use,
            build_matrix = build_matrix
           )
end

"""
    sbp4_construct_divergence(S, V, Geven, r; p=2)

Construct `D = S^{-1}(B - G^T V)`, with `B[end,end] = r[end]^p`.
"""
function sbp4_construct_divergence(S::SparseMatrixCSC{T, Ti},
                                   V::SparseMatrixCSC{T, Ti},
                                   Geven::SparseMatrixCSC{T, Ti},
                                   r::AbstractVector;
                                   p::Int = 2) where {T <: Real, Ti <: Integer}
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    N = length(r)
    size(S) == (N, N) || throw(DimensionMismatch("`S` must be $(N)x$(N)."))
    size(V) == (N, N) || throw(DimensionMismatch("`V` must be $(N)x$(N)."))
    size(Geven) == (N, N) || throw(DimensionMismatch("`Geven` must be $(N)x$(N)."))

    Sdiag = _sbp4_extract_diagonal(S)
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
    return (D = D, B = B, RHS = RHS, Sdiag = Sdiag)
end

function _sbp4_constraint_rows(N::Int)
    rows_r = collect(1:max(N - 4, 0))
    rows_r3 = collect(1:min(5, N))
    return (rows_r = rows_r, rows_r3 = rows_r3)
end

function _sbp4_constraint_error(D::SparseMatrixCSC, r::AbstractVector, p::Int, degree::Int, rows::Vector{Int})
    isempty(rows) && return 0.0
    u = r .^ degree
    exact = (p + degree) .* (r .^ (degree - 1))
    num = D * u
    return maximum(abs.(Float64.(num[rows] .- exact[rows])))
end

"""
    sbp4_solve_accuracy_constraints(setup; exact_solve=true, verbose=true)

Solve the exact rational linear system for SBP4 split masses using:
- `D = S^{-1}(B - G^T V)`
- `(D*r - (p+1))[1:n-4] = 0`,
- `(D*r^3 - (p+3)r^2)[1:5] = 0`,
- `ones' * S * ones = R^3/3`,
- unknowns only in the boundary block: `s[1:5]` and `V[1:5,1:5]`,
- for `i > 5`, set `s[i] = (Hcart_half * r^p)[i]` (for spherical `p=2`, this is
  `Hcart_half * r^2` as requested), and enforce `V[i,i] = s[i]`.
"""
function sbp4_solve_accuracy_constraints(setup::NamedTuple;
                                         exact_solve::Bool = true,
                                         verbose::Bool = true)
    hasproperty(setup, :r) || throw(ArgumentError("`setup` must include `r`."))
    hasproperty(setup, :Geven) || throw(ArgumentError("`setup` must include `Geven`."))
    hasproperty(setup, :p) || throw(ArgumentError("`setup` must include `p`."))
    hasproperty(setup, :R) || throw(ArgumentError("`setup` must include `R`."))

    r = setup.r
    Geven = setup.Geven
    p = Int(setup.p)
    N = length(r)
    size(Geven) == (N, N) || throw(DimensionMismatch("`Geven` must be $(N)x$(N)."))

    exact_solve || throw(ArgumentError("Only exact rational solve is supported; set `exact_solve=true`."))

    rows = _sbp4_constraint_rows(N)

    # Unknowns: boundary block only.
    # Free scalar masses: s[1:5].
    n_s_free = min(5, N)
    idx_s(i) = i
    next_idx = n_s_free
    idx_v2 = 0
    idx_v3 = 0
    idx_v4 = 0
    idx_v5 = 0
    idx_v23 = 0
    idx_v34 = 0
    idx_v45 = 0
    if N >= 2
        next_idx += 1
        idx_v2 = next_idx
    end
    if N >= 3
        next_idx += 1
        idx_v3 = next_idx
    end
    if N >= 4
        next_idx += 1
        idx_v4 = next_idx
    end
    if N >= 5
        next_idx += 1
        idx_v5 = next_idx
    end
    if N >= 3
        next_idx += 1
        idx_v23 = next_idx
    end
    if N >= 4
        next_idx += 1
        idx_v34 = next_idx
    end
    if N >= 5
        next_idx += 1
        idx_v45 = next_idx
    end
    n_unknowns = next_idx

    Tq = Rational{BigInt}
    rq = Tq[_sbp4_as_big_rational(ri) for ri in r]
    I, J, GV = findnz(sparse(Geven))
    Gq = sparse(I, J, Tq[_sbp4_as_big_rational(gij) for gij in GV], N, N)
    Ih, Jh, HV = findnz(sparse(setup.Hcart_half))
    Hq = sparse(Ih, Jh, Tq[_sbp4_as_big_rational(hij) for hij in HV], N, N)
    Rq = _sbp4_as_big_rational(setup.R)
    s_fixed = Hq * (rq .^ p)

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

    function add_degree_constraints!(degree::Int, row_idx::Vector{Int}, family::Symbol)
        isempty(row_idx) && return nothing
        u = rq .^ degree
        coeff = convert(Tq, p + degree)
        @inbounds for i in row_idx
            row = zeros(Tq, n_unknowns)
            exact_i = coeff * rq[i]^(degree - 1)
            constant_term = i == N ? (BNN * u[i]) : (0 // 1)

            # S term: -s_i * exact_i
            if i <= n_s_free
                row[idx_s(i)] -= exact_i
            else
                constant_term += -(s_fixed[i] * exact_i)
            end

            # Tail V diagonals tied to S for j > n_s_free.
            for j in (n_s_free + 1):N
                gij = Gq[j, i]
                gij == 0 // 1 && continue
                constant_term += -(gij * s_fixed[j] * u[j])
            end

            # Free V diagonals.
            if N >= 2
                row[idx_v2] += -(Gq[2, i] * u[2])
            end
            if N >= 3
                row[idx_v3] += -(Gq[3, i] * u[3])
            end
            if N >= 4
                row[idx_v4] += -(Gq[4, i] * u[4])
            end
            if N >= 5
                row[idx_v5] += -(Gq[5, i] * u[5])
            end
            if N >= 3
                row[idx_v23] += -(Gq[2, i] * u[3] + Gq[3, i] * u[2])
            end
            if N >= 4
                row[idx_v34] += -(Gq[3, i] * u[4] + Gq[4, i] * u[3])
            end
            if N >= 5
                row[idx_v45] += -(Gq[4, i] * u[5] + Gq[5, i] * u[4])
            end

            # V[1,1] is fixed to one (not an unknown), so its contribution is constant.
            fixed_v11_term = -(Gq[1, i] * u[1])
            constant_term += fixed_v11_term
            rhs = -constant_term
            push_eq!(row, rhs, family, degree, i)
        end
        return nothing
    end

    add_degree_constraints!(1, rows.rows_r, :divergence)
    add_degree_constraints!(3, rows.rows_r3, :divergence)

    # ones' * S * ones = R^3 / 3
    row_q0 = zeros(Tq, n_unknowns)
    @inbounds for i in 1:n_s_free
        row_q0[idx_s(i)] = 1 // 1
    end
    rhs_q0 = (Rq^3 / 3) - sum(s_fixed[i] for i in (n_s_free + 1):N)
    push_eq!(row_q0, rhs_q0, :quadrature, 0, 0)

    m = length(A_rows)
    A = m == 0 ? zeros(Tq, 0, n_unknowns) : reduce(vcat, (reshape(row, 1, :) for row in A_rows))
    bvec = collect(b_rows)

    x = _solve_exact_linear_system(A, bvec)
    residual = m == 0 ? 0 // 1 : _sbp4_maxabs(abs.(A * x .- bvec))

    sdiag = copy(s_fixed)
    @inbounds for i in 1:n_s_free
        sdiag[i] = x[idx_s(i)]
    end
    v2 = N >= 2 ? x[idx_v2] : 0 // 1
    v3 = N >= 3 ? x[idx_v3] : 0 // 1
    v4 = N >= 4 ? x[idx_v4] : 0 // 1
    v5 = N >= 5 ? x[idx_v5] : 0 // 1
    v23 = N >= 3 ? x[idx_v23] : 0 // 1
    v34 = N >= 4 ? x[idx_v34] : 0 // 1
    v45 = N >= 5 ? x[idx_v45] : 0 // 1

    S = spdiagm(0 => sdiag)
    V = sbp4_vector_mass(sdiag; v2 = v2, v3 = v3, v4 = v4, v5 = v5, v23 = v23, v34 = v34, v45 = v45)
    div_data = sbp4_construct_divergence(S, V, Gq, rq; p = p)
    D = div_data.D
    B = div_data.B

    err_r = _sbp4_constraint_error(D, rq, p, 1, rows.rows_r)
    err_r3 = _sbp4_constraint_error(D, rq, p, 3, rows.rows_r3)
    quad0_target = Rq^3 / 3
    quad0_val = sum(sdiag)
    quad0_err = Float64(abs(quad0_val - quad0_target))
    tail_fix_err = n_s_free < N ?
                   maximum(abs.(Float64.(sdiag[(n_s_free + 1):N] .- s_fixed[(n_s_free + 1):N]))) :
                   0.0

    s_sym = _sbp4_is_symmetric(S)
    v_sym = _sbp4_is_symmetric(V)
    s_pd = all(si -> si > 0 // 1, sdiag)
    v_pd = _sbp4_is_pd(V)

    if verbose
        println("SBP4 solve mode: exact")
        println("  equations = ", m, ", unknowns = ", n_unknowns, ", exact residual = ", residual)
        println("  constraint max errors: D*r = ", err_r,
                ", D*r^3(first 5 rows) = ", err_r3,
                ", ones.S.ones = ", quad0_err,
                ", fixed-tail check = ", tail_fix_err)
        println("  SPD checks: S symmetric = ", s_sym, ", S PD = ", s_pd,
                ", V symmetric = ", v_sym, ", V PD = ", v_pd)
    end

    return (
            mode = :exact,
            setup = setup,
            row_sets = rows,
            A = A,
            b = bvec,
            tags = tags,
            x = x,
            residual = residual,
            S = S,
            V = V,
            D = D,
            B = B,
            errors = (
                      Dr_first_n_minus_4_rows = err_r,
                      Dr3_first5_rows = err_r3,
                      ones_S_ones = quad0_err,
                      fixed_tail = tail_fix_err
                     ),
            spd = (
                   S_symmetric = s_sym,
                   S_positive_definite = s_pd,
                   V_symmetric = v_sym,
                   V_positive_definite = v_pd
                  )
           )
end

"""
    sbp4_solve_accuracy_constraints(source; kwargs...)

Source-based convenience overload.
"""
function sbp4_solve_accuracy_constraints(source;
                                         accuracy_order::Int = 4,
                                         points::Int = 21,
                                         h::Real = 1,
                                         N = points - 1,
                                         R = h * (points - 1),
                                         p::Int = 2,
                                         mode = SafeMode(),
                                         build_matrix::Symbol = :probe,
                                         atol = nothing,
                                         exact_solve::Bool = true,
                                         verbose::Bool = true)
    setup = sbp4_scalar_mass_gradient(
                                      source;
                                      accuracy_order = accuracy_order,
                                      points = points,
                                      h = h,
                                      N = N,
                                      R = R,
                                      p = p,
                                      mode = mode,
                                      build_matrix = build_matrix,
                                      atol = atol
                                     )
    return sbp4_solve_accuracy_constraints(
                                          setup;
                                          exact_solve = exact_solve,
                                          verbose = verbose
                                         )
end

"""
    sbp4_operators(source, points; kwargs...)

Construct SBP4 operators by solving split-mass constraints and return `(D,G,S,V,B)`.
"""
function sbp4_operators(source,
                        points::Integer;
                        h::Real = 1,
                        accuracy_order::Int = 4,
                        p::Int = 2,
                        mode = SafeMode(),
                        build_matrix::Symbol = :probe,
                        atol = nothing,
                        exact_solve::Bool = true,
                        verbose::Bool = false)
    solved = sbp4_solve_accuracy_constraints(
                                             source;
                                             accuracy_order = accuracy_order,
                                             points = Int(points),
                                             h = h,
                                             p = p,
                                             mode = mode,
                                             build_matrix = build_matrix,
                                             atol = atol,
                                             exact_solve = exact_solve,
                                             verbose = verbose
                                            )
    return (
            D = solved.D,
            G = solved.setup.Geven,
            S = solved.S,
            V = solved.V,
            B = solved.B
           )
end
