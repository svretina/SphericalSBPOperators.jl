"""
    sbp6_exp_v_offdiag_pairs(N; outer_boundary_closure_help=true)

Return the ordered list of free off-diagonal index pairs `(i,j)` used by the
experimental non-diagonal SBP6 vector-mass pattern.

Pattern:
- left/origin closure: `(2,3)`, `(2,4)`, `(3,4)`, `(3,5)`, `(4,5)`, `(5,6)`, `(6,7)`
- optional right/outer closure help: `(N-2,N-1)`, `(N-1,N)`
"""
function sbp6_exp_v_offdiag_pairs(N::Integer; outer_boundary_closure_help::Bool = true)
    Nint = Int(N)
    Nint >= 17 ||
        throw(ArgumentError("Experimental SBP6 requires `N >= 17` to keep the fixed left/right closures disjoint."))

    pairs = Tuple{Int, Int}[
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 5),
        (4, 5),
        (5, 6),
        (6, 7)
    ]
    if outer_boundary_closure_help
        push!(pairs, (Nint - 2, Nint - 1))
        push!(pairs, (Nint - 1, Nint))
    end
    return pairs
end

@inline _sbp6_exp_left_diag_indices(N::Int) = collect(2:min(10, N))
@inline _sbp6_exp_interior_diag_indices(N::Int; outer_boundary_closure_help::Bool = true) =
    outer_boundary_closure_help ? (11 <= N - 7 ? collect(11:(N - 7)) : Int[]) :
    (11 <= N ? collect(11:N) : Int[])
@inline _sbp6_exp_right_diag_indices(N::Int; outer_boundary_closure_help::Bool = true) =
    outer_boundary_closure_help ? collect((N - 6):N) : Int[]

"""
    sbp6_exp_vector_mass(; N=17, v_symbol_prefix="v", s_symbol_prefix="s",
                         outer_boundary_closure_help=true)

Construct a symbolic experimental SBP6 vector-mass template.
"""
function sbp6_exp_vector_mass(;
                              N::Integer = 17,
                              v_symbol_prefix::AbstractString = "v",
                              s_symbol_prefix::AbstractString = "s",
                              outer_boundary_closure_help::Bool = true)
    Nint = Int(N)
    Nint >= 17 ||
        throw(ArgumentError("Experimental SBP6 requires `N >= 17`."))

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
    V[1, 1] = 1

    diag_free_indices = vcat(_sbp6_exp_left_diag_indices(Nint),
                             _sbp6_exp_right_diag_indices(Nint;
                                                          outer_boundary_closure_help = outer_boundary_closure_help))
    v_diag_symbols = Symbol[]
    @inbounds for i in diag_free_indices
        sym = Symbol(v_symbol_prefix, "_", i)
        V[i, i] = sym
        push!(v_diag_symbols, sym)
    end

    v_offdiag_pairs = sbp6_exp_v_offdiag_pairs(Nint;
                                               outer_boundary_closure_help = outer_boundary_closure_help)
    v_offdiag_symbols = Symbol[]
    @inbounds for (i, j) in v_offdiag_pairs
        sym = Symbol(v_symbol_prefix, "_", i, "_", j)
        V[i, j] = sym
        V[j, i] = sym
        push!(v_offdiag_symbols, sym)
    end

    s_tail_symbols = Symbol[]
    @inbounds for i in _sbp6_exp_interior_diag_indices(Nint;
                                                       outer_boundary_closure_help = outer_boundary_closure_help)
        sym = s_diag_symbols[i]
        V[i, i] = sym
        push!(s_tail_symbols, sym)
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
            left_diag_indices = _sbp6_exp_left_diag_indices(Nint),
            interior_diag_indices = _sbp6_exp_interior_diag_indices(Nint;
                                                                     outer_boundary_closure_help = outer_boundary_closure_help),
            right_diag_indices = _sbp6_exp_right_diag_indices(Nint;
                                                               outer_boundary_closure_help = outer_boundary_closure_help),
            outer_boundary_closure_help = outer_boundary_closure_help)
end

"""
    sbp6_exp_vector_mass(s_diag; v11=one(eltype(s_diag)),
                         v_diag_block=nothing, v_offdiag_block=nothing)

Construct the experimental SBP6 vector-mass matrix `V` from scalar diagonal entries
`s_diag`.

The order of `v_diag_block` is `[v[2],...,v[10],v[N-6],...,v[N]]`.
The order of `v_offdiag_block` matches `sbp6_exp_v_offdiag_pairs(N)`.
Interior diagonal entries `V[i,i]` for `i = 11:N-7` are tied to `s_diag[i]`.
"""
function sbp6_exp_vector_mass(s_diag::AbstractVector{T};
                              v11 = one(T),
                              v_diag_block::Union{Nothing, AbstractVector} = nothing,
                              v_offdiag_block::Union{Nothing, AbstractVector} = nothing,
                              outer_boundary_closure_help::Bool = true) where {
                                                                                               T <:
                                                                                               Real}
    N = length(s_diag)
    N >= 17 ||
        throw(ArgumentError("Experimental SBP6 requires `length(s_diag) >= 17`."))

    vdiag = Vector{T}(undef, N)
    @inbounds for i in 1:N
        vdiag[i] = convert(T, s_diag[i])
    end
    vdiag[1] = convert(T, v11)

    diag_free_indices = vcat(_sbp6_exp_left_diag_indices(N),
                             _sbp6_exp_right_diag_indices(N;
                                                          outer_boundary_closure_help = outer_boundary_closure_help))
    if !isnothing(v_diag_block)
        length(v_diag_block) == length(diag_free_indices) ||
            throw(DimensionMismatch("`v_diag_block` must have length $(length(diag_free_indices)) for N=$N."))
        @inbounds for (k, i) in enumerate(diag_free_indices)
            vdiag[i] = convert(T, v_diag_block[k])
        end
    end

    pairs = sbp6_exp_v_offdiag_pairs(N; outer_boundary_closure_help = outer_boundary_closure_help)
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
    sbp6_exp_vector_mass(S::SparseMatrixCSC; kwargs...)

Convenience overload that extracts `s_diag` from diagonal scalar mass `S` and
delegates to `sbp6_exp_vector_mass(sdiag; kwargs...)`.
"""
function sbp6_exp_vector_mass(S::SparseMatrixCSC{T, Ti};
                              v11 = one(T),
                              v_diag_block::Union{Nothing, AbstractVector} = nothing,
                              v_offdiag_block::Union{Nothing, AbstractVector} = nothing,
                              outer_boundary_closure_help::Bool = true) where {
                                                                                               T <:
                                                                                               Real,
                                                                                               Ti <:
                                                                                               Integer
                                                                                               }
    sdiag = _sbp6_extract_diagonal(S)
    return sbp6_exp_vector_mass(sdiag;
                                v11 = v11,
                                v_diag_block = v_diag_block,
                                v_offdiag_block = v_offdiag_block,
                                outer_boundary_closure_help = outer_boundary_closure_help)
end

@inline sbp6_exp_scalar_mass_gradient(args...; kwargs...) = sbp6_scalar_mass_gradient(args...;
                                                                                      kwargs...)

@inline sbp6_exp_construct_divergence(args...; kwargs...) = sbp6_construct_divergence(args...;
                                                                                      kwargs...)

function _sbp6_exp_constraint_rows(N::Int,
                                   first_rows_r5::Int,
                                   closure_right::Int;
                                   outer_boundary_closure_help::Bool = true)
    rows_all = collect(1:N)
    rows_r3_end = outer_boundary_closure_help ? N : max(0, N - 6)
    rows_r5 = collect(1:min(max(0, first_rows_r5), N))
    return (rows_r = rows_all,
            rows_r3 = rows_r3_end == 0 ? Int[] : collect(1:rows_r3_end),
            rows_r5 = rows_r5,
            closure_right = clamp(closure_right, 0, N))
end

"""
    sbp6_exp_solve_accuracy_constraints(setup; first_rows_r5=3,
                                        exact_solve=true, verbose=true)

Solve the experimental non-diagonal SBP6 linear system using:

- `D = S^{-1}(B - G^T V)`
- `D*r = 3` everywhere
- `D*r^3 = 5r^2` everywhere
- `D*r^5 = 7r^4` on the first `first_rows_r5` rows
- `ones' * S * ones = R^3 / 3`
"""
function sbp6_exp_solve_accuracy_constraints(setup::NamedTuple;
                                             first_rows_r5::Int = 3,
                                             outer_boundary_closure_help::Bool = true,
                                             exact_solve::Bool = true,
                                             verbose::Bool = true)
    hasproperty(setup, :r) || throw(ArgumentError("`setup` must include `r`."))
    hasproperty(setup, :Geven) || throw(ArgumentError("`setup` must include `Geven`."))
    hasproperty(setup, :p) || throw(ArgumentError("`setup` must include `p`."))
    hasproperty(setup, :R) || throw(ArgumentError("`setup` must include `R`."))

    r = setup.r
    Geven = setup.Geven
    p = Int(setup.p)
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    N = length(r)
    N >= 17 ||
        throw(ArgumentError("Experimental SBP6 requires `N >= 17`; got N=$N."))
    size(Geven) == (N, N) || throw(DimensionMismatch("`Geven` must be $(N)x$(N)."))

    closure_right = _closure_diagnostics(Geven).closure_width_right
    rows = _sbp6_exp_constraint_rows(N, first_rows_r5, closure_right;
                                     outer_boundary_closure_help = outer_boundary_closure_help)

    tmpl = sbp6_exp_vector_mass(N = N;
                                outer_boundary_closure_help = outer_boundary_closure_help)
    n_s = length(tmpl.s_diag_symbols)
    n_vdiag = length(tmpl.v_diag_symbols)
    n_voff = length(tmpl.v_offdiag_symbols)
    n_unknowns = n_s + n_vdiag + n_voff

    diag_free_indices = vcat(tmpl.left_diag_indices, tmpl.right_diag_indices)
    diag_free_pos = Dict{Int, Int}(j => (n_s + k) for (k, j) in enumerate(diag_free_indices))
    idx_s(i) = i
    idx_vdiag(j) = diag_free_pos[j]
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
        push!(tags, (family = family, degree = degree, row = row_index))
        return nothing
    end

    pairs = tmpl.v_offdiag_pairs
    interior_indices = tmpl.interior_diag_indices
    BNN = rq[end]^p

    function add_degree_constraints!(degree::Int, rows_idx::Vector{Int}, family::Symbol)
        isempty(rows_idx) && return nothing
        u = rq .^ degree
        coeff = convert(Tq, p + degree)
        @inbounds for i in rows_idx
            row = zeros(Tq, n_unknowns)
            exact_i = coeff * rq[i]^(degree - 1)

            row[idx_s(i)] -= exact_i

            for j in interior_indices
                gij = Gq[j, i]
                gij == 0 // 1 && continue
                row[idx_s(j)] += -(gij * u[j])
            end

            for j in diag_free_indices
                gij = Gq[j, i]
                gij == 0 // 1 && continue
                row[idx_vdiag(j)] += -(gij * u[j])
            end

            for (kpair, (a, b)) in enumerate(pairs)
                coeff_pair = -(Gq[a, i] * u[b] + Gq[b, i] * u[a])
                coeff_pair == 0 // 1 && continue
                row[idx_voff(kpair)] += coeff_pair
            end

            constant_term = (i == N ? (BNN * u[i]) : (0 // 1)) - Gq[1, i] * u[1]
            rhs = -constant_term
            push_eq!(row, rhs, family, degree, i)
        end
        return nothing
    end

    add_degree_constraints!(1, rows.rows_r, :divergence)
    add_degree_constraints!(3, rows.rows_r3, :divergence)
    add_degree_constraints!(5, rows.rows_r5, :divergence)

    row_q = zeros(Tq, n_unknowns)
    @inbounds for i in 1:N
        row_q[idx_s(i)] = 1 // 1
    end
    rhs_q = Rq^(p + 1) / convert(Tq, p + 1)
    push_eq!(row_q, rhs_q, :quadrature, 0, 0)

    m = length(A_rows)
    A = m == 0 ? zeros(Tq, 0, n_unknowns) :
        reduce(vcat, (reshape(row, 1, :) for row in A_rows))
    bvec = collect(b_rows)

    exact_solve ||
        throw(ArgumentError("Only exact rational solve is supported; set `exact_solve=true`."))

    x_exact = _solve_exact_linear_system(A, bvec)
    residual_exact = m == 0 ? 0 // 1 : _sbp6_maxabs(abs.(A * x_exact .- bvec))

    Tm = Rational{BigInt}
    r_use = rq
    G_use = Gq
    sdiag = Tm[x_exact[idx_s(i)] for i in 1:N]
    v_diag_block = Tm[x_exact[idx_vdiag(j)] for j in diag_free_indices]
    v_offdiag_block = Tm[x_exact[idx_voff(k)] for k in 1:n_voff]

    S = spdiagm(0 => sdiag)
    V = sbp6_exp_vector_mass(sdiag;
                             v11 = one(Tm),
                             v_diag_block = v_diag_block,
                             v_offdiag_block = v_offdiag_block,
                             outer_boundary_closure_help = outer_boundary_closure_help)
    div_data = sbp6_exp_construct_divergence(S,
                                             V,
                                             G_use,
                                             r_use;
                                             p = p)
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
        println("SBP6 experimental solve mode: exact")
        println("  equations = ", m, ", unknowns = ", n_unknowns, ", exact residual = ",
                residual_exact)
        println("  closure width (right diagnostic) = ", rows.closure_right)
        println("  outer boundary closure help = ", outer_boundary_closure_help)
        println("  constraint max errors: D*r = ", err_r,
                ", D*r^3 = ", err_r3, ", D*r^5 = ", err_r5,
                ", quadrature = ", quad_err)
        println("  SPD checks: S symmetric = ", s_sym, ", S PD = ", s_pd,
                ", V symmetric = ", v_sym, ", V PD = ", v_pd)
    end

    return (mode = :exact,
            setup = setup,
            closure_right = rows.closure_right,
            outer_boundary_closure_help = outer_boundary_closure_help,
            row_sets = rows,
            template = tmpl,
            A = A,
            b = bvec,
            tags = tags,
            x = x_exact,
            residual = residual_exact,
            S = S,
            V = V,
            D = D,
            B = B,
            errors = (Dr = err_r,
                      Dr3 = err_r3,
                      Dr5 = err_r5,
                      quadrature = quad_err),
            spd = (S_symmetric = s_sym,
                   S_positive_definite = s_pd,
                   V_symmetric = v_sym,
                   V_positive_definite = v_pd))
end

"""
    sbp6_exp_solve_accuracy_constraints(source; kwargs...)

Convenience overload that first builds the folded SBP6 scalar-mass/gradient setup and
then solves the experimental split-mass constraints.
"""
function sbp6_exp_solve_accuracy_constraints(source;
                                             accuracy_order::Int = 6,
                                             points::Int = 21,
                                             h::Real = 1,
                                             N = points - 1,
                                             R = h * (points - 1),
                                             p::Int = 2,
                                             mode = SafeMode(),
                                             atol = nothing,
                                             first_rows_r5::Int = 3,
                                             outer_boundary_closure_help::Bool = true,
                                             exact_solve::Bool = true,
                                             verbose::Bool = true)
    setup = sbp6_exp_scalar_mass_gradient(source;
                                          accuracy_order = accuracy_order,
                                          points = points,
                                          h = h,
                                          N = N,
                                          R = R,
                                          p = p,
                                          mode = mode,
                                          atol = atol)

    return sbp6_exp_solve_accuracy_constraints(setup;
                                               first_rows_r5 = first_rows_r5,
                                               outer_boundary_closure_help = outer_boundary_closure_help,
                                               exact_solve = exact_solve,
                                               verbose = verbose)
end

"""
    sbp6_exp_operators(source, points; kwargs...)

Build experimental non-diagonal-mass SBP6 operators by solving the new split-mass
linear system and returning `(D, G, S, V, B)`.
"""
function sbp6_exp_operators(source,
                            points::Integer;
                            h::Real = 1,
                            accuracy_order::Int = 6,
                            p::Int = 2,
                            mode = SafeMode(),
                            atol = nothing,
                            first_rows_r5::Int = 3,
                            outer_boundary_closure_help::Bool = true,
                            exact_solve::Bool = true,
                            verbose::Bool = false)
    points_int = Int(points)
    points_int > 1 || throw(ArgumentError("`points` must be > 1."))

    solved = sbp6_exp_solve_accuracy_constraints(source;
                                                 accuracy_order = accuracy_order,
                                                 points = points_int,
                                                 h = h,
                                                 p = p,
                                                 mode = mode,
                                                 atol = atol,
                                                 first_rows_r5 = first_rows_r5,
                                                 outer_boundary_closure_help = outer_boundary_closure_help,
                                                 exact_solve = exact_solve,
                                                 verbose = verbose)

    return (D = solved.D,
            G = solved.setup.Geven,
            S = solved.S,
            V = solved.V,
            B = solved.B)
end
