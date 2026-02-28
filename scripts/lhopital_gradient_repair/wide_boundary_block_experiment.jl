include(joinpath(@__DIR__, "lhopital_gradient_repair_core.jl"))

using LinearAlgebra: Diagonal, norm

const RatBig = Rational{BigInt}

function _exact_ranks_augmented(A::Matrix{RatBig}, b::Vector{RatBig})
    m, n = size(A)
    m == length(b) || throw(DimensionMismatch("Incompatible system dimensions."))
    M = hcat(copy(A), copy(b))

    pivot_row = 1
    rankA = 0
    rankAug = 0

    for col in 1:(n + 1)
        pivot = 0
        for r in pivot_row:m
            if M[r, col] != 0 // 1
                pivot = r
                break
            end
        end
        pivot == 0 && continue

        if pivot != pivot_row
            M[pivot_row, :], M[pivot, :] = M[pivot, :], M[pivot_row, :]
        end

        piv = M[pivot_row, col]
        M[pivot_row, :] ./= piv

        for r in 1:m
            r == pivot_row && continue
            fac = M[r, col]
            fac == 0 // 1 && continue
            M[r, :] .-= fac .* M[pivot_row, :]
        end

        rankAug += 1
        if col <= n
            rankA += 1
        end

        pivot_row += 1
        pivot_row > m && break
    end

    return (rankA = rankA, rankAug = rankAug, consistent = rankA == rankAug)
end

function _build_wide_block_system(G::Matrix{RatBig},
                                  r::Vector{RatBig},
                                  Mdiag::Vector{RatBig},
                                  Bdiag::Vector{RatBig},
                                  p::Int,
                                  q_even::Vector{Int},
                                  q_odd::Vector{Int},
                                  Nb::Int)
    Nh = length(r)
    2 <= Nb <= Nh || throw(ArgumentError("Need 2 <= Nb <= Nh (got Nb=$Nb, Nh=$Nh)."))

    rows = collect(2:Nb)
    cols = collect(2:Nb)
    rowset = Set(rows)
    colset = Set(cols)

    n_unknowns = length(rows) * length(cols)
    n_eq_grad = length(rows) * length(q_even)
    n_eq_div = length(rows) * length(q_odd)
    n_eq = n_eq_grad + n_eq_div

    A = zeros(RatBig, n_eq, n_unknowns)
    b = zeros(RatBig, n_eq)
    var_index = Dict{Tuple{Int, Int}, Int}()

    idx = 1
    for j in rows
        for k in cols
            var_index[(j, k)] = idx
            idx += 1
        end
    end

    eq = 1

    # Constraint Set A: gradient exactness on even monomials for rows j in [2, Nb].
    for j in rows
        for q in q_even
            rhs = q == 0 ? 0 // 1 : (big(q) // 1) * (r[j]^(q - 1))
            fixed = 0 // 1

            for k in 1:Nh
                k in colset && continue
                fixed += G[j, k] * (r[k]^q)
            end

            b[eq] = rhs - fixed
            for k in cols
                A[eq, var_index[(j, k)]] = r[k]^q
            end
            eq += 1
        end
    end

    # Constraint Set B: divergence exactness on odd monomials for rows k in [2, Nb].
    # Use the full SBP-translated relation, keeping out-of-block row contributions fixed.
    for k in rows
        Mk = Mdiag[k]
        Mk != 0 // 1 || throw(ArgumentError("Encountered zero Mdiag[$k]."))

        for q in q_odd
            exact = (big(q + p) // 1) * (r[k]^(q - 1))

            fixed_numer = Bdiag[k] * (r[k]^q)
            for j in 1:Nh
                j in rowset && continue
                fixed_numer -= G[j, k] * Mdiag[j] * (r[j]^q)
            end
            rhs = exact - fixed_numer / Mk

            b[eq] = rhs
            for j in rows
                A[eq, var_index[(j, k)]] = -(Mdiag[j] / Mk) * (r[j]^q)
            end
            eq += 1
        end
    end

    return (
            A = A,
            b = b,
            rows = rows,
            cols = cols,
            var_index = var_index,
            n_eq_grad = n_eq_grad,
            n_eq_div = n_eq_div,
           )
end

function _apply_wide_block_solution!(G::Matrix{T},
                                     rows::Vector{Int},
                                     cols::Vector{Int},
                                     var_index::Dict{Tuple{Int, Int}, Int},
                                     x::AbstractVector) where {T}
    for j in rows
        for k in cols
            G[j, k] = convert(T, x[var_index[(j, k)]])
        end
    end
    return G
end

function _max_constraint_violation(G::AbstractMatrix,
                                   r::AbstractVector,
                                   Mdiag::AbstractVector,
                                   Bdiag::AbstractVector,
                                   p::Int,
                                   q_even::Vector{Int},
                                   q_odd::Vector{Int},
                                   rows::Vector{Int},
                                   cols::Vector{Int})
    colset = Set(cols)
    rowset = Set(rows)

    grad_max = zero(eltype(G))
    div_max = zero(eltype(G))

    for j in rows
        for q in q_even
            rhs = q == 0 ? zero(eltype(G)) : convert(eltype(G), q) * (r[j]^(q - 1))
            lhs = zero(eltype(G))
            for k in 1:length(r)
                lhs += G[j, k] * (r[k]^q)
            end
            e = abs(lhs - rhs)
            e > grad_max && (grad_max = e)
        end
    end

    for k in rows
        Mk = Mdiag[k]
        for q in q_odd
            exact = convert(eltype(G), q + p) * (r[k]^(q - 1))
            lhs = (Bdiag[k] * (r[k]^q))
            for j in 1:length(r)
                lhs -= G[j, k] * Mdiag[j] * (r[j]^q)
            end
            lhs /= Mk
            e = abs(lhs - exact)
            e > div_max && (div_max = e)
        end
    end

    # Optional leakage indicator outside solved rows for q=1.
    q1_leak = zero(eltype(G))
    if 1 in q_odd
        for k in 2:length(r)-1
            k in rowset && continue
            Mk = Mdiag[k]
            exact = convert(eltype(G), 1 + p) * (r[k]^0)
            lhs = (Bdiag[k] * (r[k]^1))
            for j in 1:length(r)
                lhs -= G[j, k] * Mdiag[j] * (r[j]^1)
            end
            lhs /= Mk
            e = abs(lhs - exact)
            e > q1_leak && (q1_leak = e)
        end
    end

    return (grad_max = grad_max, div_max = div_max, q1_outside_rows_max = q1_leak)
end

function run_wide_boundary_block_experiment(; accuracy_order::Int = 6,
                                            npoints::Int = 41,
                                            spatial_dim::Int = 3,
                                            Nb::Int = 15,
                                            w1::RatBig = 1 // 6,
                                            include_even_zero::Bool = true,
                                            include_odd_q1::Bool = true)
    cfg = LHopitalGradientRepairConfig(accuracy_order = accuracy_order,
                                       Nhalf = npoints,
                                       spatial_dim = spatial_dim,
                                       w1_mode = :user,
                                       w1_user = w1,
                                       stencil_mode = :raw_pattern_expand_right)

    p = cfg.spatial_dim - 1
    cart = _build_cartesian_canonical(cfg)
    fold = _build_folding_maps(cart.xfull)
    r = fold.r
    Nh = length(r)
    Nb <= Nh || throw(ArgumentError("Nb=$Nb exceeds Nh=$Nh."))

    G_raw = fold.Rop * cart.Dcart * fold.Eeven
    G_odd = fold.Rop * cart.Dcart * fold.Eodd

    M_half = (1 // 2) * transpose(fold.Eeven) * cart.Mcart * fold.Eeven
    Mdiag = zeros(RatBig, Nh)
    for j in 2:Nh
        Mdiag[j] = M_half[j, j] * (r[j]^p)
    end
    Mdiag[1] = w1
    any(m -> m <= 0 // 1, Mdiag) && throw(ArgumentError("Mass positivity violated."))

    Bdiag = _boundary_matrix_diag(Nh, r, p)
    D1 = (big(p + 1) // 1) .* G_odd[1, :]

    G = copy(G_raw)

    # Pole-column SBP lock for rows inside the dense block.
    for j in 2:Nb
        G[j, 1] = -(w1 * D1[j]) / Mdiag[j]
    end

    q_even = _even_monomial_powers(_max_even_leq(cart.q_infer))
    q_odd = _odd_monomial_powers(_max_odd_leq(cart.q_infer - p))
    include_even_zero || (q_even = [q for q in q_even if q != 0])
    include_odd_q1 || (q_odd = [q for q in q_odd if q != 1])

    sys = _build_wide_block_system(G, r, Mdiag, Bdiag, p, q_even, q_odd, Nb)
    rank_info = _exact_ranks_augmented(sys.A, sys.b)

    solve_mode = :exact
    x = nothing
    lsq_resid_max = 0.0
    grad_resid_max = 0.0
    div_resid_max = 0.0

    try
        x = _solve_exact_linear_system(sys.A, sys.b)
        _apply_wide_block_solution!(G, sys.rows, sys.cols, sys.var_index, x)
    catch err
        solve_mode = :lsq_bigfloat
        setprecision(BigFloat, 256) do
            A_big = BigFloat.(sys.A)
            b_big = BigFloat.(sys.b)
            x_big = A_big \ b_big
            resid = A_big * x_big - b_big
            lsq_resid_max = maximum(abs.(resid))

            grad_resid_max = maximum(abs.(resid[1:sys.n_eq_grad]))
            div_resid_max = maximum(abs.(resid[(sys.n_eq_grad + 1):end]))

            _apply_wide_block_solution!(G, sys.rows, sys.cols, sys.var_index, x_big)
        end
    end

    M = Matrix(Diagonal(Mdiag))
    B = Matrix(Diagonal(Bdiag))
    D = _compute_divergence_from_sbp(Mdiag, B, G)

    # Pole row consistency against prescribed LHopital row.
    pole_diff = maximum(abs.(D[1, :] .- D1))

    viol = _max_constraint_violation(G, r, Mdiag, Bdiag, p, q_even, q_odd, sys.rows, sys.cols)

    println("Wide Boundary Block experiment")
    println("  accuracy_order = ", accuracy_order, ", q_infer = ", cart.q_infer, ", p = ", p)
    println("  npoints = ", npoints, ", Nh = ", Nh, ", Nb = ", Nb, ", w1 = ", w1)
    println("  dense unknown block size = ", length(sys.rows), " x ", length(sys.cols), " = ", length(sys.rows) * length(sys.cols))
    println("  q_even = ", q_even, ", q_odd = ", q_odd)
    println("  equations: grad = ", sys.n_eq_grad, ", div = ", sys.n_eq_div, ", total = ", size(sys.A, 1))
    println("  rank(A) = ", rank_info.rankA, ", rank([A|b]) = ", rank_info.rankAug, ", consistent = ", rank_info.consistent)
    println("  solve_mode = ", solve_mode)
    if solve_mode == :lsq_bigfloat
        println("  LSQ residual max (all eq) = ", lsq_resid_max)
        println("  LSQ residual max (grad eq) = ", grad_resid_max)
        println("  LSQ residual max (div eq) = ", div_resid_max)
    end
    println("  pole row max abs(D[1,:] - D1) = ", pole_diff)
    println("  max constraint violation (grad moments on rows 2..Nb) = ", viol.grad_max)
    println("  max constraint violation (div moments on rows 2..Nb) = ", viol.div_max)
    println("  q=1 divergence leakage max outside rows 2..Nb = ", viol.q1_outside_rows_max)

    return (
            cfg = cfg,
            q_even = q_even,
            q_odd = q_odd,
            ranks = rank_info,
            solve_mode = solve_mode,
            lsq_resid_max = lsq_resid_max,
            lsq_grad_resid_max = grad_resid_max,
            lsq_div_resid_max = div_resid_max,
            pole_row_max_diff = pole_diff,
            grad_violation_max = viol.grad_max,
            div_violation_max = viol.div_max,
            q1_outside_rows_max = viol.q1_outside_rows_max,
           )
end

function _parse_int_arg(args::Vector{String}, key::String, default::Int)
    idx = findfirst(==(key), args)
    isnothing(idx) && return default
    idx < length(args) || throw(ArgumentError("Missing value for $key."))
    return parse(Int, args[idx + 1])
end

function _parse_rat_arg(args::Vector{String}, key::String, default::RatBig)
    idx = findfirst(==(key), args)
    isnothing(idx) && return default
    idx < length(args) || throw(ArgumentError("Missing value for $key."))
    return _parse_rational_token(args[idx + 1]) |> _rat
end

function main(args::Vector{String})
    if any(a -> a in ("-h", "--help"), args)
        println("Usage: julia scripts/lhopital_gradient_repair/wide_boundary_block_experiment.jl [options]")
        println("")
        println("Options:")
        println("  --accuracy-order <int>    (default: 6)")
        println("  --npoints <int>           points on [0,R] including origin and boundary (default: 41)")
        println("  --spatial-dim <int>       (default: 3)")
        println("  --nb <int>                dense boundary block size Nb (default: 15)")
        println("  --w1 <num/den|num>        origin mass weight (default: 1/6)")
        println("  --drop-even-zero          remove q=0 from gradient constraints")
        println("  --drop-odd-q1             remove q=1 from divergence constraints")
        return nothing
    end

    accuracy_order = _parse_int_arg(args, "--accuracy-order", 6)
    npoints = _parse_int_arg(args, "--npoints", 41)
    spatial_dim = _parse_int_arg(args, "--spatial-dim", 3)
    Nb = _parse_int_arg(args, "--nb", 15)
    w1 = _parse_rat_arg(args, "--w1", big(1) // big(6))
    include_even_zero = !("--drop-even-zero" in args)
    include_odd_q1 = !("--drop-odd-q1" in args)

    return run_wide_boundary_block_experiment(;
                                              accuracy_order = accuracy_order,
                                              npoints = npoints,
                                              spatial_dim = spatial_dim,
                                              Nb = Nb,
                                              w1 = w1,
                                              include_even_zero = include_even_zero,
                                              include_odd_q1 = include_odd_q1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main(ARGS)
    catch err
        println(stderr, "Error: ", sprint(showerror, err))
        rethrow(err)
    end
end
