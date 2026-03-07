using LinearAlgebra
using Random
using SparseArrays
using SummationByPartsOperators
using SphericalSBPOperators

const Mod = SphericalSBPOperators
rb(x) = Mod._sbp8_as_big_rational(x)

function parse_q(s::AbstractString)
    t = strip(s)
    if occursin("/", t)
        a, b = split(t, "/"; limit=2)
        return parse(BigInt, strip(a)) // parse(BigInt, strip(b))
    end
    return parse(BigInt, t) // big(1)
end

function parse_bool(envkey::String, default::String)
    raw = lowercase(strip(get(ENV, envkey, default)))
    return raw in ("1", "true", "yes", "y", "on")
end

function parse_bigint_list(envkey::String, default_csv::String)
    raw = get(ENV, envkey, default_csv)
    vals = BigInt[]
    for tok in split(raw, ',')
        t = strip(tok)
        isempty(t) && continue
        push!(vals, parse(BigInt, t))
    end
    isempty(vals) && error("No valid values parsed from $envkey='$raw'")
    return vals
end

const STRATEGY = lowercase(get(ENV, "SBP8_STRATEGY", "dense_nullspace"))
const FR7 = parse(Int, get(ENV, "SBP8_FR7", "11"))
const BLOCK_STOP = parse(Int, get(ENV, "SBP8_BLOCK_STOP", "16"))
const DIAG_FREE_END = parse(Int, get(ENV, "SBP8_DIAG_FREE_END", "24"))
const TAIL_START = parse(Int, get(ENV, "SBP8_TAIL_START", "24"))
const FIX_S1 = parse_bool("SBP8_FIX_S1", "true")
const S1_ANCHOR = parse_q(get(ENV, "SBP8_S1_ANCHOR", "11/20"))
const V11_FIX = parse_q(get(ENV, "SBP8_V11_FIX", "1/1"))
const STOP_ON_STRICT = parse_bool("SBP8_STOP_ON_STRICT", "false")

const DENSE_MAX_COLS = parse(Int, get(ENV, "SBP8_DENSE_MAX_COLS", "24"))
const DENSE_MAX_PASSES = parse(Int, get(ENV, "SBP8_DENSE_MAX_PASSES", "6"))
const DENSE_DENS = parse_bigint_list("SBP8_DENSE_DENS", "1000,10000,100000")

const GREEDY_ITERS = parse(Int, get(ENV, "SBP8_GREEDY_ITERS", "240"))
const GREEDY_TEMP0 = parse(Float64, get(ENV, "SBP8_GREEDY_TEMP0", "0.25"))
const GREEDY_COOL = parse(Float64, get(ENV, "SBP8_GREEDY_COOL", "0.985"))
const GREEDY_W_IMAG = parse(Float64, get(ENV, "SBP8_GREEDY_W_IMAG", "4.0"))
const GREEDY_SEED = parse(Int, get(ENV, "SBP8_GREEDY_SEED", "20260307"))

const RESULTS_DIR = get(
    ENV,
    "SBP8_RESULTS_DIR",
    "data/sbp8_server_search_2026-03-07",
)
const OUT_BASENAME = get(ENV, "SBP8_OUT_BASENAME", "sbp8_structure_strategy")

const IMAG_TOL = 0.0
const REAL_NEG_TOL = 0.0
const ZERO_Q = big(0) // big(1)

function add_fix_col!(
    A::Matrix{Rational{BigInt}},
    b::Vector{Rational{BigInt}},
    col::Int,
    val::Rational{BigInt},
)
    row = zeros(Rational{BigInt}, size(A, 2))
    row[col] = 1 // 1
    return vcat(A, reshape(row, 1, :)), vcat(b, [val])
end

function safe_cond(M::AbstractMatrix{Float64})
    try
        c = cond(M)
        return isfinite(c) ? c : Inf
    catch
        return Inf
    end
end

function ensure_dir(path::String)
    isdir(path) || mkpath(path)
end

function pair_string(pr::Tuple{Int,Int})
    return "v[$(pr[1]),$(pr[2])]"
end

function rref_pivots(A::Matrix{Rational{BigInt}})
    n_eq, n_vars = size(A)
    n_eq == 0 && return Int[], collect(1:n_vars)

    M = copy(A)
    pivot_cols = Int[]
    pivot_row = 1

    for col in 1:n_vars
        pivot = 0
        for r in pivot_row:n_eq
            if M[r, col] != ZERO_Q
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

        for r in 1:n_eq
            r == pivot_row && continue
            fac = M[r, col]
            fac == ZERO_Q && continue
            M[r, :] .-= fac .* M[pivot_row, :]
        end

        push!(pivot_cols, col)
        pivot_row += 1
        pivot_row > n_eq && break
    end

    pivot_set = Set(pivot_cols)
    free_cols = [c for c in 1:n_vars if !(c in pivot_set)]
    return pivot_cols, free_cols
end

function dense_pairs(block_stop::Int)
    pairs = Tuple{Int,Int}[]
    for i in 2:(block_stop - 1)
        for j in (i + 1):block_stop
            push!(pairs, (i, j))
        end
    end
    return pairs
end

function parity_pairs(block_stop::Int)
    pairs = Tuple{Int,Int}[]
    for d in 2:(block_stop - 2)
        iseven(d) || continue
        for i in 2:(block_stop - d)
            j = i + d
            push!(pairs, (i, j))
        end
    end
    return pairs
end

function tapered_bandwidth(i::Int)
    if i <= 4
        return 6
    elseif i <= 8
        return 4
    else
        return 2
    end
end

function mirror_taper_pairs(G::AbstractMatrix{<:Real}, block_stop::Int)
    pairs = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()
    for i in 2:(block_stop - 1)
        bw = tapered_bandwidth(i)
        jmax = min(block_stop, i + bw)
        have_mirror = false
        for j in (i + 1):jmax
            if G[i, j] != 0 || G[j, i] != 0
                pr = (i, j)
                if !(pr in seen)
                    push!(pairs, pr)
                    push!(seen, pr)
                end
                have_mirror = true
            end
        end

        # Keep tapered width even if stencil support is too compact at this row.
        if !have_mirror
            for j in (i + 1):jmax
                pr = (i, j)
                if !(pr in seen)
                    push!(pairs, pr)
                    push!(seen, pr)
                end
            end
        end
    end
    return pairs
end

function choose_pairs(strategy::String, setup, block_stop::Int)
    if strategy == "dense_nullspace" || strategy == "greedy_gap"
        return dense_pairs(block_stop)
    elseif strategy == "parity_incremental"
        return parity_pairs(block_stop)
    elseif strategy == "mirror_incremental"
        Gf = Matrix{Float64}(sparse(setup.Geven))
        return mirror_taper_pairs(Gf, block_stop)
    else
        error("Unknown SBP8_STRATEGY='$strategy'.")
    end
end

function build_problem(strategy::String)
    setup = Mod.sbp8_scalar_mass_gradient(
        Mattsson2017(:central);
        accuracy_order=8,
        points=31,
        h=1,
        p=2,
        build_matrix=:probe,
    )

    N = length(setup.r)
    3 <= BLOCK_STOP <= N || error("SBP8_BLOCK_STOP must satisfy 3 <= BLOCK_STOP <= $N.")
    1 <= DIAG_FREE_END <= N || error("SBP8_DIAG_FREE_END must satisfy 1 <= DIAG_FREE_END <= $N.")
    1 <= TAIL_START <= N || error("SBP8_TAIL_START must satisfy 1 <= TAIL_START <= $N.")

    closure_right = Mod._sbp8_infer_right_boundary_closure(setup, sparse(setup.Geven))
    rows = Mod._sbp8_constraint_rows(N, closure_right, FR7)
    diag_free_indices = collect(1:DIAG_FREE_END)
    pairs = choose_pairs(strategy, setup, BLOCK_STOP)
    isempty(pairs) && error("No off-diagonal pairs generated for strategy=$strategy and block_stop=$BLOCK_STOP.")

    sys = Mod._sbp8_build_exact_constraint_system(
        setup,
        rows;
        diag_free_indices=diag_free_indices,
        v_offdiag_pairs=pairs,
        s1_target=nothing,
    )

    A = copy(sys.A)
    b = copy(sys.b)
    idx = sys.indexing
    pair_to_k = Dict{Tuple{Int,Int},Int}(pr => k for (k, pr) in enumerate(pairs))
    sq = Rational{BigInt}[rb(si) for si in setup.Sdiag]

    for i in TAIL_START:N
        A, b = add_fix_col!(A, b, i, sq[i])
    end

    if FIX_S1
        A, b = add_fix_col!(A, b, 1, S1_ANCHOR)
    end

    col_vdiag_tail = get(idx.vdiag_to_col, TAIL_START, 0)
    if col_vdiag_tail > 0
        A, b = add_fix_col!(A, b, col_vdiag_tail, sq[TAIL_START])
    end

    col_v11 = get(idx.vdiag_to_col, 1, 0)
    col_v11 > 0 || error("vdiag[1] is not free with DIAG_FREE_END=$DIAG_FREE_END.")
    A, b = add_fix_col!(A, b, col_v11, V11_FIX)

    Gf = Matrix{Float64}(sparse(sys.G))

    return (
        strategy=strategy,
        setup=setup,
        sys=sys,
        N=N,
        idx=idx,
        pairs=pairs,
        pair_to_k=pair_to_k,
        diag_free_indices=diag_free_indices,
        baseA=A,
        baseb=b,
        cond_G=safe_cond(Gf),
        block_stop=BLOCK_STOP,
    )
end

function fail_result(problem, status::String, reason::String)
    return (
        status=status,
        reason=reason,
        strict_hard=false,
        rho=Inf,
        max_real=Inf,
        max_imag=Inf,
        cond_D=Inf,
        cond_G=problem.cond_G,
        cond_L=Inf,
        min_s=-Inf,
        s11=-Inf,
        v_boundary_pd=false,
        v_full_pd=false,
        x=nothing,
    )
end

function evaluate_solution(problem, x::Vector{Rational{BigInt}})
    idx = problem.idx
    N = problem.N

    sdiag_q = [x[i] for i in 1:idx.n_s]
    min_s = minimum(Float64.(sdiag_q))
    s11 = Float64(sdiag_q[1])
    s_pos = min_s > 0.0 && s11 > 0.0

    if !s_pos
        return (
            status="s_nonpositive",
            reason="min_s=$min_s, s11=$s11",
            strict_hard=false,
            rho=Inf,
            max_real=Inf,
            max_imag=Inf,
            cond_D=Inf,
            cond_G=problem.cond_G,
            cond_L=Inf,
            min_s=min_s,
            s11=s11,
            v_boundary_pd=false,
            v_full_pd=false,
            x=x,
        )
    end

    vdiag_q = Vector{Rational{BigInt}}(undef, N)
    for i in 1:N
        c = get(idx.vdiag_to_col, i, 0)
        vdiag_q[i] = c > 0 ? x[c] : sdiag_q[i]
    end
    voff_q = [x[idx.idx_voff(k)] for k in 1:idx.n_voff]

    S = spdiagm(0 => sdiag_q)
    V = spdiagm(0 => vdiag_q)
    @inbounds for (k, (i, j)) in enumerate(problem.pairs)
        v = voff_q[k]
        if v != ZERO_Q
            V[i, j] = v
            V[j, i] = v
        end
    end

    D = Mod.sbp8_construct_divergence(S, V, problem.sys.G, problem.sys.r; p=problem.sys.p).D
    Df = Matrix{Float64}(sparse(D))
    Lf = Matrix{Float64}(sparse(D * problem.sys.G))

    ev = eigen(Lf).values
    rho = maximum(abs.(ev))
    max_real = maximum(real.(ev))
    max_imag = maximum(abs.(imag.(ev)))

    pd_stop = min(problem.block_stop, N)
    v_boundary_pd = try
        cholesky(Symmetric(Matrix{Float64}(V[1:pd_stop, 1:pd_stop])); check=true)
        true
    catch
        false
    end
    v_full_pd = Mod._sbp8_is_pd(V)

    strict_hard = s_pos &&
                  v_full_pd &&
                  max_imag == IMAG_TOL &&
                  max_real < REAL_NEG_TOL

    return (
        status="ok",
        reason="",
        strict_hard=strict_hard,
        rho=rho,
        max_real=max_real,
        max_imag=max_imag,
        cond_D=safe_cond(Df),
        cond_G=problem.cond_G,
        cond_L=safe_cond(Lf),
        min_s=min_s,
        s11=s11,
        v_boundary_pd=v_boundary_pd,
        v_full_pd=v_full_pd,
        x=x,
    )
end

function solve_with_extra_fixes(problem, fixed_cols::Vector{Int}, fixed_vals::Vector{Rational{BigInt}})
    A = problem.baseA
    b = problem.baseb
    for (col, val) in zip(fixed_cols, fixed_vals)
        A, b = add_fix_col!(A, b, col, val)
    end
    eq_count = size(A, 1)
    unknown_count = size(A, 2)
    try
        x = Mod._solve_exact_linear_system(A, b)
        return evaluate_solution(problem, x), eq_count, unknown_count
    catch err
        return fail_result(problem, "solve_fail", sprint(showerror, err)), eq_count, unknown_count
    end
end

function build_step_system(problem, active_count::Int)
    A = problem.baseA
    b = problem.baseb
    npairs = length(problem.pairs)
    @assert 0 <= active_count <= npairs
    for k in (active_count + 1):npairs
        A, b = add_fix_col!(A, b, problem.idx.idx_voff(k), ZERO_Q)
    end
    return A, b
end

function solve_system(problem, A::Matrix{Rational{BigInt}}, b::Vector{Rational{BigInt}})
    eq_count = size(A, 1)
    unknown_count = size(A, 2)
    try
        x = Mod._solve_exact_linear_system(A, b)
        return evaluate_solution(problem, x), eq_count, unknown_count
    catch err
        return fail_result(problem, "solve_fail", sprint(showerror, err)), eq_count, unknown_count
    end
end

function result_score(res)
    status_penalty = res.status == "ok" ? 0 : 1
    s_penalty = (res.status == "ok" && res.min_s > 0.0 && res.s11 > 0.0) ? 0 : 1
    v_penalty = (res.status == "ok" && res.v_full_pd) ? 0 : 1
    imag_gap = res.status == "ok" ? abs(res.max_imag - IMAG_TOL) : Inf
    real_gap = res.status == "ok" ? max(res.max_real, REAL_NEG_TOL) : Inf
    return (
        status_penalty,
        s_penalty,
        v_penalty,
        imag_gap,
        real_gap,
        res.max_imag,
        res.max_real,
        res.cond_L,
        res.cond_D,
        res.cond_G,
        res.rho,
    )
end

function stability_distance(res)
    if res.status != "ok"
        return Inf
    end
    spd_penalty = res.v_full_pd ? 0.0 : 1.0e3
    s_penalty = (res.min_s > 0.0 && res.s11 > 0.0) ? 0.0 : 1.0e3
    return max(res.max_real, 0.0) +
           GREEDY_W_IMAG * res.max_imag +
           1.0e-3 * res.rho +
           spd_penalty +
           s_penalty
end

function col_label(problem, col::Int)
    idx = problem.idx
    if col <= idx.n_s
        return "s[$col]"
    end
    if col <= idx.n_s + idx.n_vdiag
        k = col - idx.n_s
        i = problem.diag_free_indices[k]
        return "vdiag[$i]"
    end
    k = col - idx.n_s - idx.n_vdiag
    if 1 <= k <= length(problem.pairs)
        i, j = problem.pairs[k]
        return "voff[$i,$j]"
    end
    return "col[$col]"
end

function dense_candidate_cols(problem, free_cols::Vector{Int})
    free_set = Set(free_cols)
    cols = Int[]

    # Prioritize dense V-block off-diagonals near origin.
    for (k, (i, j)) in enumerate(problem.pairs)
        i <= problem.block_stop && j <= problem.block_stop || continue
        c = problem.idx.idx_voff(k)
        if c in free_set && !(c in cols)
            push!(cols, c)
        end
    end

    # Then V diagonals and S diagonals in the same region.
    for i in 2:problem.block_stop
        c = get(problem.idx.vdiag_to_col, i, 0)
        if c > 0 && (c in free_set) && !(c in cols)
            push!(cols, c)
        end
    end
    for i in 2:problem.block_stop
        c = i
        if c in free_set && !(c in cols)
            push!(cols, c)
        end
    end

    # Keep v11 available if it remains free in the chosen system.
    c11 = get(problem.idx.vdiag_to_col, 1, 0)
    if c11 > 0 && (c11 in free_set) && !(c11 in cols)
        push!(cols, c11)
    end

    if DENSE_MAX_COLS > 0 && length(cols) > DENSE_MAX_COLS
        return cols[1:DENSE_MAX_COLS]
    end
    return cols
end

function write_header(io)
    println(
        io,
        join(
            [
                "strategy",
                "mode",
                "iter",
                "step",
                "new_pair",
                "active_pairs",
                "n_pairs_total",
                "status",
                "strict_hard",
                "stability_distance",
                "rho",
                "max_real",
                "max_imag",
                "cond_D",
                "cond_G",
                "cond_L",
                "min_s",
                "s11",
                "v_boundary_pd",
                "v_full_pd",
                "eq_count",
                "unknown_count",
                "n_fixed",
                "reason",
            ],
            '\t',
        ),
    )
end

function write_row(
    io,
    strategy::String,
    mode::String,
    iter::Int,
    step::Int,
    new_pair::String,
    active_pairs::Int,
    n_pairs_total::Int,
    res,
    eq_count::Int,
    unknown_count::Int,
    n_fixed::Int,
)
    println(
        io,
        join(
            [
                strategy,
                mode,
                string(iter),
                string(step),
                new_pair,
                string(active_pairs),
                string(n_pairs_total),
                res.status,
                string(res.strict_hard),
                string(stability_distance(res)),
                string(res.rho),
                string(res.max_real),
                string(res.max_imag),
                string(res.cond_D),
                string(res.cond_G),
                string(res.cond_L),
                string(res.min_s),
                string(res.s11),
                string(res.v_boundary_pd),
                string(res.v_full_pd),
                string(eq_count),
                string(unknown_count),
                string(n_fixed),
                res.reason,
            ],
            '\t',
        ),
    )
end

function run_incremental!(io, problem)
    npairs = length(problem.pairs)
    strict_count = 0
    row_count = 0
    best = nothing

    for step in 0:npairs
        new_pair = step == 0 ? "none" : pair_string(problem.pairs[step])
        A, b = build_step_system(problem, step)
        res, eq_count, unknown_count = solve_system(problem, A, b)
        write_row(
            io,
            problem.strategy,
            "incremental",
            row_count,
            step,
            new_pair,
            step,
            npairs,
            res,
            eq_count,
            unknown_count,
            npairs - step,
        )
        flush(io)

        row_count += 1
        if res.strict_hard
            strict_count += 1
        end
        if isnothing(best) || result_score(res) < result_score(best.res)
            best = (res=res, row=row_count, label=new_pair, active_pairs=step)
        end

        println(
            "step=", lpad(step, 3),
            " active=", lpad(step, 3), "/", npairs,
            " status=", res.status,
            " strict=", res.strict_hard,
            " max_real=", res.max_real,
            " max_imag=", res.max_imag,
        )

        if STOP_ON_STRICT && res.strict_hard
            break
        end
    end

    return (strict_count=strict_count, row_count=row_count, best=best, extra="")
end

function run_dense_nullspace!(io, problem)
    npairs = length(problem.pairs)
    base_res, base_eq, base_unknown = solve_with_extra_fixes(problem, Int[], Rational{BigInt}[])
    pivot_cols, free_cols = rref_pivots(problem.baseA)
    cand_cols = dense_candidate_cols(problem, free_cols)

    write_row(
        io,
        problem.strategy,
        "baseline",
        0,
        0,
        "none",
        npairs,
        npairs,
        base_res,
        base_eq,
        base_unknown,
        0,
    )

    strict_count = base_res.strict_hard ? 1 : 0
    row_count = 1
    best = (res=base_res, row=1, label="baseline", active_pairs=npairs)

    if isempty(cand_cols)
        return (
            strict_count=strict_count,
            row_count=row_count,
            best=best,
            extra="free_cols=$(length(free_cols));candidate_cols=0",
        )
    end

    xbase = isnothing(base_res.x) ? zeros(Rational{BigInt}, size(problem.baseA, 2)) : base_res.x
    current = Dict{Int,Rational{BigInt}}(c => xbase[c] for c in cand_cols)
    current_res = base_res
    iter = 0

    for den in DENSE_DENS
        step = big(1) // den
        for pass in 1:DENSE_MAX_PASSES
            improved = false
            for col in cand_cols
                base_val = current[col]
                for sgn in (-1, 1)
                    iter += 1
                    trial = copy(current)
                    trial[col] = base_val + sgn * step
                    fixed_cols = sort(collect(keys(trial)))
                    fixed_vals = [trial[c] for c in fixed_cols]
                    res, eq_count, unknown_count = solve_with_extra_fixes(problem, fixed_cols, fixed_vals)

                    write_row(
                        io,
                        problem.strategy,
                        "dense_r$(den)_p$(pass)",
                        iter,
                        pass,
                        col_label(problem, col),
                        npairs,
                        npairs,
                        res,
                        eq_count,
                        unknown_count,
                        length(fixed_cols),
                    )
                    flush(io)

                    row_count += 1
                    if res.strict_hard
                        strict_count += 1
                    end
                    if result_score(res) < result_score(best.res)
                        best = (res=res, row=row_count, label=col_label(problem, col), active_pairs=npairs)
                    end
                    if result_score(res) < result_score(current_res)
                        current = trial
                        current_res = res
                        improved = true
                    end
                end
            end
            !improved && break
        end
    end

    extra = "pivot_cols=$(length(pivot_cols));free_cols=$(length(free_cols));candidate_cols=$(length(cand_cols))"
    return (strict_count=strict_count, row_count=row_count, best=best, extra=extra)
end

function solve_for_active(problem, active::BitVector)
    A = problem.baseA
    b = problem.baseb
    for (k, on) in enumerate(active)
        if !on
            A, b = add_fix_col!(A, b, problem.idx.idx_voff(k), ZERO_Q)
        end
    end
    res, eq_count, unknown_count = solve_system(problem, A, b)
    return res, eq_count, unknown_count
end

function run_greedy_gap!(io, problem)
    npairs = length(problem.pairs)
    rng = MersenneTwister(GREEDY_SEED)

    active = falses(npairs)
    curr_res, curr_eq, curr_unknown = solve_for_active(problem, active)
    curr_sd = stability_distance(curr_res)
    strict_count = curr_res.strict_hard ? 1 : 0
    row_count = 0
    best = (res=curr_res, row=0, label="start_empty", active_pairs=count(active))

    write_row(
        io,
        problem.strategy,
        "greedy_init",
        0,
        0,
        "none",
        count(active),
        npairs,
        curr_res,
        curr_eq,
        curr_unknown,
        npairs - count(active),
    )
    row_count += 1

    for iter in 1:GREEDY_ITERS
        k = rand(rng, 1:npairs)
        trial = copy(active)
        trial[k] = !trial[k]
        trial_res, trial_eq, trial_unknown = solve_for_active(problem, trial)
        trial_sd = stability_distance(trial_res)

        T = max(1.0e-10, GREEDY_TEMP0 * GREEDY_COOL^(iter - 1))
        accept = false
        if isfinite(trial_sd) && !isfinite(curr_sd)
            accept = true
        elseif trial_sd < curr_sd
            accept = true
        elseif isfinite(trial_sd) && isfinite(curr_sd)
            accept = rand(rng) < exp((curr_sd - trial_sd) / T)
        end

        if accept
            active = trial
            curr_res = trial_res
            curr_sd = trial_sd
        end

        label = pair_string(problem.pairs[k]) * (trial[k] ? ":on" : ":off")
        reason = trial_res.reason
        if !isempty(reason)
            reason = (accept ? "accepted|" : "rejected|") * reason
        else
            reason = accept ? "accepted" : "rejected"
        end
        row_res = merge(trial_res, (reason=reason,))

        write_row(
            io,
            problem.strategy,
            "greedy",
            iter,
            iter,
            label,
            count(active),
            npairs,
            row_res,
            trial_eq,
            trial_unknown,
            npairs - count(active),
        )
        flush(io)

        row_count += 1
        if curr_res.strict_hard
            strict_count += 1
        end
        if result_score(curr_res) < result_score(best.res)
            best = (res=curr_res, row=row_count, label=label, active_pairs=count(active))
        end

        println(
            "iter=", lpad(iter, 4),
            " active=", lpad(count(active), 3), "/", npairs,
            " accepted=", accept,
            " sd=", curr_sd,
            " max_real=", curr_res.max_real,
            " max_imag=", curr_res.max_imag,
        )

        if STOP_ON_STRICT && curr_res.strict_hard
            break
        end
    end

    return (strict_count=strict_count, row_count=row_count, best=best, extra="iters=$(GREEDY_ITERS)")
end

function write_summary(path::String, problem, stats)
    open(path, "w") do io
        println(io, "SBP8 V-structure strategy search")
        println(io, "strategy = ", problem.strategy)
        println(io, "fr7 = ", FR7)
        println(io, "block_stop = ", BLOCK_STOP)
        println(io, "diag_free_indices = 1:", DIAG_FREE_END)
        println(io, "tail_fixed_start = ", TAIL_START)
        println(io, "fix_s1 = ", FIX_S1, FIX_S1 ? " (s11 anchor = $(S1_ANCHOR))" : "")
        println(io, "v11 fixed = ", V11_FIX)
        println(io, "hard spectral checks: max_imag == ", IMAG_TOL, ", max_real < ", REAL_NEG_TOL)
        println(io, "hard constraints: exact solve, S diag > 0, V full PD")
        println(io)
        println(io, "n_pairs = ", length(problem.pairs))
        println(io, "rows_written = ", stats.row_count)
        println(io, "strict_count = ", stats.strict_count)
        println(io, "extra = ", stats.extra)
        println(io, "cond(G) = ", problem.cond_G)
        println(io)
        println(io, "best row:")
        println(io, "  row = ", stats.best.row, ", label = ", stats.best.label, ", active_pairs = ", stats.best.active_pairs)
        println(io, "  status = ", stats.best.res.status, ", strict_hard = ", stats.best.res.strict_hard)
        println(io, "  rho = ", stats.best.res.rho)
        println(io, "  max_real = ", stats.best.res.max_real)
        println(io, "  max_imag = ", stats.best.res.max_imag)
        println(io, "  cond_D = ", stats.best.res.cond_D)
        println(io, "  cond_G = ", stats.best.res.cond_G)
        println(io, "  cond_L = ", stats.best.res.cond_L)
        println(io, "  min_s = ", stats.best.res.min_s, ", s11 = ", stats.best.res.s11)
        println(io, "  v_boundary_pd = ", stats.best.res.v_boundary_pd, ", v_full_pd = ", stats.best.res.v_full_pd)
    end
end

function main()
    ensure_dir(RESULTS_DIR)
    problem = build_problem(STRATEGY)

    tsv_path = joinpath(RESULTS_DIR, "$(OUT_BASENAME).tsv")
    summary_path = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_summary.txt")

    println("SBP8 structure search starting...")
    println("  strategy=", STRATEGY, " fr7=", FR7, " block_stop=", BLOCK_STOP, " npairs=", length(problem.pairs))
    println("  output=", tsv_path)

    stats = open(tsv_path, "w") do io
        write_header(io)
        if STRATEGY == "dense_nullspace"
            run_dense_nullspace!(io, problem)
        elseif STRATEGY == "parity_incremental" || STRATEGY == "mirror_incremental"
            run_incremental!(io, problem)
        elseif STRATEGY == "greedy_gap"
            run_greedy_gap!(io, problem)
        else
            error("Unknown SBP8_STRATEGY='$STRATEGY'.")
        end
    end

    write_summary(summary_path, problem, stats)
    println("Wrote: ", tsv_path)
    println("Wrote: ", summary_path)
end

main()
