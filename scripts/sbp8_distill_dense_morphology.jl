using LinearAlgebra
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
    isempty(vals) && error("No valid values parsed from $envkey='$raw'.")
    return vals
end

const FR7 = parse(Int, get(ENV, "SBP8_FR7", "11"))
const BLOCK_STOP = parse(Int, get(ENV, "SBP8_BLOCK_STOP", "16"))
const DIAG_FREE_END = parse(Int, get(ENV, "SBP8_DIAG_FREE_END", "24"))
const TAIL_START = parse(Int, get(ENV, "SBP8_TAIL_START", "24"))

const FIX_S1 = parse_bool("SBP8_FIX_S1", "true")
const S1_ANCHOR = parse_q(get(ENV, "SBP8_S1_ANCHOR", "11/20"))
const V11_FIX = parse_q(get(ENV, "SBP8_V11_FIX", "1/1"))

const EXTRACT_THRESH = parse(Float64, get(ENV, "SBP8_EXTRACT_THRESH", "1e-4"))

const DENSE_MAX_COLS = parse(Int, get(ENV, "SBP8_DENSE_MAX_COLS", "32"))
const DENSE_MAX_PASSES = parse(Int, get(ENV, "SBP8_DENSE_MAX_PASSES", "6"))
const DENSE_DENS = parse_bigint_list("SBP8_DENSE_DENS", "1000,10000,100000")

const REFINE_MAX_COLS = parse(Int, get(ENV, "SBP8_REFINE_MAX_COLS", "48"))
const REFINE_MAX_PASSES = parse(Int, get(ENV, "SBP8_REFINE_MAX_PASSES", "6"))
const REFINE_DENS = parse_bigint_list("SBP8_REFINE_DENS", "1000,10000,100000")

const RESULTS_DIR = get(ENV, "SBP8_RESULTS_DIR", "data/sbp8_server_search_2026-03-07")
const OUT_BASENAME = get(ENV, "SBP8_OUT_BASENAME", "sbp8_distill_dense")

const IMAG_TOL = 0.0
const REAL_NEG_TOL = 0.0
const ZERO_Q = big(0) // big(1)

function ensure_dir(path::String)
    isdir(path) || mkpath(path)
end

function pair_string(pr::Tuple{Int,Int})
    return "($(pr[1]),$(pr[2]))"
end

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

function build_problem(pairs::Vector{Tuple{Int,Int}})
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

    pivot_cols, free_cols = rref_pivots(A)
    Gf = Matrix{Float64}(sparse(sys.G))

    return (
        setup=setup,
        sys=sys,
        N=N,
        idx=idx,
        pairs=pairs,
        pair_to_k=pair_to_k,
        diag_free_indices=diag_free_indices,
        baseA=A,
        baseb=b,
        pivot_cols=pivot_cols,
        free_cols=free_cols,
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

function pd_score(res)
    if res.status != "ok"
        return (1, Inf, Inf, Inf)
    end
    pd_penalty = res.v_full_pd ? 0 : 1
    return (
        pd_penalty,
        abs(res.max_imag - IMAG_TOL),
        max(res.max_real, REAL_NEG_TOL),
        res.rho,
    )
end

function refine_score(res)
    hard = res.status == "ok" &&
           res.v_full_pd &&
           res.min_s > 0.0 &&
           res.s11 > 0.0 &&
           res.max_imag == IMAG_TOL &&
           res.max_real < REAL_NEG_TOL
    if hard
        return (0, res.rho, res.cond_L, res.cond_D)
    end
    if res.status != "ok"
        return (2, Inf, Inf, Inf)
    end
    return (
        1,
        abs(res.max_imag - IMAG_TOL),
        max(res.max_real, REAL_NEG_TOL),
        res.rho,
    )
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

function dense_candidate_cols(problem)
    free_set = Set(problem.free_cols)
    cols = Int[]

    for (k, (i, j)) in enumerate(problem.pairs)
        i <= problem.block_stop && j <= problem.block_stop || continue
        c = problem.idx.idx_voff(k)
        if c in free_set && !(c in cols)
            push!(cols, c)
        end
    end
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
    if DENSE_MAX_COLS > 0 && length(cols) > DENSE_MAX_COLS
        cols = cols[1:DENSE_MAX_COLS]
    end
    return cols
end

function refine_candidate_cols(problem)
    free_set = Set(problem.free_cols)
    cols = Int[]

    for k in 1:length(problem.pairs)
        c = problem.idx.idx_voff(k)
        if c in free_set && !(c in cols)
            push!(cols, c)
        end
    end
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

    if REFINE_MAX_COLS > 0 && length(cols) > REFINE_MAX_COLS
        cols = cols[1:REFINE_MAX_COLS]
    end
    return cols
end

function extract_pairs(problem, x::Vector{Rational{BigInt}}, thresh::Float64)
    selected = Tuple{Int,Int}[]
    magnitudes = Float64[]
    for (k, pr) in enumerate(problem.pairs)
        v = x[problem.idx.idx_voff(k)]
        a = abs(Float64(v))
        if a > thresh
            push!(selected, pr)
            push!(magnitudes, a)
        end
    end
    return selected, magnitudes
end

function write_dense_trace_header(io)
    println(
        io,
        join(
            [
                "iter",
                "den",
                "pass",
                "col",
                "col_label",
                "status",
                "strict_hard",
                "rho",
                "max_real",
                "max_imag",
                "min_s",
                "s11",
                "v_full_pd",
                "eq_count",
                "unknown_count",
                "reason",
            ],
            '\t',
        ),
    )
end

function write_dense_trace_row(io, iter::Int, den::BigInt, pass::Int, col::Int, res, eq_count::Int, unknown_count::Int)
    println(
        io,
        join(
            [
                string(iter),
                string(den),
                string(pass),
                string(col),
                col == 0 ? "baseline" : "",
                res.status,
                string(res.strict_hard),
                string(res.rho),
                string(res.max_real),
                string(res.max_imag),
                string(res.min_s),
                string(res.s11),
                string(res.v_full_pd),
                string(eq_count),
                string(unknown_count),
                res.reason,
            ],
            '\t',
        ),
    )
end

function run_dense_distill(problem, trace_path::String)
    cols = dense_candidate_cols(problem)
    baseline_res, baseline_eq, baseline_unknown = solve_with_extra_fixes(problem, Int[], Rational{BigInt}[])

    xbase = if baseline_res.status == "ok" && !isnothing(baseline_res.x)
        baseline_res.x
    else
        zeros(Rational{BigInt}, size(problem.baseA, 2))
    end
    current = Dict{Int,Rational{BigInt}}(c => xbase[c] for c in cols)
    current_res = baseline_res

    best_overall = (res=baseline_res, label="baseline")
    best_pd = (res=baseline_res, label="baseline")
    if pd_score(baseline_res) > (0, Inf, Inf, Inf)
        best_pd = (res=fail_result(problem, "no_pd_baseline", "baseline not PD"), label="none")
    end

    iter = 0
    rows = 0

    open(trace_path, "w") do io
        write_dense_trace_header(io)
        write_dense_trace_row(io, iter, big(0), 0, 0, baseline_res, baseline_eq, baseline_unknown)
        rows += 1

        for den in DENSE_DENS
            step = big(1) // den
            for pass in 1:DENSE_MAX_PASSES
                improved = false
                for col in cols
                    base_val = get(current, col, ZERO_Q)
                    for sgn in (-1, 1)
                        iter += 1
                        trial = copy(current)
                        trial[col] = base_val + sgn * step
                        fixed_cols = sort(collect(keys(trial)))
                        fixed_vals = [trial[c] for c in fixed_cols]
                        res, eq_count, unknown_count = solve_with_extra_fixes(problem, fixed_cols, fixed_vals)

                        write_dense_trace_row(io, iter, den, pass, col, res, eq_count, unknown_count)
                        rows += 1
                        flush(io)

                        if result_score(res) < result_score(best_overall.res)
                            best_overall = (res=res, label=col_label(problem, col))
                        end
                        if pd_score(res) < pd_score(best_pd.res)
                            best_pd = (res=res, label=col_label(problem, col))
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
    end

    return (
        best_overall=best_overall,
        best_pd=best_pd,
        rows=rows,
        candidate_cols=cols,
    )
end

function run_refinement(problem, xseed::Vector{Rational{BigInt}}, trace_path::String)
    cols = refine_candidate_cols(problem)
    current = Dict{Int,Rational{BigInt}}(c => xseed[c] for c in cols)
    current_res = evaluate_solution(problem, xseed)
    best = (res=current_res, label="seed")
    iter = 0
    rows = 0

    open(trace_path, "w") do io
        write_dense_trace_header(io)
        write_dense_trace_row(io, iter, big(0), 0, 0, current_res, size(problem.baseA, 1), size(problem.baseA, 2))
        rows += 1

        for den in REFINE_DENS
            step = big(1) // den
            for pass in 1:REFINE_MAX_PASSES
                improved = false
                for col in cols
                    base_val = get(current, col, xseed[col])
                    for sgn in (-1, 1)
                        iter += 1
                        trial = copy(current)
                        trial[col] = base_val + sgn * step
                        fixed_cols = sort(collect(keys(trial)))
                        fixed_vals = [trial[c] for c in fixed_cols]
                        res, eq_count, unknown_count = solve_with_extra_fixes(problem, fixed_cols, fixed_vals)
                        write_dense_trace_row(io, iter, den, pass, col, res, eq_count, unknown_count)
                        rows += 1
                        flush(io)

                        if refine_score(res) < refine_score(best.res)
                            best = (res=res, label=col_label(problem, col))
                        end
                        if refine_score(res) < refine_score(current_res)
                            current = trial
                            current_res = res
                            improved = true
                        end
                    end
                end
                !improved && break
            end
        end
    end

    return (
        best=best,
        rows=rows,
        candidate_cols=cols,
    )
end

function write_pairs_file(path::String, pairs::Vector{Tuple{Int,Int}}, mags::Vector{Float64})
    open(path, "w") do io
        println(io, "pair\tabs_value")
        for (pr, a) in zip(pairs, mags)
            println(io, "(", pr[1], ",", pr[2], ")\t", a)
        end
    end
end

function write_solution_vector(path::String, problem, x::Vector{Rational{BigInt}})
    open(path, "w") do io
        println(io, "col\tlabel\tvalue")
        for c in 1:length(x)
            println(io, c, '\t', col_label(problem, c), '\t', x[c])
        end
    end
end

function pairs_repr(pairs::Vector{Tuple{Int,Int}})
    if isempty(pairs)
        return "[]"
    end
    return "[" * join(["(" * string(i) * "," * string(j) * ")" for (i, j) in pairs], ", ") * "]"
end

function main()
    ensure_dir(RESULTS_DIR)
    dense_trace = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_dense_trace.tsv")
    refine_trace = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_refine_trace.tsv")
    pairs_file = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_pairs_gt_thresh.tsv")
    dense_vec = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_dense_best_x.tsv")
    reduced_vec = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_reduced_best_x.tsv")
    summary_file = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_summary.txt")

    dense_problem = build_problem(dense_pairs(BLOCK_STOP))
    println("Dense distill run starting...")
    println("  fr7=", FR7, " block_stop=", BLOCK_STOP, " npairs_dense=", length(dense_problem.pairs))

    dense_stats = run_dense_distill(dense_problem, dense_trace)
    dense_pd_res = dense_stats.best_pd.res

    if dense_pd_res.status != "ok" || isnothing(dense_pd_res.x)
        open(summary_file, "w") do io
            println(io, "Dense PD candidate not found.")
            println(io, "best_pd_status = ", dense_pd_res.status)
            println(io, "reason = ", dense_pd_res.reason)
        end
        println("No PD dense candidate found; wrote summary and exiting.")
        return
    end

    write_solution_vector(dense_vec, dense_problem, dense_pd_res.x)
    selected_pairs, mags = extract_pairs(dense_problem, dense_pd_res.x, EXTRACT_THRESH)
    write_pairs_file(pairs_file, selected_pairs, mags)

    println("Extracted pairs above threshold=", EXTRACT_THRESH, ": ", length(selected_pairs))

    # Reduced-pattern consistency check.
    reduced_ok = true
    reduced_res = nothing
    reduced_problem = nothing
    reduced_refine = nothing

    if isempty(selected_pairs)
        reduced_ok = false
    else
        reduced_problem = build_problem(selected_pairs)
        reduced_res, _, _ = solve_with_extra_fixes(reduced_problem, Int[], Rational{BigInt}[])
        if reduced_res.status != "ok"
            reduced_ok = false
        end
    end

    if reduced_ok
        println("Reduced pattern is exactly solvable; starting refinement.")
        reduced_refine = run_refinement(reduced_problem, reduced_res.x, refine_trace)
        if reduced_refine.best.res.status == "ok" && !isnothing(reduced_refine.best.res.x)
            write_solution_vector(reduced_vec, reduced_problem, reduced_refine.best.res.x)
        end
    end

    open(summary_file, "w") do io
        println(io, "SBP8 dense-to-sparse morphology distillation")
        println(io, "fr7 = ", FR7)
        println(io, "block_stop = ", BLOCK_STOP)
        println(io, "extract_threshold = ", EXTRACT_THRESH)
        println(io)
        println(io, "[1] Dense search result (PD target)")
        println(io, "dense_rows = ", dense_stats.rows)
        println(io, "dense_candidate_cols = ", length(dense_stats.candidate_cols))
        println(io, "dense_best_label = ", dense_stats.best_pd.label)
        println(io, "dense_status = ", dense_pd_res.status)
        println(io, "dense_max_real = ", dense_pd_res.max_real)
        println(io, "dense_max_imag = ", dense_pd_res.max_imag)
        println(io, "dense_rho = ", dense_pd_res.rho)
        println(io, "dense_v_full_pd = ", dense_pd_res.v_full_pd)
        println(io)
        println(io, "[2] Extracted sparse ansatz")
        println(io, "n_pairs_gt_thresh = ", length(selected_pairs))
        println(io, "pairs = ", pairs_repr(selected_pairs))
        println(io)
        println(io, "[3] Reduced-pattern consistency")
        println(io, "reduced_solvable = ", reduced_ok)
        if reduced_ok
            println(io, "reduced_status = ", reduced_res.status)
            println(io, "reduced_max_real = ", reduced_res.max_real)
            println(io, "reduced_max_imag = ", reduced_res.max_imag)
            println(io, "reduced_rho = ", reduced_res.rho)
            println(io, "reduced_v_full_pd = ", reduced_res.v_full_pd)
        else
            println(io, "reduced_status = fail")
            if !isnothing(reduced_res)
                println(io, "reduced_reason = ", reduced_res.reason)
            end
        end
        println(io)
        println(io, "[4] Reduced-pattern refinement")
        if reduced_ok
            b = reduced_refine.best.res
            println(io, "refine_rows = ", reduced_refine.rows)
            println(io, "refine_candidate_cols = ", length(reduced_refine.candidate_cols))
            println(io, "refine_best_label = ", reduced_refine.best.label)
            println(io, "refine_status = ", b.status)
            println(io, "refine_strict_hard = ", b.strict_hard)
            println(io, "refine_max_real = ", b.max_real)
            println(io, "refine_max_imag = ", b.max_imag)
            println(io, "refine_rho = ", b.rho)
            println(io, "refine_v_full_pd = ", b.v_full_pd)
        else
            println(io, "refinement_skipped = true")
        end
    end

    println("Wrote: ", dense_trace)
    println("Wrote: ", pairs_file)
    println("Wrote: ", summary_file)
    if reduced_ok
        println("Wrote: ", refine_trace)
        println("Wrote: ", reduced_vec)
    end
end

main()
