using LinearAlgebra
using SparseArrays
using Optim
using SummationByPartsOperators
using SphericalSBPOperators

const Mod = SphericalSBPOperators
rb(x) = Mod._sbp8_as_big_rational(x)

function parse_q(s::AbstractString)
    t = replace(strip(s), "//" => "/")
    if occursin("/", t)
        a, b = split(t, "/"; limit=2)
        return parse(BigInt, strip(a)) // parse(BigInt, strip(b))
    end
    if occursin(r"[\.eE]", t)
        r = rationalize(parse(BigFloat, t); tol=big"1e-30")
        return big(numerator(r)) // big(denominator(r))
    end
    return parse(BigInt, t) // big(1)
end

function parse_bool(envkey::String, default::String)
    raw = lowercase(strip(get(ENV, envkey, default)))
    return raw in ("1", "true", "yes", "y", "on")
end

function parse_f64_csv(envkey::String, default_csv::String)
    raw = get(ENV, envkey, default_csv)
    vals = Float64[]
    for tok in split(raw, ',')
        t = strip(tok)
        isempty(t) && continue
        push!(vals, parse(Float64, t))
    end
    isempty(vals) && error("No Float64 values parsed from $envkey='$raw'.")
    return vals
end

const FR7 = parse(Int, get(ENV, "SBP8_FR7", "11"))
const BLOCK_STOP = parse(Int, get(ENV, "SBP8_BLOCK_STOP", "16"))
const DIAG_FREE_END = parse(Int, get(ENV, "SBP8_DIAG_FREE_END", "24"))
const TAIL_START = parse(Int, get(ENV, "SBP8_TAIL_START", "24"))
const FIX_S1 = parse_bool("SBP8_FIX_S1", "true")
const S1_ANCHOR = parse_q(get(ENV, "SBP8_S1_ANCHOR", "11/20"))
const V11_FIX = parse_q(get(ENV, "SBP8_V11_FIX", "1/1"))

const PAIR_FILE = get(
    ENV,
    "SBP8_PAIR_FILE",
    "remote_results/sbp8_vsearch_2026-03-07/distill_dense_20260307_194005/main/results/sbp8_distill_dense_pairs_gt_thresh.tsv",
)
const ALPHA0_FILE = get(
    ENV,
    "SBP8_ALPHA0_FILE",
    "remote_results/sbp8_vsearch_2026-03-08/exact_morphology/exact_408925/results/sbp8_exact_morphology_alpha.tsv",
)
const REQUIRE_ALPHA0_FILE = parse_bool("SBP8_REQUIRE_ALPHA0_FILE", "true")
const TARGET_X_FILE = get(
    ENV,
    "SBP8_TARGET_X_FILE",
    "remote_results/sbp8_vsearch_2026-03-07/distill_dense_20260307_194005/main/results/sbp8_distill_dense_reduced_best_x.tsv",
)
const USE_TARGET_PROJECTION = parse_bool("SBP8_USE_TARGET_PROJECTION", "true")
const ALPHA_PROJ_TOL = parse(Float64, get(ENV, "SBP8_ALPHA_PROJ_TOL", "1e-8"))

const OPT_METHOD = lowercase(strip(get(ENV, "SBP8_OPT_METHOD", "neldermead")))
const OPT_MAX_ITERS = parse(Int, get(ENV, "SBP8_OPT_MAX_ITERS", "2000"))
const OPT_PENALTY_IMAG = parse(Float64, get(ENV, "SBP8_OPT_PENALTY_IMAG", "1000.0"))
const TARGET_MAX_REAL = parse(Float64, get(ENV, "SBP8_TARGET_MAX_REAL", "-1e-8"))
const FLOAT_IMAG_TOL = parse(Float64, get(ENV, "SBP8_FLOAT_IMAG_TOL", "1e-12"))
const RATIONAL_TOLS = parse_f64_csv("SBP8_RATIONAL_TOLS", "1e-8,1e-9,1e-10,1e-7,1e-6")

const RESULTS_DIR = get(ENV, "SBP8_RESULTS_DIR", "data/sbp8_server_search_2026-03-08")
const OUT_BASENAME = get(ENV, "SBP8_OUT_BASENAME", "sbp8_exact_subspace_opt")

const IMAG_TOL = 0.0
const REAL_NEG_TOL = 0.0
const ZERO_Q = big(0) // big(1)

function ensure_dir(path::String)
    isdir(path) || mkpath(path)
end

function safe_cond(M::AbstractMatrix{Float64})
    try
        c = cond(M)
        return isfinite(c) ? c : Inf
    catch
        return Inf
    end
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

function parse_pair_token(tok::AbstractString)
    t = strip(tok)
    m = match(r"^\((\d+)\s*,\s*(\d+)\)$", t)
    m === nothing && error("Invalid pair token '$tok'.")
    i = parse(Int, m.captures[1])
    j = parse(Int, m.captures[2])
    i < j || error("Pair must satisfy i<j, got ($i,$j).")
    return (i, j)
end

function load_pairs(path::String)
    isfile(path) || error("Pair file not found: $path")
    pairs = Tuple{Int,Int}[]
    open(path, "r") do io
        first = true
        for ln in eachline(io)
            line = strip(ln)
            isempty(line) && continue
            if first
                first = false
                if startswith(lowercase(line), "pair")
                    continue
                end
            end
            fields = split(line, '\t')
            push!(pairs, parse_pair_token(fields[1]))
        end
    end
    isempty(pairs) && error("No pairs parsed from $path.")
    return pairs
end

function load_x_vector(path::String)
    isfile(path) || error("Target x file not found: $path")
    entries = Tuple{Int,Rational{BigInt}}[]
    open(path, "r") do io
        first = true
        for ln in eachline(io)
            line = strip(ln)
            isempty(line) && continue
            if first
                first = false
                if startswith(lowercase(line), "col")
                    continue
                end
            end
            fields = split(line, '\t')
            length(fields) >= 2 || continue
            c = parse(Int, strip(fields[1]))
            v = parse_q(strip(fields[end]))
            push!(entries, (c, v))
        end
    end
    isempty(entries) && error("No entries parsed from target x file: $path")
    n = maximum(first.(entries))
    x = fill(ZERO_Q, n)
    for (c, v) in entries
        x[c] = v
    end
    return x
end

function load_alpha_vector(path::String, nfree::Int)
    isfile(path) || error("Alpha file not found: $path")
    alpha = fill(ZERO_Q, nfree)
    open(path, "r") do io
        first = true
        for ln in eachline(io)
            line = strip(ln)
            isempty(line) && continue
            if first
                first = false
                if startswith(lowercase(line), "basis_idx")
                    continue
                end
            end
            fields = split(line, '\t')
            length(fields) >= 2 || continue
            i = parse(Int, strip(fields[1]))
            1 <= i <= nfree || error("basis_idx $i out of range 1:$nfree in $path")
            alpha[i] = parse_q(strip(fields[2]))
        end
    end
    return alpha
end

function rref_matrix(A::Matrix{Rational{BigInt}})
    n_eq, n_vars = size(A)
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
    return M, pivot_cols, free_cols
end

function rational_nullspace_basis(A::Matrix{Rational{BigInt}})
    M, pivot_cols, free_cols = rref_matrix(A)
    n_vars = size(A, 2)
    n_free = length(free_cols)
    N = zeros(Rational{BigInt}, n_vars, n_free)

    for (k, fcol) in enumerate(free_cols)
        v = zeros(Rational{BigInt}, n_vars)
        v[fcol] = 1 // 1
        for (r, pcol) in enumerate(pivot_cols)
            v[pcol] = -M[r, fcol]
        end
        N[:, k] .= v
    end
    return N, pivot_cols, free_cols
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
    1 <= DIAG_FREE_END <= N || error("SBP8_DIAG_FREE_END must satisfy 1 <= BLOCK_STOP <= $N.")
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
    vdiag_cols = [get(idx.vdiag_to_col, i, 0) for i in 1:N]
    voff_cols = [idx.idx_voff(k) for k in 1:idx.n_voff]

    return (
        setup=setup,
        sys=sys,
        N=N,
        idx=idx,
        pairs=pairs,
        diag_free_indices=diag_free_indices,
        A=A,
        b=b,
        cond_G=safe_cond(Gf),
        block_stop=BLOCK_STOP,
        vdiag_cols=vdiag_cols,
        voff_cols=voff_cols,
    )
end

function compose_x(xp::Vector{Rational{BigInt}}, Nbasis::Matrix{Rational{BigInt}}, alpha::Vector{Rational{BigInt}})
    x = copy(xp)
    @inbounds for j in 1:length(alpha)
        aj = alpha[j]
        aj == ZERO_Q && continue
        x .+= aj .* Nbasis[:, j]
    end
    return x
end

function rationalize_big(x::Float64, tol::Float64)
    r = rationalize(x; tol=tol)
    return big(numerator(r)) // big(denominator(r))
end

function project_target_alpha_float(
    xp::Vector{Rational{BigInt}},
    Nbasis::Matrix{Rational{BigInt}},
    x_target::Vector{Rational{BigInt}},
)
    length(x_target) == length(xp) || error("Target x length mismatch: $(length(x_target)) != $(length(xp)).")

    Nf = Matrix{Float64}(Nbasis)
    delta = Float64.(x_target .- xp)
    alpha_float = try
        Nf \ delta
    catch
        pinv(Nf) * delta
    end

    fit_err = norm(Nf * alpha_float - delta)
    return alpha_float, fit_err
end

function exact_residual_stats(A::Matrix{Rational{BigInt}}, b::Vector{Rational{BigInt}}, x::Vector{Rational{BigInt}})
    r = A * x - b
    all_zero = true
    max_num = big(0)
    for ri in r
        if ri != ZERO_Q
            all_zero = false
        end
        anum = abs(numerator(ri))
        if anum > max_num
            max_num = anum
        end
    end
    return (all_zero=all_zero, max_num=max_num)
end

function enforce_exact_residual!(
    A::Matrix{Rational{BigInt}},
    b::Vector{Rational{BigInt}},
    x::Vector{Rational{BigInt}},
    label::AbstractString,
)
    stats = exact_residual_stats(A, b, x)
    if !stats.all_zero
        error(
            "FATAL: Exact accuracy constraints violated for " *
            label *
            "! Max numerator error: $(stats.max_num)",
        )
    end
    return stats
end

function evaluate_x_float(problem, x::Vector{Float64})
    idx = problem.idx
    N = problem.N

    sdiag = x[1:idx.n_s]
    min_s = minimum(sdiag)
    s11 = sdiag[1]
    if !(isfinite(min_s) && isfinite(s11) && min_s > 0.0 && s11 > 0.0)
        return (
            status="s_nonpositive",
            reason="min_s=$min_s, s11=$s11",
            obj=Inf,
            rho=Inf,
            max_real=Inf,
            max_imag=Inf,
            cond_D=Inf,
            cond_G=problem.cond_G,
            cond_L=Inf,
            min_s=min_s,
            s11=s11,
            v_full_pd=false,
        )
    end

    vdiag = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        c = problem.vdiag_cols[i]
        vdiag[i] = c > 0 ? x[c] : sdiag[i]
    end

    S = spdiagm(0 => sdiag)
    V = spdiagm(0 => vdiag)
    @inbounds for (k, (i, j)) in enumerate(problem.pairs)
        v = x[problem.voff_cols[k]]
        if v != 0.0
            V[i, j] = v
            V[j, i] = v
        end
    end

    v_full_pd = try
        cholesky(Symmetric(Matrix{Float64}(V)); check=true)
        true
    catch
        false
    end
    if !v_full_pd
        return (
            status="v_not_pd",
            reason="V not SPD",
            obj=Inf,
            rho=Inf,
            max_real=Inf,
            max_imag=Inf,
            cond_D=Inf,
            cond_G=problem.cond_G,
            cond_L=Inf,
            min_s=min_s,
            s11=s11,
            v_full_pd=false,
        )
    end

    D = try
        Mod.sbp8_construct_divergence(S, V, problem.sys.G, problem.sys.r; p=problem.sys.p).D
    catch err
        return (
            status="divergence_fail",
            reason=sprint(showerror, err),
            obj=Inf,
            rho=Inf,
            max_real=Inf,
            max_imag=Inf,
            cond_D=Inf,
            cond_G=problem.cond_G,
            cond_L=Inf,
            min_s=min_s,
            s11=s11,
            v_full_pd=true,
        )
    end

    Df = Matrix{Float64}(sparse(D))
    Lf = Matrix{Float64}(sparse(D * problem.sys.G))
    ev = eigen(Lf).values
    rho = maximum(abs.(ev))
    max_real = maximum(real.(ev))
    max_imag = maximum(abs.(imag.(ev)))
    obj = max_real + OPT_PENALTY_IMAG * max_imag
    if !isfinite(obj)
        obj = Inf
    end

    return (
        status="ok",
        reason="",
        obj=obj,
        rho=rho,
        max_real=max_real,
        max_imag=max_imag,
        cond_D=safe_cond(Df),
        cond_G=problem.cond_G,
        cond_L=safe_cond(Lf),
        min_s=min_s,
        s11=s11,
        v_full_pd=true,
    )
end

function evaluate_x_rational(problem, x::Vector{Rational{BigInt}})
    idx = problem.idx
    N = problem.N

    sdiag_q = [x[i] for i in 1:idx.n_s]
    min_s = minimum(Float64.(sdiag_q))
    s11 = Float64(sdiag_q[1])

    if !(min_s > 0.0 && s11 > 0.0)
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
            v_full_pd=false,
        )
    end

    vdiag_q = Vector{Rational{BigInt}}(undef, N)
    for i in 1:N
        c = problem.vdiag_cols[i]
        vdiag_q[i] = c > 0 ? x[c] : sdiag_q[i]
    end

    S = spdiagm(0 => sdiag_q)
    V = spdiagm(0 => vdiag_q)
    @inbounds for (k, (i, j)) in enumerate(problem.pairs)
        v = x[problem.voff_cols[k]]
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
    v_full_pd = Mod._sbp8_is_pd(V)
    strict_hard = v_full_pd && max_imag == IMAG_TOL && max_real < REAL_NEG_TOL

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
        v_full_pd=v_full_pd,
    )
end

function build_matrices(problem, x::Vector{Rational{BigInt}})
    idx = problem.idx
    N = problem.N
    sdiag_q = [x[i] for i in 1:idx.n_s]

    vdiag_q = Vector{Rational{BigInt}}(undef, N)
    for i in 1:N
        c = problem.vdiag_cols[i]
        vdiag_q[i] = c > 0 ? x[c] : sdiag_q[i]
    end

    Sm = fill(ZERO_Q, N, N)
    Vm = fill(ZERO_Q, N, N)
    for i in 1:N
        Sm[i, i] = sdiag_q[i]
        Vm[i, i] = vdiag_q[i]
    end
    for (k, (i, j)) in enumerate(problem.pairs)
        v = x[problem.voff_cols[k]]
        if v != ZERO_Q
            Vm[i, j] = v
            Vm[j, i] = v
        end
    end
    return Sm, Vm
end

function write_matrix(path::String, name::String, M::Matrix{Rational{BigInt}})
    open(path, "w") do io
        println(io, name, " = Rational{BigInt}[")
        for i in 1:size(M, 1)
            row = join(string.(M[i, :]), ", ")
            if i < size(M, 1)
                println(io, "  ", row, ";")
            else
                println(io, "  ", row)
            end
        end
        println(io, "]")
    end
end

function write_alpha_q(path::String, alpha::Vector{Rational{BigInt}})
    open(path, "w") do io
        println(io, "basis_idx\talpha")
        for (i, a) in enumerate(alpha)
            println(io, i, '\t', a)
        end
    end
end

function write_alpha_f(path::String, alpha::Vector{Float64})
    open(path, "w") do io
        println(io, "basis_idx\talpha_float")
        for (i, a) in enumerate(alpha)
            println(io, i, '\t', a)
        end
    end
end

function write_pairs(path::String, pairs::Vector{Tuple{Int,Int}})
    open(path, "w") do io
        println(io, "pairs = [", join(["(" * string(i) * "," * string(j) * ")" for (i, j) in pairs], ", "), "]")
    end
end

function run_subspace_opt(problem, xp, Nbasis, alpha_start::Vector{Float64}, trace_path::String)
    method = OPT_METHOD == "neldermead" ? NelderMead() : error("Unsupported SBP8_OPT_METHOD='$OPT_METHOD'. Use 'neldermead'.")

    xpf = Float64.(xp)
    Nf = Matrix{Float64}(Nbasis)
    nfree = size(Nf, 2)
    length(alpha_start) == nfree || error("alpha_start length mismatch: $(length(alpha_start)) != $nfree")

    best_alpha = copy(alpha_start)
    best_res = evaluate_x_float(problem, xpf .+ Nf * alpha_start)
    best_obj = best_res.obj
    eval_count = Ref(0)

    trace_io = open(trace_path, "w")
    println(trace_io, "eval\tobj\tmax_real\tmax_imag\tv_full_pd\tstatus\treason")

    function objfun(alpha::Vector{Float64})
        xtrial = xpf .+ Nf * alpha
        res = evaluate_x_float(problem, xtrial)
        eval_count[] += 1

        if res.obj < best_obj
            best_obj = res.obj
            best_alpha = copy(alpha)
            best_res = res
        end

        println(
            trace_io,
            eval_count[],
            '\t',
            res.obj,
            '\t',
            res.max_real,
            '\t',
            res.max_imag,
            '\t',
            res.v_full_pd,
            '\t',
            res.status,
            '\t',
            res.reason,
        )
        if eval_count[] % 25 == 0
            flush(trace_io)
        end
        return res.obj
    end

    function cb(_state)
        return best_res.v_full_pd &&
               best_res.max_imag <= FLOAT_IMAG_TOL &&
               best_res.max_real < TARGET_MAX_REAL
    end

    options = Optim.Options(
        iterations=OPT_MAX_ITERS,
        show_trace=true,
        store_trace=false,
        callback=cb,
        allow_f_increases=true,
    )

    opt_result = optimize(objfun, alpha_start, method, options)

    flush(trace_io)
    close(trace_io)

    alpha_opt = Optim.minimizer(opt_result)
    res_opt = evaluate_x_float(problem, xpf .+ Nf * alpha_opt)

    if res_opt.obj < best_obj
        best_obj = res_opt.obj
        best_alpha = copy(alpha_opt)
        best_res = res_opt
    end

    return (
        result=opt_result,
        alpha_opt=alpha_opt,
        res_opt=res_opt,
        best_alpha=best_alpha,
        best_res=best_res,
        eval_count=eval_count[],
    )
end

function main()
    ensure_dir(RESULTS_DIR)

    trace_path = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_optim_trace.tsv")
    summary_path = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_summary.txt")
    rat_scan_path = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_rational_tolerance_scan.tsv")
    pairs_out = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_pairs.jl")
    alpha_float_out = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_alpha_float.tsv")
    alpha_rat_out = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_alpha_rational.tsv")
    s_out = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_S_matrix.jl")
    v_out = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_V_matrix.jl")

    pairs = load_pairs(PAIR_FILE)
    problem = build_problem(pairs)

    println("SBP8 exact subspace optimization starting...")
    println("  pair_file=", PAIR_FILE)
    println("  n_pairs=", length(pairs), " fr7=", FR7)

    xp = Mod._solve_exact_linear_system(problem.A, problem.b)
    Nbasis, pivot_cols, free_cols = rational_nullspace_basis(problem.A)
    nfree = size(Nbasis, 2)

    null_ok = true
    for j in 1:nfree
        r = problem.A * Nbasis[:, j]
        if any(rr -> rr != ZERO_Q, r)
            null_ok = false
            break
        end
    end

    enforce_exact_residual!(problem.A, problem.b, xp, "x_p (particular solution)")

    alpha_start_q = nothing
    alpha_start = zeros(Float64, nfree)
    alpha_start_source = "zeros"
    proj_fit_err = NaN

    if isfile(ALPHA0_FILE)
        alpha_start_q = load_alpha_vector(ALPHA0_FILE, nfree)
        alpha_start = Float64.(alpha_start_q)
        alpha_start_source = "alpha_file"
    elseif REQUIRE_ALPHA0_FILE
        error("SBP8_REQUIRE_ALPHA0_FILE=true and file missing: $ALPHA0_FILE")
    elseif USE_TARGET_PROJECTION && isfile(TARGET_X_FILE)
        x_target = load_x_vector(TARGET_X_FILE)
        alpha_start, proj_fit_err = project_target_alpha_float(xp, Nbasis, x_target)
        alpha_start_source = "projected_target_x"
    elseif USE_TARGET_PROJECTION
        error("SBP8_USE_TARGET_PROJECTION=true but target x file missing: $TARGET_X_FILE")
    else
        alpha_start_source = "zeros"
    end

    Nf = Matrix{Float64}(Nbasis)
    xpf = Float64.(xp)
    start_float_res = evaluate_x_float(problem, xpf .+ Nf * alpha_start)
    if !isfinite(start_float_res.obj) &&
       alpha_start_source == "alpha_file" &&
       USE_TARGET_PROJECTION &&
       isfile(TARGET_X_FILE)
        x_target = load_x_vector(TARGET_X_FILE)
        alpha_start, proj_fit_err = project_target_alpha_float(xp, Nbasis, x_target)
        alpha_start_source = "projected_target_x_fallback"
        start_float_res = evaluate_x_float(problem, xpf .+ Nf * alpha_start)
    end
    isfinite(start_float_res.obj) || error(
        "Invalid optimization start (obj=Inf). status=$(start_float_res.status), source=$alpha_start_source",
    )

    opt = run_subspace_opt(problem, xp, Nbasis, alpha_start, trace_path)

    alpha_best_float = opt.best_alpha
    best_float_res = opt.best_res

    rat_candidates = NamedTuple[]
    selected = nothing

    open(rat_scan_path, "w") do io
        println(io, "tol\tmax_real\tmax_imag\tv_full_pd\tstrict_hard\tstatus")
        for tol in RATIONAL_TOLS
            alpha_rat = Rational{BigInt}[rationalize_big(a, tol) for a in alpha_best_float]
            x_rat = compose_x(xp, Nbasis, alpha_rat)
            enforce_exact_residual!(problem.A, problem.b, x_rat, "x_final(tol=$tol)")
            res_rat = evaluate_x_rational(problem, x_rat)
            strict = res_rat.v_full_pd && res_rat.max_imag == 0.0 && res_rat.max_real < 0.0
            println(io, tol, '\t', res_rat.max_real, '\t', res_rat.max_imag, '\t', res_rat.v_full_pd, '\t', strict, '\t', res_rat.status)
            push!(rat_candidates, (
                tol=tol,
                alpha=alpha_rat,
                x=x_rat,
                res=res_rat,
                strict=strict,
            ))
            if strict && isnothing(selected)
                selected = rat_candidates[end]
            end
        end
    end

    if isnothing(selected)
        feasible = filter(c -> c.res.v_full_pd && c.res.max_imag == 0.0, rat_candidates)
        if !isempty(feasible)
            selected = reduce((a, b) -> a.res.max_real < b.res.max_real ? a : b, feasible)
        elseif !isempty(rat_candidates)
            selected = reduce((a, b) -> a.res.max_real < b.res.max_real ? a : b, rat_candidates)
        else
            error("No rational candidates evaluated.")
        end
    end

    final_alpha_q = selected.alpha
    final_x = selected.x
    final_res = selected.res
    final_exact = enforce_exact_residual!(problem.A, problem.b, final_x, "final selected rational solution")

    S_mat, V_mat = build_matrices(problem, final_x)
    write_pairs(pairs_out, pairs)
    write_alpha_f(alpha_float_out, alpha_best_float)
    write_alpha_q(alpha_rat_out, final_alpha_q)
    write_matrix(s_out, "S", S_mat)
    write_matrix(v_out, "V", V_mat)

    open(summary_path, "w") do io
        println(io, "SBP8 multidimensional null-space optimization (float -> rational)")
        println(io, "fr7 = ", FR7)
        println(io, "block_stop = ", BLOCK_STOP)
        println(io, "pair_file = ", PAIR_FILE)
        println(io, "alpha0_file = ", ALPHA0_FILE)
        println(io, "alpha0_source = ", alpha_start_source)
        println(io, "target_x_file = ", TARGET_X_FILE)
        println(io, "use_target_projection = ", USE_TARGET_PROJECTION)
        println(io, "alpha_proj_tol = ", ALPHA_PROJ_TOL)
        println(io, "proj_fit_err = ", proj_fit_err)
        println(io, "n_pairs = ", length(pairs))
        println(io, "opt_method = ", OPT_METHOD)
        println(io, "opt_max_iters = ", OPT_MAX_ITERS)
        println(io, "opt_penalty_imag = ", OPT_PENALTY_IMAG)
        println(io, "target_max_real = ", TARGET_MAX_REAL)
        println(io, "float_imag_tol = ", FLOAT_IMAG_TOL)
        println(io, "rational_tols = ", RATIONAL_TOLS)
        println(io)
        println(io, "linear system:")
        println(io, "  equations = ", size(problem.A, 1))
        println(io, "  unknowns = ", size(problem.A, 2))
        println(io, "  pivot_cols = ", length(pivot_cols))
        println(io, "  free_cols = ", nfree)
        println(io, "  nullspace_exact = ", null_ok)
        println(io)
        println(io, "optimizer:")
        println(io, "  start_float_obj = ", start_float_res.obj)
        println(io, "  start_float_max_real = ", start_float_res.max_real)
        println(io, "  start_float_max_imag = ", start_float_res.max_imag)
        println(io, "  start_float_v_full_pd = ", start_float_res.v_full_pd)
        println(io, "  converged = ", Optim.converged(opt.result))
        println(io, "  iterations = ", Optim.iterations(opt.result))
        println(io, "  f_calls = ", opt.eval_count)
        println(io, "  minimum = ", Optim.minimum(opt.result))
        println(io, "  best_float_obj = ", best_float_res.obj)
        println(io, "  best_float_max_real = ", best_float_res.max_real)
        println(io, "  best_float_max_imag = ", best_float_res.max_imag)
        println(io, "  best_float_v_full_pd = ", best_float_res.v_full_pd)
        println(io)
        println(io, "selected rational candidate:")
        println(io, "  tol = ", selected.tol)
        println(io, "  strict_hard = ", selected.strict)
        println(io, "  max_real = ", final_res.max_real)
        println(io, "  max_imag = ", final_res.max_imag)
        println(io, "  rho = ", final_res.rho)
        println(io, "  cond_D = ", final_res.cond_D)
        println(io, "  cond_G = ", final_res.cond_G)
        println(io, "  cond_L = ", final_res.cond_L)
        println(io, "  min_s = ", final_res.min_s, ", s11 = ", final_res.s11)
        println(io, "  v_full_pd = ", final_res.v_full_pd)
        println(io, "  exact_residual_zero = ", final_exact.all_zero, " (max |numerator| = ", final_exact.max_num, ")")
        println(io)
        println(io, "success_criteria:")
        println(io, "  max_imag==0: ", final_res.max_imag == IMAG_TOL)
        println(io, "  max_real<0: ", final_res.max_real < REAL_NEG_TOL)
        println(io, "  v_full_pd: ", final_res.v_full_pd)
        println(io)
        println(io, "ansatz_pairs = ", pairs)
    end

    println("Optimization finished.")
    println("  best_float(max_real,max_imag)=(", best_float_res.max_real, ",", best_float_res.max_imag, ")")
    println("  selected_rational(max_real,max_imag)=(", final_res.max_real, ",", final_res.max_imag, ")")
    println("  strict_hard=", selected.strict)
    println("Wrote: ", trace_path)
    println("Wrote: ", rat_scan_path)
    println("Wrote: ", summary_path)
    println("Wrote: ", pairs_out)
    println("Wrote: ", alpha_float_out)
    println("Wrote: ", alpha_rat_out)
    println("Wrote: ", s_out)
    println("Wrote: ", v_out)
end

main()
