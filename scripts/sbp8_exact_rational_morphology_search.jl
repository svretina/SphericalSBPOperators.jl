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

function parse_q_list(envkey::String, default_csv::String)
    raw = get(ENV, envkey, default_csv)
    vals = Rational{BigInt}[]
    for tok in split(raw, ',')
        t = strip(tok)
        isempty(t) && continue
        push!(vals, parse_q(t))
    end
    isempty(vals) && error("No valid rational values parsed from $envkey='$raw'.")
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

const STEP_SET = parse_q_list("SBP8_ALPHA_STEPS", "1/100,1/10,1/1")
const MAX_PASSES = parse(Int, get(ENV, "SBP8_MAX_PASSES", "40"))
const STOP_ON_NEGATIVE = parse_bool("SBP8_STOP_ON_NEGATIVE", "true")

const RESULTS_DIR = get(
    ENV,
    "SBP8_RESULTS_DIR",
    "data/sbp8_server_search_2026-03-08",
)
const OUT_BASENAME = get(ENV, "SBP8_OUT_BASENAME", "sbp8_exact_morphology")

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
    )
end

function evaluate_x(problem, x::Vector{Rational{BigInt}})
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
            v_boundary_pd=false,
            v_full_pd=false,
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
        v_boundary_pd=v_boundary_pd,
        v_full_pd=v_full_pd,
    )
end

function satisfies_acceptance(res)
    return res.status == "ok" &&
           res.v_full_pd &&
           res.max_imag == IMAG_TOL
end

function better_candidate(res_a, res_b)
    # assumes both satisfy acceptance
    key_a = (res_a.max_real, res_a.rho, res_a.cond_L, res_a.cond_D)
    key_b = (res_b.max_real, res_b.rho, res_b.cond_L, res_b.cond_D)
    return key_a < key_b
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

function build_matrices(problem, x::Vector{Rational{BigInt}})
    idx = problem.idx
    N = problem.N
    sdiag_q = [x[i] for i in 1:idx.n_s]

    vdiag_q = Vector{Rational{BigInt}}(undef, N)
    for i in 1:N
        c = get(idx.vdiag_to_col, i, 0)
        vdiag_q[i] = c > 0 ? x[c] : sdiag_q[i]
    end
    voff_q = [x[idx.idx_voff(k)] for k in 1:idx.n_voff]

    Sm = fill(ZERO_Q, N, N)
    Vm = fill(ZERO_Q, N, N)
    for i in 1:N
        Sm[i, i] = sdiag_q[i]
        Vm[i, i] = vdiag_q[i]
    end
    for (k, (i, j)) in enumerate(problem.pairs)
        v = voff_q[k]
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

function write_alpha(path::String, alpha::Vector{Rational{BigInt}})
    open(path, "w") do io
        println(io, "basis_idx\talpha")
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

function run_search(problem, xp::Vector{Rational{BigInt}}, Nbasis::Matrix{Rational{BigInt}}, free_cols::Vector{Int}, trace_path::String)
    nfree = size(Nbasis, 2)
    alpha = zeros(Rational{BigInt}, nfree)

    current_x = copy(xp)
    current_res = evaluate_x(problem, current_x)
    best_res = current_res
    best_x = current_x
    best_alpha = copy(alpha)

    open(trace_path, "w") do io
        println(io, "pass\tbasis_idx\tbasis_col\tbasis_label\tstep\tsign\talpha_old\talpha_new\taccepted\tstatus\tmax_real\tmax_imag\tv_full_pd\trho\treason")
        accepted_any = false
        for pass in 1:MAX_PASSES
            improved = false
            for j in 1:nfree
                base_alpha = alpha[j]
                candidate_found = false
                candidate_alpha = base_alpha
                candidate_res = nothing
                candidate_x = nothing

                base_threshold = satisfies_acceptance(current_res) ? current_res.max_real : Inf

                for q in STEP_SET
                    for sgn in (-1, 1)
                        trial_alpha = copy(alpha)
                        trial_alpha[j] = base_alpha + sgn * q
                        trial_x = compose_x(xp, Nbasis, trial_alpha)
                        trial_res = evaluate_x(problem, trial_x)
                        accept = false

                        if satisfies_acceptance(trial_res) && trial_res.max_real < base_threshold
                            if !candidate_found || better_candidate(trial_res, candidate_res)
                                candidate_found = true
                                candidate_alpha = trial_alpha[j]
                                candidate_res = trial_res
                                candidate_x = trial_x
                                accept = true
                            end
                        end

                        println(
                            io,
                            join(
                                [
                                    string(pass),
                                    string(j),
                                    string(free_cols[j]),
                                    col_label(problem, free_cols[j]),
                                    string(q),
                                    string(sgn),
                                    string(base_alpha),
                                    string(trial_alpha[j]),
                                    string(accept),
                                    trial_res.status,
                                    string(trial_res.max_real),
                                    string(trial_res.max_imag),
                                    string(trial_res.v_full_pd),
                                    string(trial_res.rho),
                                    trial_res.reason,
                                ],
                                '\t',
                            ),
                        )
                        flush(io)
                    end
                end

                if candidate_found
                    alpha[j] = candidate_alpha
                    current_x = candidate_x
                    current_res = candidate_res
                    improved = true
                    accepted_any = true
                    println(
                        "pass=", pass,
                        " basis=", j, "/", nfree,
                        " accepted=true",
                        " max_real=", current_res.max_real,
                        " max_imag=", current_res.max_imag,
                        " v_full_pd=", current_res.v_full_pd,
                    )
                    if better_candidate(current_res, best_res) || !satisfies_acceptance(best_res)
                        best_res = current_res
                        best_x = current_x
                        best_alpha = copy(alpha)
                    end
                end
            end

            if STOP_ON_NEGATIVE &&
               satisfies_acceptance(current_res) &&
               current_res.max_real < REAL_NEG_TOL
                println("Stopping early: reached max_real<0 at pass ", pass)
                break
            end

            if !improved
                println("No accepted updates in pass ", pass, "; stopping.")
                break
            end
        end

        if !accepted_any
            println("No accepted perturbation found under hard acceptance criteria.")
        end
    end

    return (
        current_res=current_res,
        current_x=current_x,
        alpha=alpha,
        best_res=best_res,
        best_x=best_x,
        best_alpha=best_alpha,
    )
end

function main()
    ensure_dir(RESULTS_DIR)

    trace_path = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_scan.tsv")
    summary_path = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_summary.txt")
    pairs_out = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_pairs.jl")
    alpha_out = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_alpha.tsv")
    s_out = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_S_matrix.jl")
    v_out = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_V_matrix.jl")

    pairs = load_pairs(PAIR_FILE)
    problem = build_problem(pairs)

    println("Exact rational morphology search starting...")
    println("  pair_file=", PAIR_FILE)
    println("  n_pairs=", length(pairs), " fr7=", FR7)

    xp = Mod._solve_exact_linear_system(problem.A, problem.b)
    Nbasis, pivot_cols, free_cols = rational_nullspace_basis(problem.A)
    nfree = size(Nbasis, 2)

    # Verify nullspace basis exactly.
    null_ok = true
    for j in 1:nfree
        r = problem.A * Nbasis[:, j]
        if any(rr -> rr != ZERO_Q, r)
            null_ok = false
            break
        end
    end

    base_res = evaluate_x(problem, xp)
    base_exact = exact_residual_stats(problem.A, problem.b, xp)

    search = run_search(problem, xp, Nbasis, free_cols, trace_path)
    final_res = search.best_res
    final_x = search.best_x
    final_alpha = search.best_alpha

    final_exact = exact_residual_stats(problem.A, problem.b, final_x)
    S_mat, V_mat = build_matrices(problem, final_x)

    write_pairs(pairs_out, pairs)
    write_alpha(alpha_out, final_alpha)
    write_matrix(s_out, "S", S_mat)
    write_matrix(v_out, "V", V_mat)

    open(summary_path, "w") do io
        println(io, "SBP8 exact rational stability search on extracted morphology")
        println(io, "fr7 = ", FR7)
        println(io, "block_stop = ", BLOCK_STOP)
        println(io, "pair_file = ", PAIR_FILE)
        println(io, "n_pairs = ", length(pairs))
        println(io, "alpha_steps = ", STEP_SET)
        println(io, "max_passes = ", MAX_PASSES)
        println(io)
        println(io, "linear system:")
        println(io, "  equations = ", size(problem.A, 1))
        println(io, "  unknowns = ", size(problem.A, 2))
        println(io, "  pivot_cols = ", length(pivot_cols))
        println(io, "  free_cols = ", nfree)
        println(io, "  nullspace_exact = ", null_ok)
        println(io)
        println(io, "particular solution (alpha=0):")
        println(io, "  status = ", base_res.status)
        println(io, "  max_real = ", base_res.max_real)
        println(io, "  max_imag = ", base_res.max_imag)
        println(io, "  v_full_pd = ", base_res.v_full_pd)
        println(io, "  exact_residual_zero = ", base_exact.all_zero, " (max |numerator| = ", base_exact.max_num, ")")
        println(io)
        println(io, "best accepted solution:")
        println(io, "  status = ", final_res.status)
        println(io, "  strict_hard = ", final_res.strict_hard)
        println(io, "  max_real = ", final_res.max_real)
        println(io, "  max_imag = ", final_res.max_imag)
        println(io, "  rho = ", final_res.rho)
        println(io, "  cond_D = ", final_res.cond_D)
        println(io, "  cond_G = ", final_res.cond_G)
        println(io, "  cond_L = ", final_res.cond_L)
        println(io, "  min_s = ", final_res.min_s, ", s11 = ", final_res.s11)
        println(io, "  v_boundary_pd = ", final_res.v_boundary_pd, ", v_full_pd = ", final_res.v_full_pd)
        println(io, "  exact_residual_zero = ", final_exact.all_zero, " (max |numerator| = ", final_exact.max_num, ")")
        println(io)
        println(io, "success_criteria:")
        println(io, "  max_imag==0: ", final_res.max_imag == IMAG_TOL)
        println(io, "  max_real<0: ", final_res.max_real < REAL_NEG_TOL)
        println(io, "  v_full_pd: ", final_res.v_full_pd)
        println(io)
        println(io, "ansatz_pairs = ", pairs)
    end

    println("Wrote: ", trace_path)
    println("Wrote: ", summary_path)
    println("Wrote: ", pairs_out)
    println("Wrote: ", alpha_out)
    println("Wrote: ", s_out)
    println("Wrote: ", v_out)
end

main()
