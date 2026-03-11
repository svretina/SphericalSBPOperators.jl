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

const FR7 = parse(Int, get(ENV, "SBP8_FR7", "11"))
const BAND_STOP = parse(Int, get(ENV, "SBP8_BAND_STOP", "19"))
const DIAG_FREE_END = parse(Int, get(ENV, "SBP8_DIAG_FREE_END", "24"))
const TAIL_START = parse(Int, get(ENV, "SBP8_TAIL_START", "24"))

const FIX_S1 = parse_bool("SBP8_FIX_S1", "true")
const S1_ANCHOR = parse_q(get(ENV, "SBP8_S1_ANCHOR", "11/20"))
const V11_FIX = parse_q(get(ENV, "SBP8_V11_FIX", "1/1"))

const STOP_ON_STRICT = parse_bool("SBP8_STOP_ON_STRICT", "false")
const OUT_BASENAME = get(ENV, "SBP8_OUT_BASENAME", "sbp8_search")
const RESULTS_DIR = get(ENV, "SBP8_RESULTS_DIR", "data/sbp8_server_search_2026-03-06")

const IMAG_TOL = 0.0
const REAL_NEG_TOL = 0.0

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

function build_pair_order(band_stop::Int)
    order = Tuple{Int,Int}[]
    for d in 1:(band_stop - 2)
        for i in 2:(band_stop - d)
            j = i + d
            push!(order, (i, j))
        end
    end
    return order
end

pair_string(pr::Tuple{Int,Int}) = "v[$(pr[1]),$(pr[2])]"

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

function ensure_dir(path::String)
    isdir(path) || mkpath(path)
end

function build_problem()
    setup = Mod.sbp8_scalar_mass_gradient(
        Mattsson2017(:central);
        accuracy_order=8,
        points=31,
        h=1,
        p=2,
        build_matrix=:probe,
    )

    N = length(setup.r)
    3 <= BAND_STOP <= N || error("SBP8_BAND_STOP must satisfy 3 <= BAND_STOP <= $N.")
    1 <= DIAG_FREE_END <= N || error("SBP8_DIAG_FREE_END must satisfy 1 <= DIAG_FREE_END <= $N.")
    1 <= TAIL_START <= N || error("SBP8_TAIL_START must satisfy 1 <= TAIL_START <= $N.")

    closure_right = Mod._sbp8_infer_right_boundary_closure(setup, sparse(setup.Geven))
    rows = Mod._sbp8_constraint_rows(N, closure_right, FR7)

    diag_free_indices = collect(1:DIAG_FREE_END)
    pairs = build_pair_order(BAND_STOP)
    pair_to_k = Dict{Tuple{Int,Int},Int}(pr => k for (k, pr) in enumerate(pairs))

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

    # Keep outer boundary scaling in S.
    for i in TAIL_START:N
        A, b = add_fix_col!(A, b, i, sq[i])
    end

    # Keep outer boundary scaling in V diagonal at the tail start index.
    col_vdiag_tail = get(idx.vdiag_to_col, TAIL_START, 0)
    if col_vdiag_tail > 0
        A, b = add_fix_col!(A, b, col_vdiag_tail, sq[TAIL_START])
    end

    # Keep s11 positive via anchor if requested.
    if FIX_S1
        A, b = add_fix_col!(A, b, 1, S1_ANCHOR)
    end

    # v11 is fixed to user-provided arbitrary value.
    col_v11 = get(idx.vdiag_to_col, 1, 0)
    col_v11 > 0 || error("vdiag[1] is not free with current DIAG_FREE_END=$DIAG_FREE_END.")
    A, b = add_fix_col!(A, b, col_v11, V11_FIX)

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
        cond_G=safe_cond(Gf),
    )
end

function build_step_system(problem, active_count::Int)
    A = problem.baseA
    b = problem.baseb
    idx = problem.idx
    npairs = length(problem.pairs)

    @assert 0 <= active_count <= npairs
    for k in (active_count + 1):npairs
        A, b = add_fix_col!(A, b, idx.idx_voff(k), big(0) // big(1))
    end
    return A, b
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
        if v != 0 // 1
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

    v_boundary_pd = try
        cholesky(Symmetric(Matrix{Float64}(V[1:BAND_STOP, 1:BAND_STOP])); check=true)
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
    )
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

function write_header(io)
    println(
        io,
        join(
            [
                "step",
                "new_pair",
                "active_pairs",
                "n_pairs_total",
                "status",
                "strict_hard",
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
                "reason",
            ],
            '\t',
        ),
    )
end

function write_row(io, step::Int, new_pair::Union{Nothing,Tuple{Int,Int}}, active_pairs::Int, n_pairs_total::Int, res, eq_count::Int, unknown_count::Int)
    new_pair_str = isnothing(new_pair) ? "none" : pair_string(new_pair)
    println(
        io,
        join(
            [
                string(step),
                new_pair_str,
                string(active_pairs),
                string(n_pairs_total),
                res.status,
                string(res.strict_hard),
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
                res.reason,
            ],
            '\t',
        ),
    )
end

function write_summary(path::String, problem, strict_count::Int, total_steps::Int, best, strict_hits::Vector{Int})
    open(path, "w") do io
        println(io, "SBP8 incremental V-structure search")
        println(io, "algorithm: diagonal V start, v11 fixed, row/col-1 couplings excluded, add symmetric block pairs by superdiagonal bands")
        println(io)
        println(io, "fr7 = ", FR7)
        println(io, "band_stop = ", BAND_STOP)
        println(io, "diag_free_indices = 1:", DIAG_FREE_END)
        println(io, "tail_fixed_start = ", TAIL_START)
        println(io, "fix_s1 = ", FIX_S1, FIX_S1 ? " (s11 anchor = $(S1_ANCHOR))" : "")
        println(io, "v11 fixed = ", V11_FIX)
        println(io, "hard spectral checks: max_imag == ", IMAG_TOL, ", max_real < ", REAL_NEG_TOL)
        println(io, "hard constraints: exact solve, S diag > 0, V full PD")
        println(io, "stop_on_strict = ", STOP_ON_STRICT)
        println(io)
        println(io, "total_pairs = ", length(problem.pairs))
        println(io, "total_steps = ", total_steps)
        println(io, "strict_count = ", strict_count)
        println(io, "strict_hit_steps = ", isempty(strict_hits) ? "none" : join(string.(strict_hits), ","))
        println(io)
        println(io, "best row:")
        println(io, "  step = ", best.step, ", new_pair = ", best.new_pair)
        println(io, "  status = ", best.res.status, ", strict_hard = ", best.res.strict_hard)
        println(io, "  rho = ", best.res.rho)
        println(io, "  max_real = ", best.res.max_real)
        println(io, "  max_imag = ", best.res.max_imag)
        println(io, "  cond_D = ", best.res.cond_D)
        println(io, "  cond_G = ", best.res.cond_G)
        println(io, "  cond_L = ", best.res.cond_L)
        println(io, "  min_s = ", best.res.min_s, ", s11 = ", best.res.s11)
        println(io, "  v_boundary_pd = ", best.res.v_boundary_pd, ", v_full_pd = ", best.res.v_full_pd)
    end
end

function main()
    ensure_dir(RESULTS_DIR)
    out_tsv = joinpath(RESULTS_DIR, "$(OUT_BASENAME).tsv")
    out_summary = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_summary.txt")

    problem = build_problem()
    npairs = length(problem.pairs)
    strict_count = 0
    strict_hits = Int[]
    best = nothing

    println("SBP8 search starting...")
    println("  fr7=", FR7, " band_stop=", BAND_STOP, " npairs=", npairs)
    println("  output=", out_tsv)

    open(out_tsv, "w") do io
        write_header(io)

        for step in 0:npairs
            new_pair = step == 0 ? nothing : problem.pairs[step]
            A, b = build_step_system(problem, step)
            eq_count = size(A, 1)
            unknown_count = size(A, 2)

            res = try
                x = Mod._solve_exact_linear_system(A, b)
                evaluate_solution(problem, x)
            catch err
                (
                    status="solve_fail",
                    reason=sprint(showerror, err),
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

            write_row(io, step, new_pair, step, npairs, res, eq_count, unknown_count)
            flush(io)

            if res.strict_hard
                strict_count += 1
                push!(strict_hits, step)
            end

            candidate = (step=step, new_pair=isnothing(new_pair) ? "none" : pair_string(new_pair), res=res)
            if isnothing(best) || result_score(candidate.res) < result_score(best.res)
                best = candidate
            end

            println(
                "step=", lpad(step, 3),
                " active_pairs=", lpad(step, 3), "/", npairs,
                " status=", res.status,
                " strict=", res.strict_hard,
                " max_real=", res.max_real,
                " max_imag=", res.max_imag,
            )

            if STOP_ON_STRICT && res.strict_hard
                println("Stopping early on strict hit at step ", step)
                break
            end
        end
    end

    total_steps = (STOP_ON_STRICT && !isempty(strict_hits)) ? (strict_hits[1] + 1) : (npairs + 1)
    write_summary(out_summary, problem, strict_count, total_steps, best, strict_hits)

    println("Wrote: ", out_tsv)
    println("Wrote: ", out_summary)
end

main()
