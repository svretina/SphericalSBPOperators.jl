using LinearAlgebra
using SparseArrays
using SummationByPartsOperators
using SphericalSBPOperators

const Mod = SphericalSBPOperators
rb(x) = Mod._sbp8_as_big_rational(x)

const BR = 19
const BW = 7
const FR7 = 11
const DIAG_FREE_END = 24
const TAIL_START = 24

const BASE_ZERO_PAIRS = [(1, 2), (1, 3), (1, 4), (13, 16), (14, 16), (15, 16)]
const EXTRA_ZERO_PAIRS = [(14, 17), (15, 17), (16, 17)]

const S1_ANCHOR = rb(11) // rb(20)

const V11_FIX = parse(BigInt, get(ENV, "SBP8_V11_NUM", "3")) // parse(BigInt, get(ENV, "SBP8_V11_DEN", "25"))
const A51_CENTER = parse(BigInt, get(ENV, "SBP8_A51_CENTER_NUM", "3")) // parse(BigInt, get(ENV, "SBP8_A51_CENTER_DEN", "1000000"))
const A51_STEP = parse(BigInt, get(ENV, "SBP8_A51_STEP_NUM", "1")) // parse(BigInt, get(ENV, "SBP8_A51_STEP_DEN", "1000000000"))
const SPAN = parse(Int, get(ENV, "SBP8_A51_SPAN", "200"))

const IMAG_TARGET = 0.0
const REAL_TARGET = 0.0

const RESULTS_DIR = get(ENV, "SBP8_RESULTS_DIR", "data/sbp8_server_search_2026-03-06")
const OUT_BASENAME = get(ENV, "SBP8_OUT_BASENAME", "fr711_base111_a51_refine")

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
    closure_right = Mod._sbp8_infer_right_boundary_closure(setup, sparse(setup.Geven))
    rows = Mod._sbp8_constraint_rows(N, closure_right, FR7)
    diag_free_indices = collect(1:DIAG_FREE_END)
    pairs = Mod.sbp8_v_offdiag_pairs(N; boundary_rows=BR, boundary_bandwidth=BW)
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

    for i in TAIL_START:N
        A, b = add_fix_col!(A, b, i, sq[i])
    end

    A, b = add_fix_col!(A, b, 1, S1_ANCHOR)

    col_vdiag_tail = get(idx.vdiag_to_col, TAIL_START, 0)
    if col_vdiag_tail > 0
        A, b = add_fix_col!(A, b, col_vdiag_tail, sq[TAIL_START])
    end

    for pr in vcat(BASE_ZERO_PAIRS, EXTRA_ZERO_PAIRS)
        k = get(pair_to_k, pr, 0)
        k == 0 && continue
        A, b = add_fix_col!(A, b, idx.idx_voff(k), rb(0))
    end

    col_v11 = get(idx.vdiag_to_col, 1, 0)
    col_v11 > 0 || error("Missing vdiag[1] free column.")
    A, b = add_fix_col!(A, b, col_v11, V11_FIX)

    k_a51 = get(pair_to_k, (18, 19), 0)
    k_a51 > 0 || error("Missing pair (18,19) in V structure.")
    col_a51 = idx.idx_voff(k_a51)

    return (
        setup=setup,
        sys=sys,
        rows=rows,
        N=N,
        idx=idx,
        pairs=pairs,
        A=A,
        b=b,
        col_a51=col_a51,
    )
end

function eval_a51(problem, a51::Rational{BigInt})
    A2, b2 = add_fix_col!(problem.A, problem.b, problem.col_a51, a51)
    x = Mod._solve_exact_linear_system(A2, b2)

    idx = problem.idx
    N = problem.N
    sdiag_q = [x[i] for i in 1:idx.n_s]
    min_s = minimum(Float64.(sdiag_q))
    s11 = Float64(sdiag_q[1])

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
    L = Matrix{Float64}(sparse(D * problem.sys.G))
    ev = eigen(L).values
    max_real = maximum(real.(ev))
    max_imag = maximum(abs.(imag.(ev)))
    rho = maximum(abs.(ev))

    v_full_pd = Mod._sbp8_is_pd(V)

    strict = min_s > 0.0 &&
             s11 > 0.0 &&
             v_full_pd &&
             max_imag == IMAG_TARGET &&
             max_real < REAL_TARGET

    return (
        status="ok",
        strict_hard=strict,
        rho=rho,
        max_real=max_real,
        max_imag=max_imag,
        min_s=min_s,
        s11=s11,
        v_full_pd=v_full_pd,
    )
end

function score(res)
    s_pen = res.min_s > 0.0 && res.s11 > 0.0 ? 0 : 1
    v_pen = res.v_full_pd ? 0 : 1
    imag_gap = abs(res.max_imag - IMAG_TARGET)
    real_gap = max(res.max_real, REAL_TARGET)
    return (s_pen, v_pen, imag_gap, real_gap, res.max_real, res.rho)
end

function ensure_dir(path::String)
    isdir(path) || mkpath(path)
end

function main()
    ensure_dir(RESULTS_DIR)
    out_tsv = joinpath(RESULTS_DIR, "$(OUT_BASENAME).tsv")
    out_summary = joinpath(RESULTS_DIR, "$(OUT_BASENAME)_summary.txt")

    problem = build_problem()

    best = nothing
    strict_count = 0
    rows_written = 0

    open(out_tsv, "w") do io
        println(io, join([
            "idx", "delta", "a51_float", "a51_rational", "strict_hard",
            "rho", "max_real", "max_imag", "min_s", "s11", "V_full_pd",
        ], '\t'))

        for i in -SPAN:SPAN
            a51 = A51_CENTER + i * A51_STEP
            res = eval_a51(problem, a51)
            println(io, join([
                string(i),
                string(Float64(i * A51_STEP)),
                string(Float64(a51)),
                string(a51),
                string(res.strict_hard),
                string(res.rho),
                string(res.max_real),
                string(res.max_imag),
                string(res.min_s),
                string(res.s11),
                string(res.v_full_pd),
            ], '\t'))
            flush(io)
            rows_written += 1
            strict_count += res.strict_hard ? 1 : 0

            candidate = (res=res, a51=a51)
            if isnothing(best) || score(candidate.res) < score(best.res)
                best = candidate
            end
        end
    end

    open(out_summary, "w") do io
        println(io, "SBP8 fr711 base111 a51 refinement")
        println(io, "fr7 = ", FR7, ", br = ", BR, ", bw = ", BW)
        println(io, "fixed v11 = ", V11_FIX)
        println(io, "a51 center = ", A51_CENTER, ", step = ", A51_STEP, ", span = ", SPAN)
        println(io, "hard checks: max_imag == 0, max_real < 0, S diag > 0, V full PD")
        println(io, "rows = ", rows_written, ", strict_count = ", strict_count)
        println(io)
        println(io, "best:")
        println(io, "  a51 = ", best.a51, " (", Float64(best.a51), ")")
        println(io, "  rho = ", best.res.rho)
        println(io, "  max_real = ", best.res.max_real)
        println(io, "  max_imag = ", best.res.max_imag)
        println(io, "  strict_hard = ", best.res.strict_hard)
        println(io, "  min_s = ", best.res.min_s, ", s11 = ", best.res.s11, ", V_full_pd = ", best.res.v_full_pd)
    end

    println("Wrote: ", out_tsv)
    println("Wrote: ", out_summary)
end

main()

