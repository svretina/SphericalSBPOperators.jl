using LinearAlgebra
using SparseArrays
using SummationByPartsOperators
using SphericalSBPOperators

const Mod = SphericalSBPOperators
rb(x) = Mod._sbp8_as_big_rational(x)

const BR = 19
const BW = 7
const DIAG_FREE_END = 24
const TAIL_START = 24

const BASE_ZERO_PAIRS = [(1, 2), (1, 3), (1, 4), (13, 16), (14, 16), (15, 16)]
const RUNB_ZERO_PAIRS = [(14, 17), (15, 17), (16, 17)]
const S1_ANCHOR = rb(11) // rb(20)

const IMAG_TOL = 0.0
const REAL_NEG_TOL = 0.0

const PROBE_DEN = parse(BigInt, get(ENV, "SBP8_PROBE_DEN", "1000000"))
const STAGE1_DEN = parse(BigInt, get(ENV, "SBP8_STAGE1_DEN", "100000"))
const STAGE2_DEN = parse(BigInt, get(ENV, "SBP8_STAGE2_DEN", "1000000"))
const PROBE_STEP = big(1) // PROBE_DEN
const STAGE1_STEP = big(1) // STAGE1_DEN
const STAGE2_STEP = big(1) // STAGE2_DEN
const STAGE_SPAN = parse(Int, get(ENV, "SBP8_STAGE_SPAN", "8"))

const RESULTS_DIR = get(
    ENV,
    "SBP8_RESULTS_DIR",
    "data/sbp8_server_search_2026-03-06",
)

struct RunSpec
    id::String
    fr7::Int
    extra_zero_pairs::Vector{Tuple{Int,Int}}
end

function build_specs()
    sets = [
        ("base111", [(14, 17), (15, 17), (16, 17)]),
        ("drop1417", [(15, 17), (16, 17)]),
        ("drop1517", [(14, 17), (16, 17)]),
        ("drop1617", [(14, 17), (15, 17)]),
    ]
    specs = RunSpec[]
    for fr7 in (10, 11, 12)
        for (tag, zset) in sets
            push!(specs, RunSpec("fr7$(fr7)_$(tag)", fr7, zset))
        end
    end
    return specs
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

function rref_pivots(A::Matrix{Rational{BigInt}})
    n_eq, n_vars = size(A)
    n_eq == 0 && return Int[], collect(1:n_vars)

    M = copy(A)
    pivot_cols = Int[]
    pivot_row = 1

    for col in 1:n_vars
        pivot = 0
        for r in pivot_row:n_eq
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

        for r in 1:n_eq
            r == pivot_row && continue
            fac = M[r, col]
            fac == 0 // 1 && continue
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

function safe_cond(M::AbstractMatrix{Float64})
    try
        c = cond(M)
        return isfinite(c) ? c : Inf
    catch
        return Inf
    end
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

function pair_col(problem, pair::Tuple{Int,Int})
    k = get(problem.pair_to_k, pair, 0)
    return k == 0 ? 0 : problem.idx.idx_voff(k)
end

function is_vdiag1_col(problem, col::Int)
    c = get(problem.idx.vdiag_to_col, 1, 0)
    return c > 0 && col == c
end

function probe_step_for_col(problem, col::Int)
    return is_vdiag1_col(problem, col) ? (big(1) // 10) : PROBE_STEP
end

function coarse_step_for_col(problem, col::Int)
    return is_vdiag1_col(problem, col) ? (big(1) // 10) : STAGE1_STEP
end

function fine_step_for_col(problem, col::Int)
    return is_vdiag1_col(problem, col) ? (big(1) // 100) : STAGE2_STEP
end

function default_center_for_col(problem, col::Int)
    return is_vdiag1_col(problem, col) ? (big(1) // 1) : problem.x0[col]
end

function candidate_columns(problem)
    free_set = Set(problem.free_cols)
    cols = Int[]

    pair_priority = [
        (17, 18), (18, 19), (17, 19),
        (16, 18), (16, 19), (15, 18),
        (14, 17), (15, 17), (16, 17),
        (14, 15), (13, 15), (13, 14),
    ]

    for pr in pair_priority
        c = pair_col(problem, pr)
        if c > 0 && (c in free_set) && !(c in cols)
            push!(cols, c)
        end
    end

    diag_priority = [1, 14, 15, 16, 17, 18, 19]
    for i in diag_priority
        c = get(problem.idx.vdiag_to_col, i, 0)
        if c > 0 && (c in free_set) && !(c in cols)
            push!(cols, c)
        end
    end

    s_priority = [1, 14, 15, 16, 17, 18, 19]
    for i in s_priority
        c = i
        if c in free_set && !(c in cols)
            push!(cols, c)
        end
    end

    if length(cols) < 2
        for c in problem.free_cols
            if !(c in cols)
                push!(cols, c)
            end
            length(cols) >= 2 && break
        end
    end

    return cols
end

function build_problem(spec::RunSpec)
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
    rows = Mod._sbp8_constraint_rows(N, closure_right, spec.fr7)

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

    # Keep s[1,1] strictly positive and close to prior architecture.
    A, b = add_fix_col!(A, b, 1, S1_ANCHOR)

    col_vdiag_tail = get(idx.vdiag_to_col, TAIL_START, 0)
    if col_vdiag_tail > 0
        A, b = add_fix_col!(A, b, col_vdiag_tail, sq[TAIL_START])
    end

    for pr in vcat(BASE_ZERO_PAIRS, spec.extra_zero_pairs)
        k = get(pair_to_k, pr, 0)
        k == 0 && continue
        A, b = add_fix_col!(A, b, idx.idx_voff(k), rb(0))
    end

    pivot_cols, free_cols = rref_pivots(A)
    x0 = Mod._solve_exact_linear_system(A, b)
    Gf = Matrix{Float64}(sparse(sys.G))

    return (
        spec=spec,
        setup=setup,
        sys=sys,
        rows=rows,
        N=N,
        idx=idx,
        pairs=pairs,
        pair_to_k=pair_to_k,
        diag_free_indices=diag_free_indices,
        A=A,
        b=b,
        x0=x0,
        pivot_cols=pivot_cols,
        free_cols=free_cols,
        Gf=Gf,
        cond_G=safe_cond(Gf),
    )
end

function evaluate_case(problem, fixed_cols::Vector{Int}, fixed_vals::Vector{Rational{BigInt}})
    try
        A2 = problem.A
        b2 = problem.b
        for (col, val) in zip(fixed_cols, fixed_vals)
            A2, b2 = add_fix_col!(A2, b2, col, val)
        end

        x = Mod._solve_exact_linear_system(A2, b2)
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
            cholesky(Symmetric(Matrix{Float64}(V[1:BR, 1:BR])); check=true)
            true
        catch
            false
        end
        v_full_pd = Mod._sbp8_is_pd(V)

        strict_hard = min_s > 0.0 &&
                      s11 > 0.0 &&
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
    catch err
        return (
            status="fail",
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
            x=nothing,
        )
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

function write_header(io)
    println(
        io,
        join(
            [
                "spec_id",
                "fr7",
                "stage",
                "i",
                "j",
                "colA",
                "colB",
                "valueA_float",
                "valueB_float",
                "valueA_rational",
                "valueB_rational",
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
                "Vb_pd",
                "V_full_pd",
                "reason",
            ],
            '\t',
        ),
    )
end

function write_row(io, problem, stage, i, j, colA, colB, valA, valB, res)
    println(
        io,
        join(
            [
                problem.spec.id,
                string(problem.spec.fr7),
                stage,
                string(i),
                string(j),
                colA > 0 ? col_label(problem, colA) : "",
                colB > 0 ? col_label(problem, colB) : "",
                string(Float64(valA)),
                string(Float64(valB)),
                string(valA),
                string(valB),
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
                res.reason,
            ],
            '\t',
        ),
    )
end

function run_stage!(
    io,
    problem,
    colA::Int,
    colB::Int;
    stage::String,
    centerA::Rational{BigInt},
    centerB::Rational{BigInt},
    stepA::Rational{BigInt},
    stepB::Rational{BigInt},
    span::Int,
)
    best = nothing
    rows_written = 0
    strict_count = 0
    ok_count = 0

    js = colB > 0 ? (-span:span) : (0:0)
    for i in -span:span
        for j in js
            valA = centerA + i * stepA
            valB = centerB + j * stepB
            fixed_cols = colB > 0 ? [colA, colB] : [colA]
            fixed_vals = colB > 0 ? [valA, valB] : [valA]

            res = evaluate_case(problem, fixed_cols, fixed_vals)
            write_row(io, problem, stage, i, j, colA, colB, valA, valB, res)
            flush(io)

            rows_written += 1
            if res.status == "ok"
                ok_count += 1
                if res.strict_hard
                    strict_count += 1
                end
            end

            candidate = (res=res, valA=valA, valB=valB)
            if isnothing(best) || result_score(candidate.res) < result_score(best.res)
                best = candidate
            end
        end
    end

    return (
        best=best,
        rows_written=rows_written,
        ok_count=ok_count,
        strict_count=strict_count,
    )
end

function run_probe(problem, probe_cols::Vector{Int})
    rows = NamedTuple[]
    for col in probe_cols
        step = probe_step_for_col(problem, col)
        for sgn in (-1, 1)
            val = sgn * step
            res = evaluate_case(problem, [col], [val])
            push!(rows, (col=col, val=val, res=res, score=result_score(res)))
        end
    end
    sort!(rows, by=r -> r.score)
    return rows
end

function ensure_results_dir()
    isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)
end

function write_summary(path, problem, baseline, probes, selected_cols, stage1, stage2)
    open(path, "w") do io
        println(io, "SBP8 run-B relaxation search summary")
        println(io, "spec_id = ", problem.spec.id)
        println(io, "fr7 = ", problem.spec.fr7)
        println(io, "br = ", BR, ", bw = ", BW)
        println(io, "diag_free_indices = 1:", DIAG_FREE_END)
        println(io, "tail_fixed_start = ", TAIL_START)
        println(io, "base_zero_pairs = ", BASE_ZERO_PAIRS)
        println(io, "extra_zero_pairs = ", problem.spec.extra_zero_pairs)
        println(io, "hard spectral checks: max_imag == ", IMAG_TOL, ", max_real < ", REAL_NEG_TOL)
        println(io, "hard constraints: exact solve, S SPD (diag>0), V SPD (full PD), s11>0")
        println(io, "s11 anchor = ", S1_ANCHOR)
        println(io, "v11 is unconstrained in this run")
        println(io)
        println(io, "system size: equations=", size(problem.A, 1), ", unknowns=", size(problem.A, 2))
        println(io, "pivot_cols = ", length(problem.pivot_cols), ", free_cols = ", length(problem.free_cols))
        println(io, "cond(G) = ", problem.cond_G)
        println(io)
        println(io, "baseline:")
        println(io, "  status = ", baseline.status)
        println(io, "  strict_hard = ", baseline.strict_hard)
        println(io, "  rho = ", baseline.rho)
        println(io, "  max_real = ", baseline.max_real)
        println(io, "  max_imag = ", baseline.max_imag)
        println(io, "  cond_D = ", baseline.cond_D)
        println(io, "  cond_L = ", baseline.cond_L)
        println(io, "  min_s = ", baseline.min_s, ", s11 = ", baseline.s11)
        println(io, "  V_boundary_PD = ", baseline.v_boundary_pd, ", V_full_PD = ", baseline.v_full_pd)
        println(io)
        println(io, "selected columns:")
        for c in selected_cols
            println(io, "  - ", col_label(problem, c), " (col=", c, ")")
        end
        println(io)
        println(io, "top probe rows:")
        for r in Iterators.take(probes, min(length(probes), 8))
            println(io, "  ", col_label(problem, r.col), " @ ", r.val,
                ": max_real=", r.res.max_real, ", max_imag=", r.res.max_imag,
                ", strict_hard=", r.res.strict_hard, ", status=", r.res.status)
        end
        println(io)
        println(io, "stage1:")
        println(io, "  rows = ", stage1.rows_written, ", ok = ", stage1.ok_count, ", strict = ", stage1.strict_count)
        println(io, "  best: rho=", stage1.best.res.rho, ", max_real=", stage1.best.res.max_real,
            ", max_imag=", stage1.best.res.max_imag, ", strict_hard=", stage1.best.res.strict_hard)
        println(io, "  best values: A=", stage1.best.valA, ", B=", stage1.best.valB)
        println(io)
        println(io, "stage2:")
        println(io, "  rows = ", stage2.rows_written, ", ok = ", stage2.ok_count, ", strict = ", stage2.strict_count)
        println(io, "  best: rho=", stage2.best.res.rho, ", max_real=", stage2.best.res.max_real,
            ", max_imag=", stage2.best.res.max_imag, ", strict_hard=", stage2.best.res.strict_hard)
        println(io, "  best values: A=", stage2.best.valA, ", B=", stage2.best.valB)
    end
end

function run_spec(spec::RunSpec)
    println("==== Running spec: ", spec.id, " ====")
    ensure_results_dir()

    problem = build_problem(spec)
    probe_cols = candidate_columns(problem)
    probes = run_probe(problem, probe_cols)

    baseline = evaluate_case(problem, Int[], Rational{BigInt}[])
    selected_cols = Int[]
    for row in probes
        if !(row.col in selected_cols)
            push!(selected_cols, row.col)
        end
        length(selected_cols) >= 2 && break
    end
    if isempty(selected_cols)
        selected_cols = isempty(problem.free_cols) ? Int[] : [problem.free_cols[1]]
    end
    length(selected_cols) == 1 && !isempty(problem.free_cols) && begin
        for c in problem.free_cols
            if c != selected_cols[1]
                push!(selected_cols, c)
                break
            end
        end
    end

    colA = isempty(selected_cols) ? 0 : selected_cols[1]
    colB = length(selected_cols) >= 2 ? selected_cols[2] : 0

    tsv_path = joinpath(RESULTS_DIR, "$(spec.id)_scan.tsv")
    summary_path = joinpath(RESULTS_DIR, "$(spec.id)_summary.txt")

    if colA == 0
        open(tsv_path, "w") do io
            write_header(io)
            write_row(
                io,
                problem,
                "baseline",
                0,
                0,
                0,
                0,
                0 // 1,
                0 // 1,
                baseline,
            )
        end
        write_summary(summary_path, problem, baseline, probes, Int[], (best=(res=baseline, valA=0 // 1, valB=0 // 1), rows_written=1, ok_count=baseline.status == "ok" ? 1 : 0, strict_count=baseline.strict_hard ? 1 : 0), (best=(res=baseline, valA=0 // 1, valB=0 // 1), rows_written=1, ok_count=baseline.status == "ok" ? 1 : 0, strict_count=baseline.strict_hard ? 1 : 0))
        println("No free columns for ", spec.id, " - baseline only.")
        return nothing
    end

    centerA = default_center_for_col(problem, colA)
    centerB = colB > 0 ? default_center_for_col(problem, colB) : 0 // 1
    step1A = coarse_step_for_col(problem, colA)
    step1B = colB > 0 ? coarse_step_for_col(problem, colB) : (0 // 1)
    step2A = fine_step_for_col(problem, colA)
    step2B = colB > 0 ? fine_step_for_col(problem, colB) : (0 // 1)

    open(tsv_path, "w") do io
        write_header(io)
        write_row(io, problem, "baseline", 0, 0, colA, colB, centerA, centerB, baseline)

        stage1 = run_stage!(
            io,
            problem,
            colA,
            colB;
            stage="coarse",
            centerA=centerA,
            centerB=centerB,
            stepA=step1A,
            stepB=step1B,
            span=STAGE_SPAN,
        )
        println(
            "  stage1 spec=", spec.id,
            " rows=", stage1.rows_written,
            " ok=", stage1.ok_count,
            " strict=", stage1.strict_count,
            " best(max_real,max_imag)=(",
            stage1.best.res.max_real, ", ", stage1.best.res.max_imag, ")",
        )

        stage2 = run_stage!(
            io,
            problem,
            colA,
            colB;
            stage="fine",
            centerA=stage1.best.valA,
            centerB=stage1.best.valB,
            stepA=step2A,
            stepB=step2B,
            span=STAGE_SPAN,
        )
        println(
            "  stage2 spec=", spec.id,
            " rows=", stage2.rows_written,
            " ok=", stage2.ok_count,
            " strict=", stage2.strict_count,
            " best(max_real,max_imag)=(",
            stage2.best.res.max_real, ", ", stage2.best.res.max_imag, ")",
        )

        write_summary(summary_path, problem, baseline, probes, [colA, colB], stage1, stage2)
    end

    println("Wrote: ", tsv_path)
    println("Wrote: ", summary_path)
    return nothing
end

function main()
    specs = build_specs()
    if !isempty(ARGS) && ARGS[1] == "list"
        for s in specs
            println(s.id)
        end
        return
    end

    selected = if isempty(ARGS)
        specs
    else
        wanted = Set(ARGS)
        [s for s in specs if s.id in wanted]
    end

    isempty(selected) && error("No matching spec IDs. Run with `list` to show available specs.")

    for spec in selected
        run_spec(spec)
    end
end

main()
