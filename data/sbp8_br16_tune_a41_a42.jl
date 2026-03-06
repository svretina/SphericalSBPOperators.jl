using LinearAlgebra
using SparseArrays
using SummationByPartsOperators
using SphericalSBPOperators

const Mod = SphericalSBPOperators
rb(x) = Mod._sbp8_as_big_rational(x)

const BASE_ZERO_PAIRS = [(1, 2), (1, 3), (1, 4), (13, 16)]
const OUT_TSV = "data/sbp8_br16_a41_a42_tune_scan.tsv"
const OUT_SUMMARY = "data/sbp8_br16_a41_a42_tune_summary.txt"

# Strict target: L spectrum purely real (up to IMAG_TOL) and strictly negative (<= REAL_NEG_TOL).
const IMAG_TOL = 1e-12
const REAL_NEG_TOL = -1e-12

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

function is_strict_target(;
    status::String,
    max_real::Float64,
    max_imag::Float64,
    min_s::Float64,
    v_boundary_pd::Bool,
    v_full_pd::Bool,
)
    return status == "ok" &&
           min_s > 0.0 &&
           v_boundary_pd &&
           v_full_pd &&
           max_imag <= IMAG_TOL &&
           max_real <= REAL_NEG_TOL
end

function build_problem(; fr7::Int=11, br::Int=16, bw::Int=7)
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
    rows = Mod._sbp8_constraint_rows(N, closure_right, fr7)
    diag_free_indices = collect(1:24)
    pairs = Mod.sbp8_v_offdiag_pairs(N; boundary_rows=br, boundary_bandwidth=bw)

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

    for i in 24:N
        A, b = add_fix_col!(A, b, i, sq[i])
    end

    col24 = get(idx.vdiag_to_col, 24, 0)
    if col24 > 0
        A, b = add_fix_col!(A, b, col24, sq[24])
    end

    col_v1 = get(idx.vdiag_to_col, 1, 0)
    col_v1 > 0 || throw(ArgumentError("Missing Vdiag[1] column in system."))
    A, b = add_fix_col!(A, b, col_v1, rb(1))
    A, b = add_fix_col!(A, b, 1, rb(11) // rb(20))

    pair_to_k = Dict{Tuple{Int,Int},Int}(pr => k for (k, pr) in enumerate(pairs))
    for pr in BASE_ZERO_PAIRS
        k = get(pair_to_k, pr, 0)
        k == 0 && continue
        A, b = add_fix_col!(A, b, idx.idx_voff(k), rb(0))
    end

    k41 = get(pair_to_k, (14, 16), 0)
    k42 = get(pair_to_k, (15, 16), 0)
    k41 > 0 || throw(ArgumentError("Could not find pair (14,16) in br=$br, bw=$bw pair list."))
    k42 > 0 || throw(ArgumentError("Could not find pair (15,16) in br=$br, bw=$bw pair list."))
    col41 = idx.idx_voff(k41)
    col42 = idx.idx_voff(k42)

    x0 = Mod._solve_exact_linear_system(A, b)

    return (
        setup=setup,
        sys=sys,
        rows=rows,
        br=br,
        bw=bw,
        fr7=fr7,
        N=N,
        idx=idx,
        pairs=pairs,
        pair_to_k=pair_to_k,
        A=A,
        b=b,
        col41=col41,
        col42=col42,
        base_a41=x0[col41],
        base_a42=x0[col42],
    )
end

function evaluate_case(problem, a41::Rational{BigInt}, a42::Rational{BigInt})
    A2, b2 = add_fix_col!(problem.A, problem.b, problem.col41, a41)
    A2, b2 = add_fix_col!(A2, b2, problem.col42, a42)

    x = Mod._solve_exact_linear_system(A2, b2)
    idx = problem.idx
    n_s = idx.n_s
    n_voff = idx.n_voff
    N = problem.N

    sdiag_q = [x[i] for i in 1:n_s]
    min_s = minimum(Float64.(sdiag_q))
    if !(min_s > 0.0)
        return (
            status="s_nonpositive",
            reason="min_s=$min_s",
            rho=NaN,
            max_real=NaN,
            max_imag=NaN,
            err_r=NaN,
            err_r3=NaN,
            err_r5=NaN,
            err_r7=NaN,
            min_s=min_s,
            v_boundary_pd=false,
            v_full_pd=false,
            strict_target=false,
            a41=a41,
            a42=a42,
        )
    end

    vdiag_q = Vector{Rational{BigInt}}(undef, N)
    for i in 1:N
        c = get(idx.vdiag_to_col, i, 0)
        vdiag_q[i] = c > 0 ? x[c] : sdiag_q[i]
    end
    voff_q = [x[idx.idx_voff(k)] for k in 1:n_voff]

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
    err_r = Mod._sbp8_constraint_error_metrics(D, problem.sys.r, problem.sys.p, 1, problem.rows.rows_r).abs
    err_r3 = Mod._sbp8_constraint_error_metrics(D, problem.sys.r, problem.sys.p, 3, problem.rows.rows_r3).abs
    err_r5 = Mod._sbp8_constraint_error_metrics(D, problem.sys.r, problem.sys.p, 5, problem.rows.rows_r5).abs
    err_r7 = Mod._sbp8_constraint_error_metrics(D, problem.sys.r, problem.sys.p, 7, problem.rows.rows_r7).abs

    L = Matrix{Float64}(sparse(D * problem.sys.G))
    ev = eigen(L).values
    rho = maximum(abs.(ev))
    max_real = maximum(real.(ev))
    max_imag = maximum(abs.(imag.(ev)))

    v_boundary_pd = try
        cholesky(Symmetric(Matrix{Float64}(V[1:problem.br, 1:problem.br])); check=true)
        true
    catch
        false
    end
    v_full_pd = Mod._sbp8_is_pd(V)

    strict_target = is_strict_target(
        status="ok",
        max_real=max_real,
        max_imag=max_imag,
        min_s=min_s,
        v_boundary_pd=v_boundary_pd,
        v_full_pd=v_full_pd,
    )

    return (
        status="ok",
        reason="",
        rho=rho,
        max_real=max_real,
        max_imag=max_imag,
        err_r=err_r,
        err_r3=err_r3,
        err_r5=err_r5,
        err_r7=err_r7,
        min_s=min_s,
        v_boundary_pd=v_boundary_pd,
        v_full_pd=v_full_pd,
        strict_target=strict_target,
        a41=a41,
        a42=a42,
    )
end

function result_score(res)
    # Priority: valid solve -> strict target satisfaction -> tighten target gaps -> smaller rho.
    status_penalty = res.status == "ok" ? 0 : 1
    strict_penalty = res.strict_target ? 0 : 1
    imag_gap = res.status == "ok" ? max(res.max_imag - IMAG_TOL, 0.0) : Inf
    real_gap = res.status == "ok" ? max(res.max_real - REAL_NEG_TOL, 0.0) : Inf
    rho_value = res.status == "ok" ? res.rho : Inf
    return (status_penalty, strict_penalty, imag_gap, real_gap, res.max_imag, res.max_real, rho_value)
end

function target_score(res)
    return (res.rho, res.max_real, res.max_imag)
end

function write_header(io)
    println(io, join([
        "stage",
        "i",
        "j",
        "delta41",
        "delta42",
        "a41_float",
        "a42_float",
        "a41_rational",
        "a42_rational",
        "status",
        "strict_target",
        "rho",
        "max_real",
        "max_imag",
        "err_Dr",
        "err_Dr3",
        "err_Dr5",
        "err_Dr7",
        "min_s",
        "Vb_pd",
        "V_full_pd",
        "reason",
    ], '\t'))
end

function write_row(io, stage::String, i::Int, j::Int, delta41::Rational{BigInt}, delta42::Rational{BigInt}, res)
    println(io, join([
        stage,
        string(i),
        string(j),
        string(Float64(delta41)),
        string(Float64(delta42)),
        string(Float64(res.a41)),
        string(Float64(res.a42)),
        string(res.a41),
        string(res.a42),
        res.status,
        string(res.strict_target),
        string(res.rho),
        string(res.max_real),
        string(res.max_imag),
        string(res.err_r),
        string(res.err_r3),
        string(res.err_r5),
        string(res.err_r7),
        string(res.min_s),
        string(res.v_boundary_pd),
        string(res.v_full_pd),
        res.reason,
    ], '\t'))
end

function run_stage!(
    io,
    problem;
    stage::String,
    center41::Rational{BigInt},
    center42::Rational{BigInt},
    step::Rational{BigInt},
    span::Int,
)
    best = nothing
    best_target = nothing
    rows_written = 0
    ok_count = 0
    strict_count = 0

    for i in -span:span
        for j in -span:span
            delta41 = i * step
            delta42 = j * step
            a41 = center41 + delta41
            a42 = center42 + delta42

            res = try
                evaluate_case(problem, a41, a42)
            catch err
                (
                    status="fail",
                    strict_target=false,
                    reason=sprint(showerror, err),
                    rho=NaN,
                    max_real=NaN,
                    max_imag=NaN,
                    err_r=NaN,
                    err_r3=NaN,
                    err_r5=NaN,
                    err_r7=NaN,
                    min_s=NaN,
                    v_boundary_pd=false,
                    v_full_pd=false,
                    a41=a41,
                    a42=a42,
                )
            end

            write_row(io, stage, i, j, delta41, delta42, res)
            flush(io)
            rows_written += 1

            if res.status == "ok"
                ok_count += 1
                if res.strict_target
                    strict_count += 1
                    if isnothing(best_target) || target_score(res) < target_score(best_target)
                        best_target = res
                    end
                end
            end

            if isnothing(best) || result_score(res) < result_score(best)
                best = res
            end
        end
    end

    return (
        best=best,
        best_target=best_target,
        rows_written=rows_written,
        ok_count=ok_count,
        strict_count=strict_count,
    )
end

function write_best(io, label::String, res)
    println(io, label, ":")
    if isnothing(res)
        println(io, "  none")
        return
    end
    println(io, "  rho=", res.rho)
    println(io, "  max_real=", res.max_real)
    println(io, "  max_imag=", res.max_imag)
    println(io, "  strict_target=", res.strict_target)
    println(io, "  Vb_pd=", res.v_boundary_pd, ", V_full_pd=", res.v_full_pd)
    println(io, "  a41=", res.a41, " (", Float64(res.a41), ")")
    println(io, "  a42=", res.a42, " (", Float64(res.a42), ")")
end

function write_summary(path::String, problem, stage1, stage2)
    open(path, "w") do io
        println(io, "SBP8 br=16 a41/a42 tuning summary")
        println(io, "source = Mattsson2017(:central)")
        println(io, "points = 31")
        println(io, "fr7 = ", problem.fr7)
        println(io, "boundary_rows = ", problem.br)
        println(io, "boundary_bandwidth = ", problem.bw)
        println(io, "base_zero_pairs = ", BASE_ZERO_PAIRS)
        println(io, "a41 pair = (14,16), a42 pair = (15,16)")
        println(io, "strict target tolerances: max_imag <= ", IMAG_TOL,
            ", max_real <= ", REAL_NEG_TOL,
            ", S SPD, V boundary SPD, V full SPD")
        println(io)
        println(io, "baseline:")
        println(io, "  a41 = ", problem.base_a41, "  (", Float64(problem.base_a41), ")")
        println(io, "  a42 = ", problem.base_a42, "  (", Float64(problem.base_a42), ")")
        println(io)
        println(io, "stage1:")
        println(io, "  rows = ", stage1.rows_written)
        println(io, "  ok_count = ", stage1.ok_count)
        println(io, "  strict_count = ", stage1.strict_count)
        write_best(io, "  best_by_score", stage1.best)
        write_best(io, "  best_strict", stage1.best_target)
        println(io)
        println(io, "stage2:")
        println(io, "  rows = ", stage2.rows_written)
        println(io, "  ok_count = ", stage2.ok_count)
        println(io, "  strict_count = ", stage2.strict_count)
        write_best(io, "  best_by_score", stage2.best)
        write_best(io, "  best_strict", stage2.best_target)
        println(io)

        global_best = result_score(stage1.best) <= result_score(stage2.best) ? stage1.best : stage2.best
        write_best(io, "global_best_by_score", global_best)

        if !isnothing(stage1.best_target) && !isnothing(stage2.best_target)
            global_target = target_score(stage1.best_target) <= target_score(stage2.best_target) ? stage1.best_target : stage2.best_target
            write_best(io, "global_best_strict", global_target)
        elseif !isnothing(stage1.best_target)
            write_best(io, "global_best_strict", stage1.best_target)
        elseif !isnothing(stage2.best_target)
            write_best(io, "global_best_strict", stage2.best_target)
        else
            write_best(io, "global_best_strict", nothing)
        end
    end
end

function main()
    problem = build_problem(fr7=11, br=16, bw=7)
    println("Baseline a41 = ", problem.base_a41, " (", Float64(problem.base_a41), ")")
    println("Baseline a42 = ", problem.base_a42, " (", Float64(problem.base_a42), ")")
    println("Strict target tolerances: max_imag <= ", IMAG_TOL, ", max_real <= ", REAL_NEG_TOL)

    open(OUT_TSV, "w") do io
        write_header(io)

        stage1 = run_stage!(
            io,
            problem;
            stage="coarse",
            center41=problem.base_a41,
            center42=problem.base_a42,
            step=rb(1) // rb(100000),   # 1e-5
            span=10,                    # +/-1e-4
        )
        println("Stage 1 done: rows=", stage1.rows_written,
            ", ok=", stage1.ok_count,
            ", strict=", stage1.strict_count,
            ", best(max_real,max_imag)=(", stage1.best.max_real, ", ", stage1.best.max_imag, ")")

        stage2_center = isnothing(stage1.best_target) ? stage1.best : stage1.best_target
        stage2 = run_stage!(
            io,
            problem;
            stage="fine",
            center41=stage2_center.a41,
            center42=stage2_center.a42,
            step=rb(1) // rb(1000000),  # 1e-6
            span=10,                    # +/-1e-5
        )
        println("Stage 2 done: rows=", stage2.rows_written,
            ", ok=", stage2.ok_count,
            ", strict=", stage2.strict_count,
            ", best(max_real,max_imag)=(", stage2.best.max_real, ", ", stage2.best.max_imag, ")")

        write_summary(OUT_SUMMARY, problem, stage1, stage2)
    end

    println("Wrote: ", OUT_TSV)
    println("Wrote: ", OUT_SUMMARY)
end

main()
