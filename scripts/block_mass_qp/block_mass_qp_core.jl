using JuMP
import Ipopt
using LinearAlgebra: ColumnNorm, Diagonal, dot, norm, qr, rank, svdvals
using Printf: @printf
using SummationByPartsOperators: MattssonNordström2004, derivative_operator

const MOI = JuMP.MOI

Base.@kwdef struct BlockMassQPConfig
    R::Float64 = 30.0
    d::Int = 4
    p::Int = 2
    epsilon::Union{Nothing, Float64} = 1e-3
    quadrature_cap::Union{Nothing, Int} = nothing
    quad_max_degree::Union{Nothing, Int} = nothing
    div_max_odd::Union{Nothing, Int} = nothing
    boundary_div_max_odd::Union{Nothing, Int} = nothing
    block_radius_start::Int = 0
    block_radius_max::Union{Nothing, Int} = nothing
    block_radius_step::Int = 1
    offdiag_penalty::Float64 = 1.0
    enforce_reflection_symmetry::Bool = true
    stop_on_first_feasible::Bool = true
    solve_tol::Float64 = 1e-8
    ipopt_tol::Float64 = 1e-12
    ipopt_acceptable_tol::Float64 = 1e-12
    ipopt_constr_viol_tol::Float64 = 1e-12
    ipopt_print_level::Int = 0
end

@inline _odd_cap(k::Int) = k < 1 ? 0 : (isodd(k) ? k : k - 1)
@inline _odd_degrees(k::Int) = k < 1 ? Int[] : collect(1:2:_odd_cap(k))
@inline _mirror(i::Int, N::Int) = N + 1 - i

function _integral_rhs(R::Float64, p::Int, q::Int)
    exponent = q + p
    if isodd(exponent)
        return 0.0
    end
    return 2.0 * R^(q + p + 1) / (q + p + 1)
end

function _canonical_grid(R::Float64)
    Rint = round(Int, R)
    abs(R - Rint) <= 1e-12 || throw(ArgumentError("R must be integer-valued for x=-R:1:R; got R=$R"))
    x = collect(-Float64(Rint):1.0:Float64(Rint))
    N = length(x)
    isodd(N) || throw(ArgumentError("Expected odd N on symmetric grid; got N=$N"))
    return x, Float64(Rint)
end

function _infer_boundary_count(Dcart, G::Matrix{Float64}; atol::Float64 = 1e-13)
    if hasproperty(Dcart, :coefficients)
        coeffs = getproperty(Dcart, :coefficients)
        if hasproperty(coeffs, :left_weights)
            return length(getproperty(coeffs, :left_weights)), :left_weights
        elseif hasproperty(coeffs, :right_weights)
            return length(getproperty(coeffs, :right_weights)), :right_weights
        end
    end

    n = size(G, 1)
    ref = clamp(fld(n, 2), 1, n)
    function pattern(i::Int)
        idx = Int[]
        @inbounds for j in 1:n
            abs(G[i, j]) > atol && push!(idx, j - i)
        end
        return idx
    end
    pref = pattern(ref)
    left = 0
    @inbounds for i in 1:n
        pattern(i) == pref || (left += 1; continue)
        break
    end
    return left, :pattern_fallback
end

function _infer_boundary_accuracy(G::Matrix{Float64}, x::Vector{Float64}, d::Int; atol::Float64 = 1e-12)
    n = length(x)
    n == size(G, 1) == size(G, 2) || throw(DimensionMismatch("G/x size mismatch"))
    row = 1
    max_ok = -1
    @inbounds for degree in 0:d
        u = x .^ degree
        exact = degree == 0 ? 0.0 : degree * x[row]^(degree - 1)
        num = dot(G[row, :], u)
        if abs(num - exact) <= atol * max(1.0, abs(exact))
            max_ok = degree
        else
            break
        end
    end
    return max_ok >= 0 ? max_ok : nothing
end

function _build_block_pairs(N::Int, c::Int, radius::Int)
    radius >= 0 || throw(ArgumentError("block radius must be non-negative"))
    lo = max(1, c - radius)
    hi = min(N, c + radius)
    pairs = Tuple{Int, Int}[]
    for i in lo:hi
        for j in (i + 1):hi
            push!(pairs, (i, j))
        end
    end
    return pairs, lo, hi
end

function _add_row!(Arows::Vector{Vector{Float64}},
                   beq::Vector{Float64},
                   fam::Dict{Symbol, Vector{Int}},
                   key::Symbol,
                   coeff::Vector{Float64},
                   rhs::Float64)
    push!(Arows, coeff)
    push!(beq, rhs)
    push!(fam[key], length(Arows))
    return nothing
end

function _build_linear_system(cfg::BlockMassQPConfig, radius::Int)
    x, R = _canonical_grid(cfg.R)
    N = length(x)
    c = (N + 1) ÷ 2

    source = MattssonNordström2004()
    Dcart = derivative_operator(source;
                                derivative_order = 1,
                                accuracy_order = cfg.d,
                                xmin = -R,
                                xmax = R,
                                N = N)
    G = Matrix(Dcart)
    Bdiag = zeros(Float64, N)
    Bdiag[1] = -R^cfg.p
    Bdiag[end] = R^cfg.p

    boundary_count, boundary_count_source = _infer_boundary_count(Dcart, G)
    boundary_accuracy = _infer_boundary_accuracy(G, x, cfg.d)
    if boundary_accuracy === nothing
        if cfg.d == 6
            boundary_accuracy = 3
        else
            throw(ArgumentError("Failed to infer boundary accuracy for d=$(cfg.d)."))
        end
    end

    quad_cap_default = fld(cfg.d, 2)
    quad_cap = isnothing(cfg.quadrature_cap) ? quad_cap_default : max(0, cfg.quadrature_cap)
    quad_default = min(2 * cfg.d - cfg.p - 1, quad_cap)
    quad_max_degree = isnothing(cfg.quad_max_degree) ? quad_default : max(0, cfg.quad_max_degree)

    interior_default = _odd_cap(cfg.d - cfg.p)
    interior_max_odd = isnothing(cfg.div_max_odd) ? interior_default : _odd_cap(cfg.div_max_odd)
    boundary_default = _odd_cap(boundary_accuracy - cfg.p)
    boundary_max_odd = isnothing(cfg.boundary_div_max_odd) ? boundary_default : _odd_cap(cfg.boundary_div_max_odd)

    pairs, lo, hi = _build_block_pairs(N, c, radius)
    npairs = length(pairs)
    nvar = N + npairs

    diag_target = abs.(x) .^ cfg.p

    Arows = Vector{Vector{Float64}}()
    beq = Float64[]
    family = Dict{Symbol, Vector{Int}}(
        :diag_symmetry => Int[],
        :offdiag_reflection => Int[],
        :quadrature => Int[],
        :divergence => Int[],
    )
    divergence_degrees = Int[]

    # Diagonal reflection symmetry: M[i,i] = M[mirror(i),mirror(i)]
    for i in 1:(c - 1)
        mi = _mirror(i, N)
        coeff = zeros(Float64, nvar)
        coeff[i] = 1.0
        coeff[mi] = -1.0
        _add_row!(Arows, beq, family, :diag_symmetry, coeff, 0.0)
    end

    if cfg.enforce_reflection_symmetry && npairs > 0
        pair_to_idx = Dict{Tuple{Int, Int}, Int}()
        for (idx, p) in enumerate(pairs)
            pair_to_idx[p] = idx
        end

        for (idx, (i, j)) in enumerate(pairs)
            mi = _mirror(i, N)
            mj = _mirror(j, N)
            rp = mi < mj ? (mi, mj) : (mj, mi)
            jdx = get(pair_to_idx, rp, 0)
            if jdx > 0 && idx < jdx
                coeff = zeros(Float64, nvar)
                coeff[N + idx] = 1.0
                coeff[N + jdx] = -1.0
                _add_row!(Arows, beq, family, :offdiag_reflection, coeff, 0.0)
            end
        end
    end

    # Quadrature: 1^T M x^q = ∫ x^q r^p dr
    for q in 0:quad_max_degree
        coeff = zeros(Float64, nvar)
        for i in 1:N
            coeff[i] = x[i]^q
        end
        for (t, (a, b)) in enumerate(pairs)
            coeff[N + t] = x[a]^q + x[b]^q
        end
        rhs = _integral_rhs(R, cfg.p, q)
        _add_row!(Arows, beq, family, :quadrature, coeff, rhs)
    end

    # Divergence constraints from SBP: M d + G^T M x^k = B x^k
    for i in 1:N
        boundary_row = i <= boundary_count || i > (N - boundary_count)
        kmax = boundary_row ? boundary_max_odd : interior_max_odd
        for k in _odd_degrees(kmax)
            if !(k in divergence_degrees)
                push!(divergence_degrees, k)
            end
            dvec = (cfg.p + k) .* (x .^ (k - 1))
            xk = x .^ k

            coeff = zeros(Float64, nvar)
            for a in 1:N
                coeff[a] = (i == a ? dvec[a] : 0.0) + G[a, i] * xk[a]
            end
            for (t, (a, b)) in enumerate(pairs)
                ca = (i == a ? dvec[b] : 0.0)
                cb = (i == b ? dvec[a] : 0.0)
                cg = G[a, i] * xk[b] + G[b, i] * xk[a]
                coeff[N + t] = ca + cb + cg
            end

            rhs = Bdiag[i] * xk[i]
            _add_row!(Arows, beq, family, :divergence, coeff, rhs)
        end
    end

    n_eq = length(Arows)
    Aeq = zeros(Float64, n_eq, nvar)
    for r in 1:n_eq
        Aeq[r, :] .= Arows[r]
    end

    return (
        x = x,
        R = R,
        N = N,
        c = c,
        G = G,
        Bdiag = Bdiag,
        boundary_count = boundary_count,
        boundary_count_source = boundary_count_source,
        boundary_accuracy = boundary_accuracy,
        quadrature_cap = quad_cap,
        quad_max_degree = quad_max_degree,
        interior_max_odd = interior_max_odd,
        boundary_max_odd = boundary_max_odd,
        pairs = pairs,
        block_lo = lo,
        block_hi = hi,
        diag_target = diag_target,
        Aeq = Aeq,
        beq = beq,
        family = family,
        divergence_degrees = sort!(divergence_degrees),
    )
end

function _independent_row_indices(Aeq::Matrix{Float64})
    nrows, _ = size(Aeq)
    nrows == 0 && return Int[]
    r = rank(Aeq)
    r == 0 && return Int[]
    F = qr(transpose(Aeq), ColumnNorm())
    piv = Vector(F.p)
    return sort(piv[1:r])
end

function _assemble_mass(z::Vector{Float64}, pairs::Vector{Tuple{Int, Int}}, N::Int)
    M = zeros(Float64, N, N)
    for i in 1:N
        M[i, i] = z[i]
    end
    for (t, (a, b)) in enumerate(pairs)
        v = z[N + t]
        M[a, b] = v
        M[b, a] = v
    end
    return M
end

function _print_family_residuals(res::Vector{Float64}, fam::Dict{Symbol, Vector{Int}})
    for key in (:diag_symmetry, :offdiag_reflection, :quadrature, :divergence)
        idx = get(fam, key, Int[])
        isempty(idx) && continue
        rr = res[idx]
        @printf("    %-18s rows=%4d  ||r||2=%12.4e  ||r||∞=%12.4e\n",
                string(key), length(idx), norm(rr), maximum(abs.(rr)))
    end
end

function _solve_block_radius(sys, cfg::BlockMassQPConfig)
    Aeq = sys.Aeq
    beq = sys.beq
    n_eq, n_var = size(Aeq)
    n_diag = sys.N

    rankA = rank(Aeq)
    rankAug = rank(hcat(Aeq, beq))
    svals = svdvals(Aeq)
    sigma_max = isempty(svals) ? 0.0 : maximum(svals)
    sigma_min = isempty(svals) ? 0.0 : minimum(svals)
    condA = sigma_min > 0 ? sigma_max / sigma_min : Inf

    z_ls = Aeq \ beq
    r_ls = Aeq * z_ls - beq

    rows_opt = n_eq > n_var ? _independent_row_indices(Aeq) : collect(1:n_eq)
    Aopt = Aeq[rows_opt, :]
    bopt = beq[rows_opt]

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol", cfg.ipopt_tol)
    set_optimizer_attribute(model, "acceptable_tol", cfg.ipopt_acceptable_tol)
    set_optimizer_attribute(model, "constr_viol_tol", cfg.ipopt_constr_viol_tol)
    set_optimizer_attribute(model, "print_level", cfg.ipopt_print_level)

    @variable(model, z[1:n_var])
    if cfg.epsilon !== nothing
        for i in 1:n_diag
            set_lower_bound(z[i], cfg.epsilon)
        end
    end

    @objective(model, Min,
        sum((z[i] - sys.diag_target[i])^2 for i in 1:n_diag) +
        cfg.offdiag_penalty * sum(z[n_diag + t]^2 for t in 1:length(sys.pairs))
    )

    for r in 1:size(Aopt, 1)
        @constraint(model, sum(Aopt[r, j] * z[j] for j in 1:n_var) == bopt[r])
    end

    optimize!(model)

    term = termination_status(model)
    primal = primal_status(model)
    dual = dual_status(model)
    has_vals = has_values(model)

    z_opt = has_vals ? value.(z) : z_ls
    r_opt = Aeq * z_opt - beq
    maxeq = maximum(abs.(r_opt))

    feasible = has_vals && maxeq <= cfg.solve_tol

    M = _assemble_mass(z_opt, sys.pairs, sys.N)
    quad_err = Dict{Int, Float64}()
    for q in 0:sys.quad_max_degree
        xq = sys.x .^ q
        lhs = sum(M * xq)
        rhs_q = _integral_rhs(sys.R, cfg.p, q)
        quad_err[q] = abs(lhs - rhs_q)
    end
    max_quad_err = isempty(quad_err) ? NaN : maximum(values(quad_err))

    D = nothing
    div_err = Dict{Int, Float64}()
    rhs = Diagonal(sys.Bdiag) - transpose(sys.G) * M
    D = try
        M \ rhs
    catch
        nothing
    end
    if D !== nothing
        for k in sys.divergence_degrees
            xk = sys.x .^ k
            exact = (cfg.p + k) .* (sys.x .^ (k - 1))
            div_err[k] = maximum(abs.(D * xk - exact))
        end
    end
    max_div_err = isempty(div_err) ? NaN : maximum(values(div_err))

    return (
        n_eq = n_eq,
        n_var = n_var,
        rankA = rankA,
        rankAug = rankAug,
        sigma_max = sigma_max,
        sigma_min = sigma_min,
        condA = condA,
        z_ls = z_ls,
        r_ls = r_ls,
        term = term,
        primal = primal,
        dual = dual,
        has_vals = has_vals,
        z_opt = z_opt,
        r_opt = r_opt,
        maxeq = maxeq,
        feasible = feasible,
        M = M,
        quad_err = quad_err,
        max_quad_err = max_quad_err,
        D = D,
        div_err = div_err,
        max_div_err = max_div_err,
    )
end

function run_block_mass_qp(cfg::BlockMassQPConfig = BlockMassQPConfig())
    cfg.d > 0 || throw(ArgumentError("d must be positive"))
    cfg.p >= 0 || throw(ArgumentError("p must be non-negative"))
    cfg.block_radius_step > 0 || throw(ArgumentError("block_radius_step must be positive"))
    cfg.offdiag_penalty > 0 || throw(ArgumentError("offdiag_penalty must be positive"))
    if cfg.epsilon !== nothing
        cfg.epsilon > 0 || throw(ArgumentError("epsilon must be positive when provided"))
    end

    x, _ = _canonical_grid(cfg.R)
    N = length(x)
    c = (N + 1) ÷ 2
    rmin = max(0, cfg.block_radius_start)
    rmax_default = c - 1
    rmax = isnothing(cfg.block_radius_max) ? rmax_default : min(max(cfg.block_radius_max, rmin), rmax_default)

    println("Block-mass QP sweep")
    println("  grid: R=$(cfg.R), N=$N, center_index=$c, dx=1")
    println("  settings: d=$(cfg.d), p=$(cfg.p), epsilon=$(cfg.epsilon), offdiag_penalty=$(cfg.offdiag_penalty), reflection_symmetry=$(cfg.enforce_reflection_symmetry)")
    println("  radius sweep: $rmin:$(cfg.block_radius_step):$rmax")

    println("\n  radius  block_size   vars   eqs  rankA rankAug mismatch  term                feasible   max|A z-b|  max_quad_err  max_div_err")

    results = NamedTuple[]
    first_feasible = nothing

    for radius in rmin:cfg.block_radius_step:rmax
        sys = _build_linear_system(cfg, radius)
        sol = _solve_block_radius(sys, cfg)
        block_size = sys.block_hi - sys.block_lo + 1
        mismatch = sol.rankAug > sol.rankA ? "yes" : "no"

        @printf("  %6d  %10d %6d %5d %6d %7d %8s  %-18s %-8s %12.4e %13.4e %12.4e\n",
                radius,
                block_size,
                sol.n_var,
                sol.n_eq,
                sol.rankA,
                sol.rankAug,
                mismatch,
                string(sol.term),
                sol.feasible ? "yes" : "no",
                sol.maxeq,
                sol.max_quad_err,
                sol.max_div_err)

        push!(results, (radius = radius, system = sys, solution = sol))

        if sol.feasible && isnothing(first_feasible)
            first_feasible = last(results)
            println("\nFeasible solve found at radius=$radius.")
            println("  rank(A)= $(sol.rankA), rank([A|b])= $(sol.rankAug), cond(A)= $(sol.condA)")
            println("  full residual norms: ||r||2 = $(norm(sol.r_opt)), ||r||∞ = $(maximum(abs.(sol.r_opt)))")
            println("  max quadrature error = $(sol.max_quad_err)")
            println("  max divergence error = $(sol.max_div_err)")
            _print_family_residuals(sol.r_opt, sys.family)
            println("  quadrature verification errors:")
            for q in sort!(collect(keys(sol.quad_err)))
                @printf("    q=%d: |Q_q - I_q| = %.6e\n", q, sol.quad_err[q])
            end
            if !isempty(sol.div_err)
                println("  divergence verification errors:")
                for k in sort!(collect(keys(sol.div_err)))
                    @printf("    k=%d: max|D*x^k - d^(k)| = %.6e\n", k, sol.div_err[k])
                end
            end
            if cfg.stop_on_first_feasible
                break
            end
        end
    end

    if isnothing(first_feasible)
        println("\nNo feasible solve found in the requested radius range.")
    end

    return (config = cfg, results = results, first_feasible = first_feasible)
end

function print_block_cli_help(io::IO = stdout)
    println(io, "Usage: julia scripts/block_mass_qp/run_block_mass_qp.jl [options]")
    println(io, "")
    println(io, "Options:")
    println(io, "  --R <float>                      Domain half-width (integer-valued). Default: 30")
    println(io, "  --d <int>                        Gradient order. Default: 4")
    println(io, "  --p <int>                        Geometry power. Default: 2")
    println(io, "  --epsilon <float>                Diagonal lower bound (default: 1e-3). Use <=0 to disable")
    println(io, "  --quadrature-cap <int>           Cap for auto quadrature degree (default d/2)")
    println(io, "  --quad-max-degree <int>          Explicit quadrature max degree override")
    println(io, "  --div-max-odd <int>              Interior odd divergence max degree override")
    println(io, "  --boundary-div-max-odd <int>     Boundary odd divergence max degree override")
    println(io, "  --block-radius-start <int>       Start radius around origin. Default: 0")
    println(io, "  --block-radius-max <int>         Max radius around origin. Default: full half-grid")
    println(io, "  --block-radius-step <int>        Radius increment. Default: 1")
    println(io, "  --offdiag-penalty <float>        Objective weight on off-diagonal DOFs. Default: 1")
    println(io, "  --disable-reflection-symmetry    Do not constrain mirrored off-diagonals equal")
    println(io, "  --no-stop                        Continue sweep after first feasible radius")
    println(io, "  --solve-tol <float>              Full-system feasibility tolerance. Default: 1e-8")
    println(io, "  --ipopt-tol <float>              Ipopt tol. Default: 1e-12")
    println(io, "  --ipopt-acceptable-tol <float>   Ipopt acceptable_tol. Default: 1e-12")
    println(io, "  --ipopt-constr-viol-tol <float>  Ipopt constr_viol_tol. Default: 1e-12")
    println(io, "  --ipopt-print-level <int>        Ipopt print_level. Default: 0")
    println(io, "  -h, --help                       Show this help text")
end

function parse_block_cli_args(args::Vector{String})
    cfg = BlockMassQPConfig()
    R = cfg.R
    d = cfg.d
    p = cfg.p
    epsilon = cfg.epsilon
    quadrature_cap = cfg.quadrature_cap
    quad_max_degree = cfg.quad_max_degree
    div_max_odd = cfg.div_max_odd
    boundary_div_max_odd = cfg.boundary_div_max_odd
    block_radius_start = cfg.block_radius_start
    block_radius_max = cfg.block_radius_max
    block_radius_step = cfg.block_radius_step
    offdiag_penalty = cfg.offdiag_penalty
    enforce_reflection_symmetry = cfg.enforce_reflection_symmetry
    stop_on_first_feasible = cfg.stop_on_first_feasible
    solve_tol = cfg.solve_tol
    ipopt_tol = cfg.ipopt_tol
    ipopt_acceptable_tol = cfg.ipopt_acceptable_tol
    ipopt_constr_viol_tol = cfg.ipopt_constr_viol_tol
    ipopt_print_level = cfg.ipopt_print_level

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "-h" || arg == "--help"
            print_block_cli_help()
            return nothing
        elseif arg == "--disable-reflection-symmetry"
            enforce_reflection_symmetry = false
            i += 1
            continue
        elseif arg == "--no-stop"
            stop_on_first_feasible = false
            i += 1
            continue
        end

        i < length(args) || throw(ArgumentError("Missing value for argument $arg"))
        val = args[i + 1]
        if arg == "--R"
            R = parse(Float64, val)
        elseif arg == "--d"
            d = parse(Int, val)
        elseif arg == "--p"
            p = parse(Int, val)
        elseif arg == "--epsilon"
            eps = parse(Float64, val)
            epsilon = eps > 0 ? eps : nothing
        elseif arg == "--quadrature-cap"
            quadrature_cap = parse(Int, val)
        elseif arg == "--quad-max-degree"
            quad_max_degree = parse(Int, val)
        elseif arg == "--div-max-odd"
            div_max_odd = parse(Int, val)
        elseif arg == "--boundary-div-max-odd"
            boundary_div_max_odd = parse(Int, val)
        elseif arg == "--block-radius-start"
            block_radius_start = parse(Int, val)
        elseif arg == "--block-radius-max"
            block_radius_max = parse(Int, val)
        elseif arg == "--block-radius-step"
            block_radius_step = parse(Int, val)
        elseif arg == "--offdiag-penalty"
            offdiag_penalty = parse(Float64, val)
        elseif arg == "--solve-tol"
            solve_tol = parse(Float64, val)
        elseif arg == "--ipopt-tol"
            ipopt_tol = parse(Float64, val)
        elseif arg == "--ipopt-acceptable-tol"
            ipopt_acceptable_tol = parse(Float64, val)
        elseif arg == "--ipopt-constr-viol-tol"
            ipopt_constr_viol_tol = parse(Float64, val)
        elseif arg == "--ipopt-print-level"
            ipopt_print_level = parse(Int, val)
        else
            throw(ArgumentError("Unknown argument `$arg`. Use --help for usage."))
        end
        i += 2
    end

    return BlockMassQPConfig(
        R = R,
        d = d,
        p = p,
        epsilon = epsilon,
        quadrature_cap = quadrature_cap,
        quad_max_degree = quad_max_degree,
        div_max_odd = div_max_odd,
        boundary_div_max_odd = boundary_div_max_odd,
        block_radius_start = block_radius_start,
        block_radius_max = block_radius_max,
        block_radius_step = block_radius_step,
        offdiag_penalty = offdiag_penalty,
        enforce_reflection_symmetry = enforce_reflection_symmetry,
        stop_on_first_feasible = stop_on_first_feasible,
        solve_tol = solve_tol,
        ipopt_tol = ipopt_tol,
        ipopt_acceptable_tol = ipopt_acceptable_tol,
        ipopt_constr_viol_tol = ipopt_constr_viol_tol,
        ipopt_print_level = ipopt_print_level,
    )
end
