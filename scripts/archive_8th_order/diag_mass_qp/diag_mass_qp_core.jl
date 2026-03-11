using JuMP
import Ipopt
using LinearAlgebra: Diagonal, dot, norm, rank, svdvals
using Printf: @printf
using SummationByPartsOperators: MattssonNordström2004, derivative_operator

const MOI = JuMP.MOI

Base.@kwdef struct DiagMassQPConfig
    R::Float64 = 10.0
    d::Int = 6
    p::Int = 2
    quadrature_cap::Union{Nothing, Int} = nothing
    epsilon::Union{Nothing, Float64} = nothing
    anti_stiffness::Bool = false
    quad_max_degree::Union{Nothing, Int} = nothing
    div_max_odd::Union{Nothing, Int} = nothing
    boundary_div_max_odd::Union{Nothing, Int} = nothing
    ipopt_tol::Float64 = 1e-12
    ipopt_acceptable_tol::Float64 = 1e-12
    ipopt_constr_viol_tol::Float64 = 1e-12
    ipopt_print_level::Int = 0
    print_matrices::Bool = false
end

@inline _odd_cap(k::Int) = k < 1 ? 0 : (isodd(k) ? k : k - 1)
@inline _odd_degrees(k::Int) = k < 1 ? Int[] : collect(1:2:_odd_cap(k))

function _integral_rhs(R::Float64, q::Int)
    exponent = q + 2
    if isodd(exponent)
        return 0.0
    end
    return 2.0 * R^(q + 3) / (q + 3)
end

function _next_cell_volume_from_origin(x::Vector{Float64})
    N = length(x)
    c = (N + 1) ÷ 2
    c < N || throw(ArgumentError("Need at least one node to the right of r=0 to define next-cell volume."))
    dx = x[c + 1] - x[c]
    dx > 0 || throw(ArgumentError("Expected positive spacing to the next cell from r=0."))
    # Spherical cell-volume weight: ∫_0^Δr r^2 dr = Δr^3 / 3.
    return dx^3 / 3
end

function _canonical_grid(R::Float64)
    Rint = round(Int, R)
    abs(R - Rint) <= 1e-12 ||
        throw(ArgumentError("R must be an integer so x = -R:1:R has center x=0 exactly; got R=$R."))
    x = collect(-Float64(Rint):1.0:Float64(Rint))
    N = length(x)
    isodd(N) || throw(ArgumentError("Expected odd N for symmetric grid; got N=$N."))
    return x, Float64(Rint)
end

function _infer_boundary_count(Dcart, G::Matrix{Float64}; atol::Float64 = 1e-13)
    if hasproperty(Dcart, :coefficients)
        coeffs = getproperty(Dcart, :coefficients)
        if hasproperty(coeffs, :left_weights)
            lw = getproperty(coeffs, :left_weights)
            return length(lw), :left_weights
        elseif hasproperty(coeffs, :right_weights)
            rw = getproperty(coeffs, :right_weights)
            return length(rw), :right_weights
        end
    end

    n = size(G, 1)
    ref = clamp(fld(n, 2), 1, n)
    function row_pattern(i::Int)
        idx = Int[]
        @inbounds for j in 1:n
            abs(G[i, j]) > atol && push!(idx, j - i)
        end
        return idx
    end
    pref = row_pattern(ref)
    left = 0
    @inbounds for i in 1:n
        row_pattern(i) == pref || (left += 1; continue)
        break
    end
    return left, :pattern_fallback
end

function _infer_boundary_accuracy(G::Matrix{Float64}, x::Vector{Float64}, d::Int; atol::Float64 = 1e-12)
    N = length(x)
    N == size(G, 1) == size(G, 2) || throw(DimensionMismatch("G/x size mismatch while inferring boundary accuracy."))
    row = 1
    max_ok = -1
    @inbounds for degree in 0:d
        u = x .^ degree
        exact = degree == 0 ? 0.0 : degree * x[row]^(degree - 1)
        num = dot(G[row, :], u)
        scale = max(1.0, abs(exact))
        if abs(num - exact) <= atol * scale
            max_ok = degree
        else
            break
        end
    end
    return max_ok >= 0 ? max_ok : nothing
end

function _assemble_constraints(x::Vector{Float64},
                               G::Matrix{Float64},
                               Bdiag::Vector{Float64},
                               boundary_count::Int,
                               boundary_accuracy::Int,
                               cfg::DiagMassQPConfig)
    N = length(x)
    c = (N + 1) ÷ 2

    quadrature_cap_default = fld(cfg.d, 2)
    quadrature_cap = isnothing(cfg.quadrature_cap) ? quadrature_cap_default : max(0, cfg.quadrature_cap)

    quad_default = min(2 * cfg.d - cfg.p - 1, quadrature_cap)
    quad_max_degree = isnothing(cfg.quad_max_degree) ? quad_default : cfg.quad_max_degree
    quad_max_degree = max(0, quad_max_degree)

    interior_default = _odd_cap(cfg.d - cfg.p)
    interior_max_odd = isnothing(cfg.div_max_odd) ? interior_default : _odd_cap(cfg.div_max_odd)

    boundary_default = _odd_cap(boundary_accuracy - cfg.p)
    boundary_max_odd = isnothing(cfg.boundary_div_max_odd) ? boundary_default : _odd_cap(cfg.boundary_div_max_odd)

    rows = Vector{Vector{Float64}}()
    rhs = Float64[]
    family_rows = Dict{Symbol, Vector{Int}}(
        :symmetry => Int[],
        :anti => Int[],
        :quadrature => Int[],
        :divergence => Int[],
    )
    enforced_rows_by_degree = Dict{Int, Vector{Int}}()

    function add_row!(coeff::Vector{Float64}, b::Float64, fam::Symbol)
        push!(rows, copy(coeff))
        push!(rhs, b)
        push!(family_rows[fam], length(rows))
        return nothing
    end

    # Symmetry constraints around center.
    for k in 1:(c - 1)
        coeff = zeros(Float64, N)
        coeff[c - k] = 1.0
        coeff[c + k] = -1.0
        add_row!(coeff, 0.0, :symmetry)
    end

    # Anti-stiffness constraint.
    if cfg.anti_stiffness
        coeff = zeros(Float64, N)
        coeff[c] = 1.0
        coeff[c + 1] = -1.0
        add_row!(coeff, 0.0, :anti)
    end

    # Quadrature constraints.
    for q in 0:quad_max_degree
        coeff = x .^ q
        b = _integral_rhs(x[end], q)
        add_row!(coeff, b, :quadrature)
    end

    # Divergence constraints with boundary-lowered odd degree.
    @inbounds for i in 1:N
        boundary_row = i <= boundary_count || i > (N - boundary_count)
        local_max_odd = boundary_row ? boundary_max_odd : interior_max_odd
        for k in _odd_degrees(local_max_odd)
            xk = x .^ k
            coeff = zeros(Float64, N)
            for j in 1:N
                coeff[j] = G[j, i] * xk[j]
            end
            coeff[i] += (k + cfg.p) * x[i]^(k - 1)
            b = Bdiag[i] * xk[i]
            add_row!(coeff, b, :divergence)

            if !haskey(enforced_rows_by_degree, k)
                enforced_rows_by_degree[k] = Int[]
            end
            push!(enforced_rows_by_degree[k], i)
        end
    end

    n_eq = length(rows)
    Aeq = zeros(Float64, n_eq, N)
    @inbounds for r in 1:n_eq
        Aeq[r, :] .= rows[r]
    end
    beq = copy(rhs)

    return (
        Aeq = Aeq,
        beq = beq,
        family_rows = family_rows,
        enforced_rows_by_degree = enforced_rows_by_degree,
        quadrature_cap = quadrature_cap,
        quad_max_degree = quad_max_degree,
        interior_max_odd = interior_max_odd,
        boundary_max_odd = boundary_max_odd,
    )
end

function _print_family_residuals(residual::Vector{Float64},
                                 family_rows::Dict{Symbol, Vector{Int}};
                                 label::String = "")
    isempty(label) || println(label)
    for fam in (:symmetry, :anti, :quadrature, :divergence)
        idx = get(family_rows, fam, Int[])
        isempty(idx) && continue
        r = residual[idx]
        @printf("  %-10s rows=%3d  ||r||2=%12.4e  ||r||∞=%12.4e\n",
                string(fam), length(idx), norm(r), maximum(abs.(r)))
    end
end

function _print_linear_diagnostics(Aeq::Matrix{Float64},
                                   beq::Vector{Float64},
                                   family_rows::Dict{Symbol, Vector{Int}};
                                   m_opt::Union{Nothing, Vector{Float64}} = nothing,
                                   print_matrices::Bool = false)
    println("\nLinear diagnostics (hard equalities)")
    println("  size(Aeq) = ", size(Aeq))
    rankA = rank(Aeq)
    rankAug = rank(hcat(Aeq, beq))
    svals = svdvals(Aeq)
    smax = isempty(svals) ? 0.0 : maximum(svals)
    smin = isempty(svals) ? 0.0 : minimum(svals)
    condA = smin > 0 ? smax / smin : Inf
    println("  rank(Aeq) = ", rankA)
    println("  rank([Aeq|beq]) = ", rankAug)
    @printf("  sigma_max = %.6e, sigma_min = %.6e, cond(Aeq) = %.6e\n", smax, smin, condA)

    m_ls = Aeq \ beq
    r_ls = Aeq * m_ls - beq
    @printf("  LS residual: ||Aeq*m_ls-beq||2 = %.6e, ||.||∞ = %.6e\n",
            norm(r_ls), maximum(abs.(r_ls)))
    _print_family_residuals(r_ls, family_rows; label = "  Per-family LS residuals:")

    if m_opt !== nothing
        r_opt = Aeq * m_opt - beq
        @printf("  Optimizer-point residual: ||Aeq*m-beq||2 = %.6e, ||.||∞ = %.6e\n",
                norm(r_opt), maximum(abs.(r_opt)))
        _print_family_residuals(r_opt, family_rows; label = "  Per-family optimizer residuals:")
    end

    if print_matrices
        println("\nAeq =")
        show(stdout, "text/plain", Aeq)
        println("\nbeq =")
        show(stdout, "text/plain", beq)
        println()
    end

    return nothing
end

function _independent_row_indices(Aeq::Matrix{Float64})
    nrows = size(Aeq, 1)
    idx = Int[]
    if nrows == 0
        return idx
    end
    rank_now = 0
    @inbounds for i in 1:nrows
        cand = isempty(idx) ? Aeq[i:i, :] : Aeq[[idx; i], :]
        rank_cand = rank(cand)
        if rank_cand > rank_now
            push!(idx, i)
            rank_now = rank_cand
        end
    end
    return idx
end

function _verify_divergence(m_diag::Vector{Float64},
                            x::Vector{Float64},
                            G::Matrix{Float64},
                            Bdiag::Vector{Float64},
                            p::Int,
                            enforced_rows_by_degree::Dict{Int, Vector{Int}})
    M = Diagonal(m_diag)
    B = Diagonal(Bdiag)
    D = Diagonal(1.0 ./ m_diag) * (B - transpose(G) * M)

    println("\nDivergence verification")
    all_k = sort!(collect(keys(enforced_rows_by_degree)))
    if isempty(all_k)
        println("  No divergence constraints were enforced.")
        return D
    end

    for k in all_k
        u = x .^ k
        exact = (k + p) .* (x .^ (k - 1))
        err = D * u - exact
        rows = enforced_rows_by_degree[k]
        enforced_err = isempty(rows) ? NaN : maximum(abs.(err[rows]))
        @printf("  k=%d  max|D*x^k - d^(k)| full = %.6e, enforced-rows = %.6e\n",
                k, maximum(abs.(err)), enforced_err)
    end
    return D
end

function run_diag_mass_qp(cfg::DiagMassQPConfig = DiagMassQPConfig())
    cfg.d > 0 || throw(ArgumentError("d must be positive."))
    cfg.p >= 0 || throw(ArgumentError("p must be non-negative."))
    if !isnothing(cfg.epsilon)
        cfg.epsilon > 0 || throw(ArgumentError("epsilon override must be positive."))
    end
    if !isnothing(cfg.quadrature_cap)
        cfg.quadrature_cap >= 0 || throw(ArgumentError("quadrature_cap override must be non-negative."))
    end

    x, R = _canonical_grid(cfg.R)
    N = length(x)
    c = (N + 1) ÷ 2
    lower_bound = isnothing(cfg.epsilon) ? _next_cell_volume_from_origin(x) : cfg.epsilon

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
            println("Boundary accuracy inference failed; using fallback db=3 for d=6.")
        else
            throw(ArgumentError("Failed to infer boundary accuracy from G for d=$(cfg.d)."))
        end
    end

    system = _assemble_constraints(x, G, Bdiag, boundary_count, boundary_accuracy, cfg)
    Aeq = system.Aeq
    beq = system.beq
    family_rows = system.family_rows

    println("Diagonal-M QP infeasibility probe")
    println("  grid: R=$R, N=$N, center_index=$c, dx=1")
    println("  settings: d=$(cfg.d), p=$(cfg.p), anti_stiffness=$(cfg.anti_stiffness)")
    println("  mass lower bound = ", lower_bound, isnothing(cfg.epsilon) ? " (next-cell volume from r=0)" : " (CLI override)")
    println("  quadrature default cap = ", system.quadrature_cap, isnothing(cfg.quadrature_cap) ? " (default d/2)" : " (CLI override)")
    println("  boundary_count=$boundary_count ($boundary_count_source), inferred_boundary_accuracy=$boundary_accuracy")
    println("  quadrature max degree = ", system.quad_max_degree)
    println("  interior divergence max odd degree = ", system.interior_max_odd)
    println("  boundary divergence max odd degree = ", system.boundary_max_odd)
    println("  hard equalities: ", size(Aeq, 1))
    rankA = rank(Aeq)
    rankAug = rank(hcat(Aeq, beq))
    println("  rank(Aeq) = $rankA, rank([Aeq|beq]) = $rankAug")
    if rankAug > rankA
        println("  note: full hard-equality system is algebraically inconsistent (rank mismatch).")
    end

    ind_rows = _independent_row_indices(Aeq)
    Aopt = Aeq[ind_rows, :]
    bopt = beq[ind_rows]
    println("  optimization equalities (independent subset): ", size(Aopt, 1))

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol", cfg.ipopt_tol)
    set_optimizer_attribute(model, "acceptable_tol", cfg.ipopt_acceptable_tol)
    set_optimizer_attribute(model, "constr_viol_tol", cfg.ipopt_constr_viol_tol)
    set_optimizer_attribute(model, "print_level", cfg.ipopt_print_level)

    @variable(model, m[1:N] >= lower_bound)
    target = x .^ 2
    @objective(model, Min, sum((m[i] - target[i])^2 for i in 1:N))
    @inbounds for r in 1:size(Aopt, 1)
        @constraint(model, sum(Aopt[r, j] * m[j] for j in 1:N) == bopt[r])
    end

    optimize!(model)

    term = termination_status(model)
    primal = primal_status(model)
    dual = dual_status(model)
    has_vals = has_values(model)
    m_opt = has_vals ? value.(m) : nothing

    println("\nSolver status")
    println("  termination_status = ", term)
    println("  primal_status      = ", primal)
    println("  dual_status        = ", dual)
    if has_vals
        println("  objective_value    = ", objective_value(model))
        r_opt = Aeq * m_opt - beq
        @printf("  max|Aeq*m-beq|     = %.6e\n", maximum(abs.(r_opt)))
        @printf("  min(m)             = %.6e\n", minimum(m_opt))
    else
        println("  no primal values were returned by optimizer")
    end

    term_ok = term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
    tol_feas = max(1e-9, 10 * cfg.ipopt_constr_viol_tol)
    eq_ok = has_vals ? maximum(abs.(Aeq * m_opt - beq)) <= tol_feas : false
    feasible = term_ok && has_vals && eq_ok

    if feasible
        D = _verify_divergence(m_opt, x, G, Bdiag, cfg.p, system.enforced_rows_by_degree)
        return (
            config = cfg,
            feasible = true,
            termination_status = term,
            primal_status = primal,
            x = x,
            G = G,
            Bdiag = Bdiag,
            Aeq = Aeq,
            beq = beq,
            m = m_opt,
            D = D,
            family_rows = family_rows,
        )
    end

    println("\nVerification skipped due infeasibility or high constraint violation.")
    _print_linear_diagnostics(Aeq, beq, family_rows;
                              m_opt = m_opt,
                              print_matrices = cfg.print_matrices)

    return (
        config = cfg,
        feasible = false,
        termination_status = term,
        primal_status = primal,
        x = x,
        G = G,
        Bdiag = Bdiag,
        Aeq = Aeq,
        beq = beq,
        m = m_opt,
        family_rows = family_rows,
    )
end

function print_cli_help(io::IO = stdout)
    println(io, "Usage: julia scripts/diag_mass_qp/run_diag_mass_qp.jl [options]")
    println(io, "")
    println(io, "Options:")
    println(io, "  --R <float>                      Domain half-width (integer-valued). Default: 10")
    println(io, "  --d <int>                        Gradient order d. Default: 6")
    println(io, "  --p <int>                        Divergence parameter p. Default: 2")
    println(io, "  --quadrature-cap <int>           Default cap for auto quadrature target (default: d/2)")
    println(io, "  --epsilon <float>                Override lower bound (default: next-cell volume from r=0)")
    println(io, "  --disable-anti                   Disable anti-stiffness constraint m[c]=m[c+1] (default)")
    println(io, "  --enable-anti                    Enable anti-stiffness constraint")
    println(io, "  --quad-max-degree <int>          Override quadrature max degree (default: 2d-p-1)")
    println(io, "  --div-max-odd <int>              Override interior divergence max odd degree (default: d-p)")
    println(io, "  --boundary-div-max-odd <int>     Override boundary divergence max odd degree")
    println(io, "  --ipopt-tol <float>              Ipopt tol. Default: 1e-12")
    println(io, "  --ipopt-acceptable-tol <float>   Ipopt acceptable_tol. Default: 1e-12")
    println(io, "  --ipopt-constr-viol-tol <float>  Ipopt constr_viol_tol. Default: 1e-12")
    println(io, "  --ipopt-print-level <int>        Ipopt print_level. Default: 0")
    println(io, "  --print-matrices                 Print full Aeq and beq in diagnostics")
    println(io, "  -h, --help                       Show this help text")
end

function parse_cli_args(args::Vector{String})
    cfg = DiagMassQPConfig()

    R = cfg.R
    d = cfg.d
    p = cfg.p
    quadrature_cap = cfg.quadrature_cap
    epsilon = cfg.epsilon
    anti_stiffness = cfg.anti_stiffness
    quad_max_degree = cfg.quad_max_degree
    div_max_odd = cfg.div_max_odd
    boundary_div_max_odd = cfg.boundary_div_max_odd
    ipopt_tol = cfg.ipopt_tol
    ipopt_acceptable_tol = cfg.ipopt_acceptable_tol
    ipopt_constr_viol_tol = cfg.ipopt_constr_viol_tol
    ipopt_print_level = cfg.ipopt_print_level
    print_matrices = cfg.print_matrices

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "-h" || arg == "--help"
            print_cli_help()
            return nothing
        elseif arg == "--disable-anti"
            anti_stiffness = false
            i += 1
            continue
        elseif arg == "--enable-anti"
            anti_stiffness = true
            i += 1
            continue
        elseif arg == "--print-matrices"
            print_matrices = true
            i += 1
            continue
        end

        i < length(args) || throw(ArgumentError("Missing value for argument $arg."))
        val = args[i + 1]

        if arg == "--R"
            R = parse(Float64, val)
        elseif arg == "--d"
            d = parse(Int, val)
        elseif arg == "--p"
            p = parse(Int, val)
        elseif arg == "--quadrature-cap"
            quadrature_cap = parse(Int, val)
        elseif arg == "--epsilon"
            epsilon = parse(Float64, val)
        elseif arg == "--quad-max-degree"
            quad_max_degree = parse(Int, val)
        elseif arg == "--div-max-odd"
            div_max_odd = parse(Int, val)
        elseif arg == "--boundary-div-max-odd"
            boundary_div_max_odd = parse(Int, val)
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

    return DiagMassQPConfig(
        R = R,
        d = d,
        p = p,
        quadrature_cap = quadrature_cap,
        epsilon = epsilon,
        anti_stiffness = anti_stiffness,
        quad_max_degree = quad_max_degree,
        div_max_odd = div_max_odd,
        boundary_div_max_odd = boundary_div_max_odd,
        ipopt_tol = ipopt_tol,
        ipopt_acceptable_tol = ipopt_acceptable_tol,
        ipopt_constr_viol_tol = ipopt_constr_viol_tol,
        ipopt_print_level = ipopt_print_level,
        print_matrices = print_matrices,
    )
end
