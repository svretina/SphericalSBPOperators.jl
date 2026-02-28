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
    s_quad_order::Union{Nothing, Int} = nothing
    v_quad_order::Union{Nothing, Int} = nothing
    enforce_v_neg1::Bool = false
    staged_quadrature::Bool = false
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
@inline _even_cap(k::Int) = k < 0 ? -2 : (iseven(k) ? k : k - 1)
@inline _even_degrees(k::Int) = k < 0 ? Int[] : collect(0:2:_even_cap(k))

function _integral_rhs(R::Float64, p::Int, q::Int)
    exponent = q + p
    if isodd(exponent)
        return 0.0
    end
    denom = q + p + 1
    denom != 0 || throw(ArgumentError("Integral for q=$q and p=$p is logarithmic/singular on [-R,R]."))
    return 2.0 * R^(q + p + 1) / denom
end

function _moment_vector(x::Vector{Float64}, q::Int)
    if q != -1
        return x .^ q
    end
    invx = zeros(Float64, length(x))
    @inbounds for i in eachindex(x)
        xi = x[i]
        if abs(xi) > 0.0
            invx[i] = 1.0 / xi
        end
    end
    return invx
end

function _resolve_split_quadrature_orders(cfg::DiagMassQPConfig)
    cap_default = fld(cfg.d, 2)
    quadrature_cap = isnothing(cfg.quadrature_cap) ? cap_default : max(0, cfg.quadrature_cap)
    legacy_default = min(2 * cfg.d - cfg.p - 1, quadrature_cap)
    legacy_quad_max = isnothing(cfg.quad_max_degree) ? legacy_default : cfg.quad_max_degree
    legacy_quad_max = max(-1, legacy_quad_max)

    s_default = max(0, _even_cap(legacy_quad_max))
    v_default = legacy_quad_max >= 1 ? _odd_cap(legacy_quad_max) : -1

    s_quad_order = isnothing(cfg.s_quad_order) ? s_default : cfg.s_quad_order
    v_quad_order = isnothing(cfg.v_quad_order) ? v_default : cfg.v_quad_order

    (s_quad_order >= 0 && iseven(s_quad_order)) ||
        throw(ArgumentError("s_quad_order must be an even exponent >= 0 (0,2,4,...); got $s_quad_order."))
    (v_quad_order == -1 || (v_quad_order >= 1 && isodd(v_quad_order))) ||
        throw(ArgumentError("v_quad_order must be -1 or an odd exponent >= 1 (1,3,5,...); got $v_quad_order."))

    return (
        quadrature_cap = quadrature_cap,
        legacy_quad_max = legacy_quad_max,
        s_quad_order = s_quad_order,
        v_quad_order = v_quad_order,
    )
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
    nvar = 2 * N

    quad = _resolve_split_quadrature_orders(cfg)
    s_quad_order = quad.s_quad_order
    v_quad_order = quad.v_quad_order

    interior_default = _odd_cap(cfg.d - cfg.p)
    interior_max_odd = isnothing(cfg.div_max_odd) ? interior_default : _odd_cap(cfg.div_max_odd)

    boundary_default = _odd_cap(boundary_accuracy - cfg.p)
    boundary_max_odd = isnothing(cfg.boundary_div_max_odd) ? boundary_default : _odd_cap(cfg.boundary_div_max_odd)

    rows = Vector{Vector{Float64}}()
    rhs = Float64[]
    family_rows = Dict{Symbol, Vector{Int}}(
        :s_symmetry => Int[],
        :v_symmetry => Int[],
        :anti => Int[],
        :s_quadrature => Int[],
        :v_quadrature => Int[],
        :divergence => Int[],
    )
    enforced_rows_by_degree = Dict{Int, Vector{Int}}()
    divergence_row_meta = NamedTuple{(:row, :i, :k, :boundary_row), Tuple{Int, Int, Int, Bool}}[]
    quadrature_row_meta = NamedTuple{(:row, :family, :q), Tuple{Int, Symbol, Int}}[]

    function add_row!(coeff::Vector{Float64}, b::Float64, fam::Symbol)
        push!(rows, copy(coeff))
        push!(rhs, b)
        row_idx = length(rows)
        push!(family_rows[fam], row_idx)
        return row_idx
    end

    # Symmetry constraints around center for both diagonal masses.
    for k in 1:(c - 1)
        coeff_s = zeros(Float64, nvar)
        coeff_s[c - k] = 1.0
        coeff_s[c + k] = -1.0
        add_row!(coeff_s, 0.0, :s_symmetry)

        coeff_v = zeros(Float64, nvar)
        coeff_v[N + c - k] = 1.0
        coeff_v[N + c + k] = -1.0
        add_row!(coeff_v, 0.0, :v_symmetry)
    end

    # Anti-stiffness constraint.
    if cfg.anti_stiffness
        coeff_s = zeros(Float64, nvar)
        coeff_s[c] = 1.0
        coeff_s[c + 1] = -1.0
        add_row!(coeff_s, 0.0, :anti)

        coeff_v = zeros(Float64, nvar)
        coeff_v[N + c] = 1.0
        coeff_v[N + c + 1] = -1.0
        add_row!(coeff_v, 0.0, :anti)
    end

    # Scalar quadrature constraints: S integrates even monomials.
    for q in _even_degrees(s_quad_order)
        coeff = zeros(Float64, nvar)
        coeff[1:N] .= _moment_vector(x, q)
        b = _integral_rhs(x[end], cfg.p, q)
        row_idx = add_row!(coeff, b, :s_quadrature)
        push!(quadrature_row_meta, (row = row_idx, family = :S, q = q))
    end

    # Vector quadrature constraints: V integrates odd monomials, with optional q = -1.
    if cfg.enforce_v_neg1
        coeff = zeros(Float64, nvar)
        coeff[(N + 1):(2 * N)] .= _moment_vector(x, -1)
        b = _integral_rhs(x[end], cfg.p, -1)
        row_idx = add_row!(coeff, b, :v_quadrature)
        push!(quadrature_row_meta, (row = row_idx, family = :V, q = -1))
    end
    for q in _odd_degrees(v_quad_order)
        coeff = zeros(Float64, nvar)
        coeff[(N + 1):(2 * N)] .= _moment_vector(x, q)
        b = _integral_rhs(x[end], cfg.p, q)
        row_idx = add_row!(coeff, b, :v_quadrature)
        push!(quadrature_row_meta, (row = row_idx, family = :V, q = q))
    end

    # Divergence constraints from S*D + GᵀV = B on odd monomials.
    @inbounds for i in 1:N
        boundary_row = i <= boundary_count || i > (N - boundary_count)
        local_max_odd = boundary_row ? boundary_max_odd : interior_max_odd
        for k in _odd_degrees(local_max_odd)
            xk = x .^ k
            coeff = zeros(Float64, nvar)
            coeff[i] = (k + cfg.p) * x[i]^(k - 1)
            for j in 1:N
                coeff[N + j] = G[j, i] * xk[j]
            end
            b = Bdiag[i] * xk[i]
            row_idx = add_row!(coeff, b, :divergence)
            push!(divergence_row_meta, (row = row_idx, i = i, k = k, boundary_row = boundary_row))

            if !haskey(enforced_rows_by_degree, k)
                enforced_rows_by_degree[k] = Int[]
            end
            push!(enforced_rows_by_degree[k], i)
        end
    end

    n_eq = length(rows)
    Aeq = zeros(Float64, n_eq, nvar)
    @inbounds for r in 1:n_eq
        Aeq[r, :] .= rows[r]
    end
    beq = copy(rhs)

    return (
        Aeq = Aeq,
        beq = beq,
        family_rows = family_rows,
        enforced_rows_by_degree = enforced_rows_by_degree,
        divergence_row_meta = divergence_row_meta,
        quadrature_row_meta = quadrature_row_meta,
        quadrature_cap = quad.quadrature_cap,
        legacy_quad_max = quad.legacy_quad_max,
        s_quad_order = s_quad_order,
        v_quad_order = v_quad_order,
        enforce_v_neg1 = cfg.enforce_v_neg1,
        interior_max_odd = interior_max_odd,
        boundary_max_odd = boundary_max_odd,
    )
end

function _print_family_residuals(residual::Vector{Float64},
                                 family_rows::Dict{Symbol, Vector{Int}};
                                 label::String = "")
    isempty(label) || println(label)
    for fam in (:s_symmetry, :v_symmetry, :anti, :s_quadrature, :v_quadrature, :divergence)
        idx = get(family_rows, fam, Int[])
        isempty(idx) && continue
        r = residual[idx]
        @printf("  %-12s rows=%3d  ||r||2=%12.4e  ||r||∞=%12.4e\n",
                string(fam), length(idx), norm(r), maximum(abs.(r)))
    end
end

function _print_linear_diagnostics(Aeq::Matrix{Float64},
                                   beq::Vector{Float64},
                                   family_rows::Dict{Symbol, Vector{Int}};
                                   z_opt::Union{Nothing, Vector{Float64}} = nothing,
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

    z_ls = Aeq \ beq
    r_ls = Aeq * z_ls - beq
    @printf("  LS residual: ||Aeq*z_ls-beq||2 = %.6e, ||.||∞ = %.6e\n",
            norm(r_ls), maximum(abs.(r_ls)))
    _print_family_residuals(r_ls, family_rows; label = "  Per-family LS residuals:")

    if z_opt !== nothing
        r_opt = Aeq * z_opt - beq
        @printf("  Optimizer-point residual: ||Aeq*z-beq||2 = %.6e, ||.||∞ = %.6e\n",
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

function _summarize_divergence_constraint_residual(residual::Vector{Float64},
                                                   divergence_row_meta::Vector{NamedTuple{(:row, :i, :k, :boundary_row), Tuple{Int, Int, Int, Bool}}},
                                                   center_index::Int)
    isempty(divergence_row_meta) && return (
        max_all = NaN,
        row_all = 0,
        i_all = 0,
        k_all = 0,
        source_is_boundary = false,
        source_is_origin = false,
        max_origin = NaN,
        row_origin = 0,
        k_origin = 0,
        max_boundary = NaN,
        row_boundary = 0,
        i_boundary = 0,
        k_boundary = 0,
    )

    max_all = -1.0
    row_all = 0
    i_all = 0
    k_all = 0
    max_origin = -1.0
    row_origin = 0
    k_origin = 0
    max_boundary = -1.0
    row_boundary = 0
    i_boundary = 0
    k_boundary = 0

    for meta in divergence_row_meta
        r = meta.row
        v = abs(residual[r])
        if v > max_all
            max_all = v
            row_all = r
            i_all = meta.i
            k_all = meta.k
        end
        if meta.i == center_index && v > max_origin
            max_origin = v
            row_origin = r
            k_origin = meta.k
        end
        if meta.boundary_row && v > max_boundary
            max_boundary = v
            row_boundary = r
            i_boundary = meta.i
            k_boundary = meta.k
        end
    end

    return (
        max_all = max_all,
        row_all = row_all,
        i_all = i_all,
        k_all = k_all,
        source_is_boundary = row_all == 0 ? false : (i_all != center_index && (row_boundary == row_all)),
        source_is_origin = row_all == 0 ? false : (i_all == center_index),
        max_origin = row_origin == 0 ? NaN : max_origin,
        row_origin = row_origin,
        k_origin = row_origin == 0 ? 0 : k_origin,
        max_boundary = row_boundary == 0 ? NaN : max_boundary,
        row_boundary = row_boundary,
        i_boundary = i_boundary,
        k_boundary = row_boundary == 0 ? 0 : k_boundary,
    )
end

function _solve_with_active_rows(Aeq::Matrix{Float64},
                                 beq::Vector{Float64},
                                 active_rows::Vector{Int},
                                 lower_bound::Float64,
                                 target::Vector{Float64},
                                 cfg::DiagMassQPConfig)
    rows = sort!(unique(copy(active_rows)))
    isempty(rows) && throw(ArgumentError("active_rows must be non-empty."))

    Aactive = Aeq[rows, :]
    bactive = beq[rows]
    ind_rows = _independent_row_indices(Aactive)
    Aopt = Aactive[ind_rows, :]
    bopt = bactive[ind_rows]

    nvar = size(Aeq, 2)
    N = nvar ÷ 2

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "tol", cfg.ipopt_tol)
    set_optimizer_attribute(model, "acceptable_tol", cfg.ipopt_acceptable_tol)
    set_optimizer_attribute(model, "constr_viol_tol", cfg.ipopt_constr_viol_tol)
    set_optimizer_attribute(model, "print_level", cfg.ipopt_print_level)

    @variable(model, z[1:nvar] >= lower_bound)
    @objective(model, Min,
               sum((z[i] - target[i])^2 for i in 1:N) +
               sum((z[N + i] - target[i])^2 for i in 1:N))
    @inbounds for r in 1:size(Aopt, 1)
        @constraint(model, sum(Aopt[r, j] * z[j] for j in 1:nvar) == bopt[r])
    end

    optimize!(model)

    term = termination_status(model)
    primal = primal_status(model)
    dual = dual_status(model)
    has_vals = has_values(model)
    z_opt = has_vals ? value.(z) : nothing
    r_active = has_vals ? (Aactive * z_opt - bactive) : nothing
    max_active = has_vals ? maximum(abs.(r_active)) : Inf
    tol_feas = max(1e-9, 10 * cfg.ipopt_constr_viol_tol)
    term_ok = term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
    feasible = term_ok && has_vals && max_active <= tol_feas

    return (
        active_rows = rows,
        Aactive = Aactive,
        bactive = bactive,
        Aopt = Aopt,
        bopt = bopt,
        term = term,
        primal = primal,
        dual = dual,
        has_vals = has_vals,
        z_opt = z_opt,
        r_active = r_active,
        max_active = max_active,
        objective = has_vals ? objective_value(model) : NaN,
        feasible = feasible,
        tol_feas = tol_feas,
    )
end

function _quadrature_q_lists_from_rows(quadrature_row_meta::Vector{NamedTuple{(:row, :family, :q), Tuple{Int, Symbol, Int}}},
                                       active_rows::Vector{Int})
    active_set = Set(active_rows)
    s_qs = Int[]
    v_qs = Int[]
    for meta in quadrature_row_meta
        meta.row in active_set || continue
        if meta.family == :S
            push!(s_qs, meta.q)
        else
            push!(v_qs, meta.q)
        end
    end
    return sort!(unique(s_qs)), sort!(unique(v_qs))
end

function _staged_active_rows(system,
                             Aeq::Matrix{Float64},
                             beq::Vector{Float64},
                             lower_bound::Float64,
                             target::Vector{Float64},
                             cfg::DiagMassQPConfig)
    family_rows = system.family_rows
    base_rows = sort!(unique(vcat(get(family_rows, :s_symmetry, Int[]),
                                  get(family_rows, :v_symmetry, Int[]),
                                  get(family_rows, :anti, Int[]),
                                  get(family_rows, :divergence, Int[]))))
    isempty(base_rows) && throw(ArgumentError("No base constraints were assembled for staged mode."))

    base_sol = _solve_with_active_rows(Aeq, beq, base_rows, lower_bound, target, cfg)
    steps = NamedTuple[]
    push!(steps, (stage = :base,
                  family = :base,
                  q = 0,
                  row = 0,
                  accepted = base_sol.feasible,
                  term = base_sol.term,
                  max_active = base_sol.max_active))

    current_rows = copy(base_rows)
    current_sol = base_sol

    quad_meta = sort(copy(system.quadrature_row_meta); by = m -> (m.q, m.family == :S ? 0 : 1))
    if base_sol.feasible
        for meta in quad_meta
            candidate_rows = sort!(unique(vcat(current_rows, [meta.row])))
            cand_sol = _solve_with_active_rows(Aeq, beq, candidate_rows, lower_bound, target, cfg)
            accepted = cand_sol.feasible
            push!(steps, (stage = :quadrature,
                          family = meta.family,
                          q = meta.q,
                          row = meta.row,
                          accepted = accepted,
                          term = cand_sol.term,
                          max_active = cand_sol.max_active))
            if accepted
                current_rows = candidate_rows
                current_sol = cand_sol
            end
        end
    end

    return (active_rows = current_rows, solution = current_sol, steps = steps, base_rows = base_rows)
end

function _verify_split_quadrature(s_diag::Vector{Float64},
                                  v_diag::Vector{Float64},
                                  x::Vector{Float64},
                                  p::Int,
                                  s_qs::Vector{Int},
                                  v_qs::Vector{Int})
    println("\nQuadrature verification")
    sq_err = Dict{Int, Float64}()
    for q in sort!(unique(copy(s_qs)))
        lhs = dot(s_diag, _moment_vector(x, q))
        rhs = _integral_rhs(x[end], p, q)
        sq_err[q] = abs(lhs - rhs)
        @printf("  S q=%2d  |Q-I| = %.6e\n", q, sq_err[q])
    end

    vq_err = Dict{Int, Float64}()
    for q in sort!(unique(copy(v_qs)))
        lhs = dot(v_diag, _moment_vector(x, q))
        rhs = _integral_rhs(x[end], p, q)
        vq_err[q] = abs(lhs - rhs)
        @printf("  V q=%2d  |Q-I| = %.6e\n", q, vq_err[q])
    end

    max_sq = isempty(sq_err) ? NaN : maximum(values(sq_err))
    max_vq = isempty(vq_err) ? NaN : maximum(values(vq_err))
    @printf("  max S quadrature error = %.6e\n", max_sq)
    @printf("  max V quadrature error = %.6e\n", max_vq)

    return (s_errors = sq_err, v_errors = vq_err, max_s_error = max_sq, max_v_error = max_vq)
end

function _verify_divergence(s_diag::Vector{Float64},
                            v_diag::Vector{Float64},
                            x::Vector{Float64},
                            G::Matrix{Float64},
                            Bdiag::Vector{Float64},
                            p::Int,
                            enforced_rows_by_degree::Dict{Int, Vector{Int}})
    V = Diagonal(v_diag)
    B = Diagonal(Bdiag)
    D = Diagonal(1.0 ./ s_diag) * (B - transpose(G) * V)

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
    if !isnothing(cfg.s_quad_order)
        (cfg.s_quad_order >= 0 && iseven(cfg.s_quad_order)) ||
            throw(ArgumentError("s_quad_order must be an even exponent >= 0 (0,2,4,...)."))
    end
    if !isnothing(cfg.v_quad_order)
        (cfg.v_quad_order == -1 || (cfg.v_quad_order >= 1 && isodd(cfg.v_quad_order))) ||
            throw(ArgumentError("v_quad_order must be -1 or an odd exponent >= 1 (1,3,5,...)."))
    end
    if cfg.enforce_v_neg1 && cfg.p == 0
        throw(ArgumentError("q=-1 V quadrature moment is singular for p=0."))
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

    println("Diagonal split-mass QP infeasibility probe")
    println("  grid: R=$R, N=$N, center_index=$c, dx=1")
    println("  settings: d=$(cfg.d), p=$(cfg.p), anti_stiffness=$(cfg.anti_stiffness), staged_quadrature=$(cfg.staged_quadrature)")
    println("  hard positivity lower bound on S and V = ", lower_bound,
            isnothing(cfg.epsilon) ? " (next-cell volume from r=0)" : " (CLI override)")
    println("  quadrature default cap = ", system.quadrature_cap, isnothing(cfg.quadrature_cap) ? " (default d/2)" : " (CLI override)")
    println("  boundary_count=$boundary_count ($boundary_count_source), inferred_boundary_accuracy=$boundary_accuracy")
    println("  legacy quadrature max degree = ", system.legacy_quad_max)
    println("  S quadrature max exponent q (even) = ", system.s_quad_order)
    println("  V quadrature max exponent q (odd, or -1 for none) = ", system.v_quad_order)
    println("  enforce V q=-1 moment = ", system.enforce_v_neg1)
    println("  interior divergence max odd degree = ", system.interior_max_odd)
    println("  boundary divergence max odd degree = ", system.boundary_max_odd)
    println("  hard equalities: ", size(Aeq, 1))
    rankA = rank(Aeq)
    rankAug = rank(hcat(Aeq, beq))
    println("  rank(Aeq) = $rankA, rank([Aeq|beq]) = $rankAug")
    if rankAug > rankA
        println("  note: full hard-equality system is algebraically inconsistent (rank mismatch).")
    end

    target = abs.(x) .^ cfg.p
    solve_report = nothing
    active_rows = Int[]
    staged_report = nothing
    if cfg.staged_quadrature
        staged_report = _staged_active_rows(system, Aeq, beq, lower_bound, target, cfg)
        solve_report = staged_report.solution
        active_rows = staged_report.active_rows
        println("  staged_quadrature = true")
        println("  staged base rows = ", length(staged_report.base_rows))
        println("  staged final active rows = ", length(active_rows), " / ", size(Aeq, 1))
        accepted = [s for s in staged_report.steps if s.stage == :quadrature && s.accepted]
        rejected = [s for s in staged_report.steps if s.stage == :quadrature && !s.accepted]
        println("  staged accepted quadrature rows = ", length(accepted))
        println("  staged rejected quadrature rows = ", length(rejected))
        if !staged_report.steps[1].accepted
            println("  base divergence/symmetry solve is infeasible; quadrature rows were not attempted.")
        end
        if !isempty(accepted)
            println("  accepted q constraints:")
            for s in accepted
                @printf("    %s q=%d (row=%d)\n", string(s.family), s.q, s.row)
            end
        end
        if !isempty(rejected)
            println("  rejected q constraints:")
            for s in rejected
                @printf("    %s q=%d (row=%d), term=%s, max_active=%.6e\n",
                        string(s.family), s.q, s.row, string(s.term), s.max_active)
            end
        end
    else
        active_rows = collect(1:size(Aeq, 1))
        solve_report = _solve_with_active_rows(Aeq, beq, active_rows, lower_bound, target, cfg)
    end
    println("  optimization equalities (independent subset): ", size(solve_report.Aopt, 1))

    term = solve_report.term
    primal = solve_report.primal
    dual = solve_report.dual
    has_vals = solve_report.has_vals
    z_opt = solve_report.z_opt
    s_opt = has_vals ? copy(z_opt[1:N]) : nothing
    v_opt = has_vals ? copy(z_opt[(N + 1):(2 * N)]) : nothing
    r_full = has_vals ? (Aeq * z_opt - beq) : nothing
    div_constraint_report = has_vals ? _summarize_divergence_constraint_residual(r_full, system.divergence_row_meta, c) : nothing

    println("\nSolver status")
    println("  termination_status = ", term)
    println("  primal_status      = ", primal)
    println("  dual_status        = ", dual)
    if has_vals
        println("  objective_value    = ", solve_report.objective)
        @printf("  max|A_active*z-b_active| = %.6e\n", solve_report.max_active)
        @printf("  max|Aeq*z-beq| (full)    = %.6e\n", maximum(abs.(r_full)))
        @printf("  min(S)             = %.6e\n", minimum(s_opt))
        @printf("  min(V)             = %.6e\n", minimum(v_opt))
        @printf("  max|div-constraint residual| = %.6e (row=%d, i=%d, k=%d)\n",
                div_constraint_report.max_all, div_constraint_report.row_all, div_constraint_report.i_all, div_constraint_report.k_all)
        if isnan(div_constraint_report.max_origin)
            println("  origin div residual (r=0) = n/a (no origin divergence row was constrained)")
        else
            @printf("  origin div residual (r=0) = %.6e (row=%d, k=%d)\n",
                    div_constraint_report.max_origin, div_constraint_report.row_origin, div_constraint_report.k_origin)
        end
        if isnan(div_constraint_report.max_boundary)
            println("  boundary div residual = n/a (no boundary divergence row was constrained)")
        else
            @printf("  boundary div residual = %.6e (row=%d, i=%d, k=%d)\n",
                    div_constraint_report.max_boundary,
                    div_constraint_report.row_boundary,
                    div_constraint_report.i_boundary,
                    div_constraint_report.k_boundary)
        end
    else
        println("  no primal values were returned by optimizer")
    end

    feasible = solve_report.feasible

    if feasible
        s_qs = Int[]
        v_qs = Int[]
        if cfg.staged_quadrature
            s_qs, v_qs = _quadrature_q_lists_from_rows(system.quadrature_row_meta, active_rows)
        else
            s_qs = _even_degrees(system.s_quad_order)
            v_qs = vcat(system.enforce_v_neg1 ? Int[-1] : Int[], _odd_degrees(system.v_quad_order))
            v_qs = sort!(unique(v_qs))
        end
        quad_report = _verify_split_quadrature(s_opt,
                                               v_opt,
                                               x,
                                               cfg.p,
                                               s_qs,
                                               v_qs)
        D = _verify_divergence(s_opt, v_opt, x, G, Bdiag, cfg.p, system.enforced_rows_by_degree)
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
            S = s_opt,
            V = v_opt,
            D = D,
            quadrature_report = quad_report,
            div_constraint_report = div_constraint_report,
            active_rows = active_rows,
            staged_report = staged_report,
            family_rows = family_rows,
        )
    end

    println("\nVerification skipped due infeasibility or high constraint violation.")
    _print_linear_diagnostics(Aeq, beq, family_rows;
                              z_opt = z_opt,
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
        z = z_opt,
        S = s_opt,
        V = v_opt,
        div_constraint_report = div_constraint_report,
        active_rows = active_rows,
        staged_report = staged_report,
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
    println(io, "  --quad-max-degree <int>          Legacy split-default driver (for backward compatibility)")
    println(io, "  --s-quad-order <int>             Max even exponent q for S moments (0,2,4,...)")
    println(io, "  --v-quad-order <int>             Max odd exponent q for V moments (1,3,5,...) or -1 for none")
    println(io, "  --enforce-v-neg1                 Also enforce V q=-1 moment (default: off)")
    println(io, "  --skip-v-neg1                    Disable V q=-1 moment")
    println(io, "  --staged-quadrature              Enforce divergence first, then add quadrature rows greedily")
    println(io, "  --no-staged-quadrature           Disable staged quadrature mode (default)")
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
    s_quad_order = cfg.s_quad_order
    v_quad_order = cfg.v_quad_order
    enforce_v_neg1 = cfg.enforce_v_neg1
    staged_quadrature = cfg.staged_quadrature
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
        elseif arg == "--enforce-v-neg1"
            enforce_v_neg1 = true
            i += 1
            continue
        elseif arg == "--skip-v-neg1"
            enforce_v_neg1 = false
            i += 1
            continue
        elseif arg == "--staged-quadrature"
            staged_quadrature = true
            i += 1
            continue
        elseif arg == "--no-staged-quadrature"
            staged_quadrature = false
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
        elseif arg == "--s-quad-order"
            s_quad_order = parse(Int, val)
        elseif arg == "--v-quad-order"
            v_quad_order = parse(Int, val)
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
        s_quad_order = s_quad_order,
        v_quad_order = v_quad_order,
        enforce_v_neg1 = enforce_v_neg1,
        staged_quadrature = staged_quadrature,
        div_max_odd = div_max_odd,
        boundary_div_max_odd = boundary_div_max_odd,
        ipopt_tol = ipopt_tol,
        ipopt_acceptable_tol = ipopt_acceptable_tol,
        ipopt_constr_viol_tol = ipopt_constr_viol_tol,
        ipopt_print_level = ipopt_print_level,
        print_matrices = print_matrices,
    )
end
