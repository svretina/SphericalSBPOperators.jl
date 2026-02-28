using LinearAlgebra: Diagonal
using Printf: @printf
using SummationByPartsOperators: MattssonNordström2004, derivative_operator, grid, mass_matrix

const RatBig = Rational{BigInt}

Base.@kwdef struct LHopitalGradientRepairConfig
    accuracy_order::Int
    Nhalf::Int
    spatial_dim::Int = 3
    source = MattssonNordström2004()
    mode = nothing
    w1_mode::Symbol = :from_j2_raw
    w1_user::Union{Nothing, RatBig} = nothing
    stencil_mode::Symbol = :raw_pattern_expand_right
    custom_stencil_cols::Union{Nothing, Vector{Int}} = nothing
end

@inline _rat(x::Integer) = big(x) // 1
@inline _rat(x::RatBig) = x
@inline function _rat(x::Rational)
    return big(numerator(x)) // big(denominator(x))
end
@inline function _rat(x::AbstractFloat)
    return rationalize(BigInt, x)
end

function _to_ratvec(v)
    out = Vector{RatBig}(undef, length(v))
    @inbounds for i in eachindex(v)
        out[i] = _rat(v[i])
    end
    return out
end

function _to_ratmat(A)
    n, m = size(A)
    out = Matrix{RatBig}(undef, n, m)
    @inbounds for i in 1:n, j in 1:m
        out[i, j] = _rat(A[i, j])
    end
    return out
end

function _ratstr(x)
    xr = _rat(x)
    return string(numerator(xr), "//", denominator(xr))
end

function _dense_derivative_matrix(Dfull, M::Int)
    maybe = try
        Matrix(Dfull)
    catch
        nothing
    end
    if maybe !== nothing && size(maybe) == (M, M)
        return maybe
    end

    T = eltype(Dfull)
    G = Matrix{T}(undef, M, M)
    e = fill(zero(T), M)
    @inbounds for j in 1:M
        e[j] = one(T)
        G[:, j] = Dfull * e
        e[j] = zero(T)
    end
    return G
end

function _build_cartesian_canonical(cfg::LHopitalGradientRepairConfig)
    cfg.accuracy_order > 0 || throw(ArgumentError("`accuracy_order` must be positive."))
    cfg.Nhalf > 1 || throw(ArgumentError("`Nhalf` must be >= 2 (number of points on [0,R])."))
    cfg.spatial_dim > 0 || throw(ArgumentError("`spatial_dim` must be positive."))

    # User-facing convention: Nhalf is the number of points on [0,R].
    # Canonical internal grid has spacing 1 and index range 0:(Nhalf-1).
    Nhalf_index = cfg.Nhalf - 1
    R = big(Nhalf_index) // 1
    Nfull = 2 * cfg.Nhalf - 1

    kwargs = (
              derivative_order = 1,
              accuracy_order = cfg.accuracy_order,
              xmin = -R,
              xmax = R,
              N = Nfull,
             )
    Dfull = cfg.mode === nothing ?
            derivative_operator(cfg.source; kwargs...) :
            derivative_operator(cfg.source; kwargs..., mode = cfg.mode)

    xfull_raw = collect(grid(Dfull))
    M = length(xfull_raw)
    Dcart_raw = _dense_derivative_matrix(Dfull, M)
    Mcart_raw = Matrix(mass_matrix(Dfull))

    xfull = _to_ratvec(xfull_raw)
    Dcart = _to_ratmat(Dcart_raw)
    Mcart = _to_ratmat(Mcart_raw)

    q_infer = if hasproperty(Dfull, :coefficients) && hasproperty(getproperty(Dfull, :coefficients), :accuracy_order)
        Int(getproperty(getproperty(Dfull, :coefficients), :accuracy_order))
    else
        throw(ArgumentError("Could not infer operator accuracy from `coefficients.accuracy_order`."))
    end
    q_infer == cfg.accuracy_order ||
        throw(ArgumentError("Inferred accuracy_order=$q_infer does not match requested $(cfg.accuracy_order)."))

    return (
            Dfull = Dfull,
            xfull = xfull,
            Dcart = Dcart,
            Mcart = Mcart,
            q_infer = q_infer,
            Nhalf_index = Nhalf_index,
           )
end

function _build_folding_maps(xfull::Vector{RatBig})
    M = length(xfull)
    half_indices = [i for i in eachindex(xfull) if xfull[i] >= 0 // 1]
    sort!(half_indices; by = i -> xfull[i])
    isempty(half_indices) && throw(ArgumentError("No nonnegative half-grid nodes found."))

    r = xfull[half_indices]
    r[1] == 0 // 1 || throw(ArgumentError("First half-grid node must be zero."))

    Nh = length(r)
    Rop = zeros(RatBig, Nh, M)
    @inbounds for (i, idx) in enumerate(half_indices)
        Rop[i, idx] = 1 // 1
    end

    Eeven = zeros(RatBig, M, Nh)
    Eodd = zeros(RatBig, M, Nh)

    value_to_half_index = Dict{RatBig, Int}()
    @inbounds for j in 1:Nh
        value_to_half_index[r[j]] = j
    end

    @inbounds for i in 1:M
        absx = abs(xfull[i])
        haskey(value_to_half_index, absx) ||
            throw(ArgumentError("Could not fold point |x|=$absx onto the half-grid."))
        j = value_to_half_index[absx]
        Eeven[i, j] = 1 // 1
        if xfull[i] > 0 // 1
            Eodd[i, j] = 1 // 1
        elseif xfull[i] < 0 // 1
            Eodd[i, j] = -1 // 1
        end
    end

    return (r = r, Rop = Rop, Eeven = Eeven, Eodd = Eodd)
end

function _solve_exact_linear_system(A::Matrix{RatBig}, b::Vector{RatBig})
    n_eq, n_vars = size(A)
    n_eq == length(b) || throw(DimensionMismatch("Incompatible linear system dimensions."))
    n_eq == 0 && return zeros(RatBig, n_vars)

    M = hcat(copy(A), copy(b))
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

    for r in 1:n_eq
        all_zero = true
        for c in 1:n_vars
            if M[r, c] != 0 // 1
                all_zero = false
                break
            end
        end
        if all_zero && M[r, n_vars + 1] != 0 // 1
            throw(ArgumentError("No exact solution exists for this linear system."))
        end
    end

    x = zeros(RatBig, n_vars)
    for (r, col) in enumerate(pivot_cols)
        x[col] = M[r, n_vars + 1]
    end
    return x
end

@inline function _is_effective_nonzero(v::RatBig)
    return v != 0 // 1
end

@inline function _is_effective_nonzero(v::AbstractFloat; atol::Float64 = 1e-14)
    return abs(v) > max(atol, 1024 * eps(typeof(v)))
end

function _rows_with_nonzero_first_column_dense(G::AbstractMatrix{T}) where {T <: Real}
    n = size(G, 1)
    rows = Int[]
    @inbounds for i in 2:n
        _is_effective_nonzero(G[i, 1]) && push!(rows, i)
    end
    return rows
end

function _right_closure_width_from_operator(Dfull)
    hasproperty(Dfull, :coefficients) ||
        throw(ArgumentError("Could not infer closure width: operator has no `coefficients` field."))
    coeffs = getproperty(Dfull, :coefficients)
    if hasproperty(coeffs, :right_weights)
        return length(getproperty(coeffs, :right_weights))
    elseif hasproperty(coeffs, :left_weights)
        return length(getproperty(coeffs, :left_weights))
    end
    throw(ArgumentError("Could not infer closure width: coefficients expose neither `right_weights` nor `left_weights`."))
end

function _validate_fixed_stencil(stencil_cols::Vector{Int}, Nh::Int, num_constraints::Int)
    isempty(stencil_cols) && throw(ArgumentError("`custom_stencil_cols` must be non-empty."))
    1 in stencil_cols && throw(ArgumentError("`custom_stencil_cols` must not contain 1 (origin column)."))
    any(c -> c < 2 || c > Nh, stencil_cols) &&
        throw(ArgumentError("`custom_stencil_cols` entries must lie in 2:$Nh."))
    length(unique(stencil_cols)) == length(stencil_cols) ||
        throw(ArgumentError("`custom_stencil_cols` must not contain duplicate indices."))
    length(stencil_cols) >= num_constraints ||
        throw(ArgumentError("Stencil width $(length(stencil_cols)) is too narrow for $num_constraints constraints."))
    return sort(stencil_cols)
end

function _solve_repaired_row_weights(row::Int,
                                     Gcol1_new::RatBig,
                                     r::Vector{RatBig},
                                     q_even::Vector{Int},
                                     initial_cols::Vector{Int},
                                     Nh::Int,
                                     mode::Symbol)
    num_constraints = length(q_even)
    cols = sort(unique(initial_cols))
    isempty(cols) && (cols = Int[2])

    cols = [c for c in cols if 2 <= c <= Nh]
    isempty(cols) && (cols = Int[2])

    max_extra_points = 2
    max_allowed_cols = min((Nh - 1), num_constraints + max_extra_points)

    if mode == :fixed_cols
        cols = _validate_fixed_stencil(cols, Nh, num_constraints)
    elseif mode == :raw_pattern_expand_right
        # Start with stencil size equal to the number of constraints.
        if length(cols) > num_constraints
            cols = cols[1:num_constraints]
        end
        while length(cols) < num_constraints
            nxt = isempty(cols) ? 2 : (maximum(cols) + 1)
            nxt <= Nh || throw(ArgumentError("Cannot satisfy constraints for row $row: insufficient half-grid columns."))
            push!(cols, nxt)
        end
    else
        throw(ArgumentError("Unsupported stencil mode `$mode`."))
    end

    base_cols = copy(cols)
    while true
        n_eq = num_constraints
        n_vars = length(cols)
        A = zeros(RatBig, n_eq, n_vars)
        b = zeros(RatBig, n_eq)
        rj = r[row]

        @inbounds for eq in 1:n_eq
            q = q_even[eq]
            target = if q == 0
                -Gcol1_new
            else
                (big(q) // 1) * (rj^(q - 1))
            end

            b[eq] = target
            for (v, col) in enumerate(cols)
                A[eq, v] = r[col]^q
            end
        end

        try
            x = _solve_exact_linear_system(A, b)
            return (
                    weights = x,
                    stencil_cols = copy(cols),
                    expanded = (mode == :raw_pattern_expand_right && length(cols) > length(base_cols)),
                    extra_points = max(0, length(cols) - num_constraints),
                   )
        catch err
            if !(mode == :raw_pattern_expand_right && err isa ArgumentError && occursin("No exact solution exists", sprint(showerror, err)))
                rethrow(err)
            end
            length(cols) < max_allowed_cols ||
                throw(ArgumentError("No exact repaired row solve for row $row with stencil width in [$num_constraints, $(max_allowed_cols)]."))
            nxt = maximum(cols) + 1
            nxt <= Nh ||
                throw(ArgumentError("No exact repaired row solve for row $row up to column $Nh (started from stencil $(initial_cols))."))
            push!(cols, nxt)
        end
    end
end

function _build_row_stencil_from_raw(raw_cols::Vector{Int}, len_target::Int, Nh::Int)
    len_target > 0 || throw(ArgumentError("Target stencil length must be positive."))
    cols = sort(unique([c for c in raw_cols if 2 <= c <= Nh]))
    isempty(cols) && (cols = Int[2])

    if length(cols) > len_target
        cols = cols[1:len_target]
    end
    while length(cols) < len_target
        nxt = maximum(cols) + 1
        nxt <= Nh || throw(ArgumentError("Cannot build row stencil of length $len_target within columns 2:$Nh."))
        push!(cols, nxt)
    end

    return cols
end

function _solve_coupled_gradient_divergence!(G_new::Matrix{RatBig},
                                             rows::Vector{Int},
                                             r::Vector{RatBig},
                                             Mdiag::Vector{RatBig},
                                             Bdiag::Vector{RatBig},
                                             p::Int,
                                             q_even::Vector{Int},
                                             q_odd::Vector{Int},
                                             cfg::LHopitalGradientRepairConfig)
    isempty(rows) && return (
                            row_stencils = Dict{Int, Vector{Int}}(),
                            divergence_rows_q1 = Int[],
                            extra_points = 0,
                            expanded = false,
                           )

    Nh = length(r)
    n_even = length(q_even)
    n_odd = length(q_odd)
    target_len = n_even + n_odd
    target_len > 0 || throw(ArgumentError("No constraints provided for coupled solve."))
    target_len <= Nh - 1 || throw(ArgumentError("Constraint count $target_len exceeds available half-grid columns $(Nh - 1)."))

    row_set = Set(rows)

    function _solve_for_stencil(stencil_cols::Vector{Int})
        stencil_set = Set(stencil_cols)
        all(i -> i in stencil_set, rows) ||
            throw(ArgumentError("Stencil must include all repaired rows for coupled divergence constraints."))

        has_q1 = 1 in q_odd
        odd_high = [q for q in q_odd if q != 1]
        divergence_rows_q1 = has_q1 ? copy(rows) : Int[]

        n_unknowns = length(rows) * length(stencil_cols)
        n_eq = length(rows) * n_even + (has_q1 ? length(divergence_rows_q1) : 0) + length(rows) * length(odd_high)
        A = zeros(RatBig, n_eq, n_unknowns)
        b = zeros(RatBig, n_eq)
        var_index = Dict{Tuple{Int, Int}, Int}()

        idx = 1
        for row in rows
            for col in stencil_cols
                var_index[(row, col)] = idx
                idx += 1
            end
        end

        eq = 1

        # Gradient constraints: keep non-stencil entries fixed.
        for row in rows
            for q in q_even
                rhs = q == 0 ? 0 // 1 : (big(q) // 1) * (r[row]^(q - 1))

                fixed = 0 // 1
                for col in 1:Nh
                    col in stencil_set && continue
                    fixed += G_new[row, col] * (r[col]^q)
                end

                b[eq] = rhs - fixed
                for col in stencil_cols
                    A[eq, var_index[(row, col)]] = r[col]^q
                end
                eq += 1
            end
        end

        # Divergence constraints:
        # - q=1 on solved rows.
        # - higher odd moments on solved rows only.
        for q in q_odd
            target_rows = q == 1 ? divergence_rows_q1 : rows
            for i in target_rows
                exact = (big(q + p) // 1) * (r[i]^(q - 1))
                fixed_numer = Bdiag[i] * (r[i]^q)

                for j in 1:Nh
                    if (j in row_set) && (i in stencil_set)
                        continue
                    end
                    fixed_numer -= G_new[j, i] * Mdiag[j] * (r[j]^q)
                end

                b[eq] = fixed_numer - Mdiag[i] * exact
                for j in rows
                    A[eq, var_index[(j, i)]] = Mdiag[j] * (r[j]^q)
                end
                eq += 1
            end
        end

        x = _solve_exact_linear_system(A, b)
        return (x = x, var_index = var_index, divergence_rows_q1 = divergence_rows_q1)
    end

    if cfg.stencil_mode == :fixed_cols
        cfg.custom_stencil_cols === nothing &&
            throw(ArgumentError("`stencil_mode=:fixed_cols` requires `custom_stencil_cols`."))
        stencil_cols = _validate_fixed_stencil(copy(cfg.custom_stencil_cols), Nh, target_len)
        all(i -> i in Set(stencil_cols), rows) ||
            throw(ArgumentError("`custom_stencil_cols` must include all repaired rows $rows."))

        solved = _solve_for_stencil(stencil_cols)
        for row in rows
            for col in stencil_cols
                G_new[row, col] = solved.x[solved.var_index[(row, col)]]
            end
        end
        return (
                row_stencils = Dict(row => copy(stencil_cols) for row in rows),
                divergence_rows_q1 = solved.divergence_rows_q1,
                extra_points = max(0, length(stencil_cols) - target_len),
                expanded = false,
               )
    end

    cfg.stencil_mode == :raw_pattern_expand_right ||
        throw(ArgumentError("Unsupported stencil mode `$(cfg.stencil_mode)`."))

    len_start = max(target_len, maximum(rows) - 1)
    max_len = min(Nh - 1, len_start + max(2, length(rows)))
    last_err = nothing

    for len_try in len_start:max_len
        stencil_cols = collect(2:(1 + len_try))
        try
            solved = _solve_for_stencil(stencil_cols)
            for row in rows
                for col in stencil_cols
                    G_new[row, col] = solved.x[solved.var_index[(row, col)]]
                end
            end
            return (
                    row_stencils = Dict(row => copy(stencil_cols) for row in rows),
                    divergence_rows_q1 = solved.divergence_rows_q1,
                    extra_points = len_try - target_len,
                    expanded = len_try > target_len,
                   )
        catch err
            if err isa ArgumentError && occursin("No exact solution exists", sprint(showerror, err))
                last_err = err
                continue
            end
            rethrow(err)
        end
    end

    throw(ArgumentError("No exact coupled gradient+divergence solve for rows $rows with stencil width in [$len_start, $max_len]. Last error: $(last_err)"))
end

function _boundary_matrix_diag(Nh::Int, r::Vector{RatBig}, p::Int)
    Bdiag = zeros(RatBig, Nh)
    Bdiag[end] = r[end]^p
    return Bdiag
end

function _build_diagonal_matrix(diag_entries::Vector{RatBig})
    return Matrix(Diagonal(diag_entries))
end

function _compute_divergence_from_sbp(Mdiag::Vector{RatBig}, B::Matrix{RatBig}, G::Matrix{RatBig})
    RHS = B - transpose(G) * _build_diagonal_matrix(Mdiag)
    n = length(Mdiag)
    D = zeros(RatBig, n, n)
    @inbounds for i in 1:n
        mi = Mdiag[i]
        mi != 0 // 1 || throw(ArgumentError("Mass diagonal entry M[$i,$i] is zero; cannot invert M."))
        for j in 1:n
            D[i, j] = RHS[i, j] / mi
        end
    end
    return D
end

function _max_even_leq(q::Int)
    q < 0 && return -2
    return iseven(q) ? q : (q - 1)
end

function _even_monomial_powers(qmax::Int)
    qmax < 0 && return Int[]
    return collect(0:2:qmax)
end

function _max_odd_leq(q::Int)
    q < 1 && return 0
    return isodd(q) ? q : (q - 1)
end

function _odd_monomial_powers(qmax::Int)
    qmax < 1 && return Int[]
    return collect(1:2:qmax)
end

function _gradient_even_accuracy_error_for_row(row::Int,
                                               Grow::AbstractVector{RatBig},
                                               r::Vector{RatBig},
                                               q_even::Vector{Int})
    errs = Dict{Int, RatBig}()
    r0 = r[1]
    @inbounds for q in q_even
        target = q == 0 ? 0 // 1 : (big(q) // 1) * (r[row]^(q - 1))
        num = Grow[1] * (q == 0 ? 1 // 1 : r0^q)
        for k in 2:length(r)
            num += Grow[k] * (r[k]^q)
        end
        errs[q] = num - target
    end
    return errs
end

function _divergence_odd_accuracy_error_for_row_from_G(row::Int,
                                                       G::Matrix{RatBig},
                                                       Mdiag::Vector{RatBig},
                                                       Bdiag::Vector{RatBig},
                                                       r::Vector{RatBig},
                                                       p::Int,
                                                       q_odd::Vector{Int})
    errs = Dict{Int, RatBig}()
    @inbounds for q in q_odd
        numer = Bdiag[row] * (r[row]^q)
        for j in 1:length(r)
            numer -= G[j, row] * Mdiag[j] * (r[j]^q)
        end
        lhs = numer / Mdiag[row]
        rhs = (big(q + p) // 1) * (r[row]^(q - 1))
        errs[q] = lhs - rhs
    end
    return errs
end

function _build_repaired_gradient(cfg::LHopitalGradientRepairConfig,
                                  G_raw::Matrix{RatBig},
                                  G_odd::Matrix{RatBig},
                                  Mdiag_interior::Vector{RatBig},
                                  r::Vector{RatBig},
                                  q_infer::Int,
                                  p::Int;
                                  right_closure_width::Int)
    Nh = length(r)
    D1 = (big(p + 1) // 1) .* G_odd[1, :]

    # Match src/construct.jl row-selection rule:
    # repair exactly rows with nonzero first-column leakage in raw folded Geven.
    J_aff = _rows_with_nonzero_first_column_dense(G_raw)
    isempty(J_aff) && throw(ArgumentError("No repair rows found from nonzero first-column leakage in raw G."))

    Mdiag = copy(Mdiag_interior)

    w1 = if cfg.w1_mode == :user
        cfg.w1_user === nothing && throw(ArgumentError("`w1_mode=:user` requires `w1_user`."))
        _rat(cfg.w1_user)
    elseif cfg.w1_mode == :from_j2_raw
        D1[2] != 0 // 1 || throw(ArgumentError("Cannot compute w1 from j=2 because D1[2]=0."))
        -(G_raw[2, 1] * Mdiag[2]) / D1[2]
    else
        throw(ArgumentError("Unsupported `w1_mode=$(cfg.w1_mode)`."))
    end
    w1 > 0 // 1 || throw(ArgumentError("Origin weight w1 must be strictly positive; got $w1."))

    Mdiag[1] = w1
    any(m -> m <= 0 // 1, Mdiag) &&
        throw(ArgumentError("Mass positivity failed: all diagonal entries must be > 0."))

    G_new = copy(G_raw)
    row_repairs = NamedTuple[]

    q_even = _even_monomial_powers(_max_even_leq(q_infer))
    q_odd = _odd_monomial_powers(_max_odd_leq(q_infer - p))
    isempty(q_even) && throw(ArgumentError("No even-monomial constraints derived from q_infer=$q_infer."))

    @inbounds for j in J_aff
        G_new[j, 1] = -(w1 * D1[j]) / Mdiag[j]
    end

    Bdiag = _boundary_matrix_diag(Nh, r, p)
    row_block_end = min(Nh - 1, 1 + max(length(q_even) + length(q_odd), maximum(J_aff) - 1))
    J_solve = if 1 in q_odd
        sort(unique(vcat(J_aff, collect(2:row_block_end))))
    else
        copy(J_aff)
    end

    solve_info = _solve_coupled_gradient_divergence!(
                                                  G_new,
                                                  J_solve,
                                                  r,
                                                  Mdiag,
                                                  Bdiag,
                                                  p,
                                                  q_even,
                                                  q_odd,
                                                  cfg
                                                 )

    for j in J_solve
        grad_errs = _gradient_even_accuracy_error_for_row(j, view(G_new, j, :), r, q_even)
        grad_max = maximum(abs(v) for v in values(grad_errs))
        div_errs = _divergence_odd_accuracy_error_for_row_from_G(j, G_new, Mdiag, Bdiag, r, p, q_odd)
        div_max = isempty(div_errs) ? 0 // 1 : maximum(abs(v) for v in values(div_errs))

        push!(
              row_repairs,
              (
               row = j,
               stencil_cols = solve_info.row_stencils[j],
               grad_errors = grad_errs,
               grad_max_error = grad_max,
               div_errors = div_errs,
               div_max_error = div_max,
               expanded = solve_info.expanded,
               extra_points = solve_info.extra_points,
              )
             )
    end

    return (
            w1 = w1,
            D1 = D1,
            J_aff = J_aff,
            J_solve = J_solve,
            Mdiag = Mdiag,
            G_new = G_new,
            q_even = q_even,
            q_odd = q_odd,
            divergence_rows_q1 = solve_info.divergence_rows_q1,
            row_repairs = row_repairs,
           )
end

function _verify_exact_sbp(Mdiag::Vector{RatBig}, D::Matrix{RatBig}, G::Matrix{RatBig}, B::Matrix{RatBig})
    R = _build_diagonal_matrix(Mdiag) * D + transpose(G) * _build_diagonal_matrix(Mdiag) - B
    n, m = size(R)
    max_abs = 0 // 1
    @inbounds for i in 1:n, j in 1:m
        v = abs(R[i, j])
        if v > max_abs
            max_abs = v
        end
    end
    return (all_zero = max_abs == 0 // 1, residual = R, max_abs = max_abs)
end

function _build_w1_diagnostics(Js::Vector{Int}, G_raw::Matrix{RatBig}, G_new::Matrix{RatBig}, D1::Vector{RatBig}, Mdiag::Vector{RatBig})
    rows = NamedTuple[]
    for j in Js
        if 1 <= j <= length(D1)
            d = D1[j]
            graw = G_raw[j, 1]
            gnew = G_new[j, 1]
            mjj = Mdiag[j]
            w1_raw = d == 0 // 1 ? nothing : -(graw * mjj) / d
            w1_new = d == 0 // 1 ? nothing : -(gnew * mjj) / d
            push!(rows, (j = j, D1j = d, Grawj1 = graw, Gnewj1 = gnew, Mjj = mjj, w1_from_raw = w1_raw, w1_from_new = w1_new))
        end
    end
    return rows
end

function build_lhopital_repaired_canonical(cfg::LHopitalGradientRepairConfig)
    cfg.w1_mode in (:user, :from_j2_raw) || throw(ArgumentError("`w1_mode` must be :user or :from_j2_raw."))
    cfg.stencil_mode in (:raw_pattern_expand_right, :fixed_cols) ||
        throw(ArgumentError("`stencil_mode` must be :raw_pattern_expand_right or :fixed_cols."))

    p = cfg.spatial_dim - 1
    p >= 0 || throw(ArgumentError("Need `spatial_dim >= 1` so that p=spatial_dim-1 is nonnegative."))

    cart = _build_cartesian_canonical(cfg)
    xfull = cart.xfull
    Dcart = cart.Dcart
    Mcart = cart.Mcart

    folding = _build_folding_maps(xfull)
    r = folding.r
    Rop = folding.Rop
    Eeven = folding.Eeven
    Eodd = folding.Eodd

    G_raw = Rop * Dcart * Eeven
    G_odd = Rop * Dcart * Eodd
    right_closure_width = _right_closure_width_from_operator(cart.Dfull)

    M_half = (1 // 2) * transpose(Eeven) * Mcart * Eeven
    Nh = length(r)
    Mdiag_interior = zeros(RatBig, Nh)
    @inbounds for j in 2:Nh
        Mdiag_interior[j] = M_half[j, j] * (r[j]^p)
    end

    repaired = _build_repaired_gradient(cfg,
                                        G_raw,
                                        G_odd,
                                        Mdiag_interior,
                                        r,
                                        cart.q_infer,
                                        p;
                                        right_closure_width = right_closure_width)
    Mdiag = repaired.Mdiag
    G_new = repaired.G_new

    Bdiag = _boundary_matrix_diag(Nh, r, p)
    B = _build_diagonal_matrix(Bdiag)

    D_global = _compute_divergence_from_sbp(Mdiag, B, G_new)

    pole_row_match = D_global[1, :] == repaired.D1
    sbp = _verify_exact_sbp(Mdiag, D_global, G_new, B)

    grad_checks = NamedTuple[]
    for rr in repaired.row_repairs
        push!(grad_checks, (row = rr.row, max_abs_error = rr.grad_max_error, errors = rr.grad_errors))
    end

    all_grad_ok = all(rr -> rr.max_abs_error == 0 // 1, grad_checks)
    div_checks = NamedTuple[]
    for rr in repaired.row_repairs
        push!(div_checks, (row = rr.row, max_abs_error = rr.div_max_error, errors = rr.div_errors))
    end
    all_div_ok = all(rr -> rr.max_abs_error == 0 // 1, div_checks)

    div_constraint_q1_checks = NamedTuple[]
    if 1 in repaired.q_odd
        for i in repaired.divergence_rows_q1
            errs = _divergence_odd_accuracy_error_for_row_from_G(i, G_new, Mdiag, Bdiag, r, p, [1])
            maxe = isempty(errs) ? 0 // 1 : maximum(abs(v) for v in values(errs))
            push!(div_constraint_q1_checks, (row = i, max_abs_error = maxe, errors = errs))
        end
    end
    all_div_constraint_q1_ok = all(rr -> rr.max_abs_error == 0 // 1, div_constraint_q1_checks)

    pole_row_match || throw(ArgumentError("Verification failed: D_global first row does not match the prescribed L'Hopital row."))
    sbp.all_zero || throw(ArgumentError("Verification failed: SBP residual is nonzero (max abs $(sbp.max_abs))."))
    all_grad_ok || throw(ArgumentError("Verification failed: repaired gradient does not exactly satisfy even-monomial constraints."))
    all_div_ok || throw(ArgumentError("Verification failed: repaired divergence odd-monomial constraints are not exact."))
    all_div_constraint_q1_ok || throw(ArgumentError("Verification failed: constrained q=1 divergence rows are not exact."))

    w1_diag_rows = _build_w1_diagnostics(collect(2:min(4, Nh)), G_raw, G_new, repaired.D1, Mdiag)

    return (
            config = cfg,
            canonical_spacing = 1 // 1,
            Nhalf = cfg.Nhalf,
            Nhalf_index = cart.Nhalf_index,
            p = p,
            q_infer = cart.q_infer,
            xfull = xfull,
            r = r,
            D_cart = Dcart,
            M_cart = Mcart,
            Rop = Rop,
            Eeven = Eeven,
            Eodd = Eodd,
            M_half = M_half,
            Mdiag = Mdiag,
            Bdiag = Bdiag,
            B = B,
            G_raw = G_raw,
            G_odd = G_odd,
            G = G_new,
            D1 = repaired.D1,
            D = D_global,
            w1 = repaired.w1,
            J_aff = repaired.J_aff,
            J_solve = repaired.J_solve,
            q_even = repaired.q_even,
            q_odd = repaired.q_odd,
            divergence_rows_q1 = repaired.divergence_rows_q1,
            right_closure_width = right_closure_width,
            row_repairs = repaired.row_repairs,
            checks = (
                      pole_row_match = pole_row_match,
                      sbp_all_zero = sbp.all_zero,
                      sbp_max_abs = sbp.max_abs,
                      grad_all_exact = all_grad_ok,
                      grad_checks = grad_checks,
                      div_all_exact = all_div_ok,
                      div_checks = div_checks,
                      div_q1_constrained_all_exact = all_div_constraint_q1_ok,
                      div_q1_constrained_checks = div_constraint_q1_checks,
                     ),
            diagnostics = (
                           w1_rows_2_to_4 = w1_diag_rows,
                          ),
           )
end

function scale_repaired_operators(canonical_report; R_phys)
    haskey(canonical_report, :Nhalf) || throw(ArgumentError("`canonical_report` is missing `Nhalf`."))
    haskey(canonical_report, :p) || throw(ArgumentError("`canonical_report` is missing `p`."))

    Nhalf_index = haskey(canonical_report, :Nhalf_index) ? canonical_report.Nhalf_index : (canonical_report.Nhalf - 1)
    p = canonical_report.p
    Nhalf_index > 0 || throw(ArgumentError("Need at least 2 points on [0,R] to define spacing for scaling."))

    is_rat_scale = R_phys isa Integer || R_phys isa Rational
    T = is_rat_scale ? RatBig : typeof(float(R_phys))

    Rv = is_rat_scale ? _rat(R_phys) : convert(T, R_phys)
    rho = Rv / convert(T, Nhalf_index)

    r = canonical_report.r
    G = canonical_report.G
    D = canonical_report.D
    Mdiag = canonical_report.Mdiag
    Bdiag = canonical_report.Bdiag

    r_scaled = Vector{T}(undef, length(r))
    Mdiag_scaled = similar(r_scaled)
    Bdiag_scaled = similar(r_scaled)

    @inbounds for i in eachindex(r)
        r_scaled[i] = convert(T, r[i]) * rho
        Mdiag_scaled[i] = convert(T, Mdiag[i]) * rho^(p + 1)
        Bdiag_scaled[i] = convert(T, Bdiag[i]) * rho^p
    end

    n = length(r)
    G_scaled = Matrix{T}(undef, n, n)
    D_scaled = Matrix{T}(undef, n, n)
    invrho = inv(rho)
    @inbounds for i in 1:n, j in 1:n
        G_scaled[i, j] = convert(T, G[i, j]) * invrho
        D_scaled[i, j] = convert(T, D[i, j]) * invrho
    end

    M_scaled = Matrix(Diagonal(Mdiag_scaled))
    B_scaled = Matrix(Diagonal(Bdiag_scaled))

    sbp_resid = M_scaled * D_scaled + transpose(G_scaled) * M_scaled - B_scaled

    # Entrywise scaling-law residuals against the canonical operators.
    r_law_resid = Vector{T}(undef, n)
    Mdiag_law_resid = Vector{T}(undef, n)
    Bdiag_law_resid = Vector{T}(undef, n)
    G_law_resid = Matrix{T}(undef, n, n)
    D_law_resid = Matrix{T}(undef, n, n)
    @inbounds for i in 1:n
        r_law_resid[i] = r_scaled[i] - convert(T, r[i]) * rho
        Mdiag_law_resid[i] = Mdiag_scaled[i] - convert(T, Mdiag[i]) * rho^(p + 1)
        Bdiag_law_resid[i] = Bdiag_scaled[i] - convert(T, Bdiag[i]) * rho^p
        for j in 1:n
            G_law_resid[i, j] = G_scaled[i, j] - convert(T, G[i, j]) * invrho
            D_law_resid[i, j] = D_scaled[i, j] - convert(T, D[i, j]) * invrho
        end
    end

    sbp_ok = if is_rat_scale
        all(v -> v == 0 // 1, sbp_resid)
    else
        max_abs = maximum(abs, sbp_resid)
        max_abs <= 1e-12
    end

    scale_laws_ok = if is_rat_scale
        all(v -> v == 0 // 1, r_law_resid) &&
        all(v -> v == 0 // 1, Mdiag_law_resid) &&
        all(v -> v == 0 // 1, Bdiag_law_resid) &&
        all(v -> v == 0 // 1, G_law_resid) &&
        all(v -> v == 0 // 1, D_law_resid)
    else
        tol = 1e-12
        maximum(abs, r_law_resid) <= tol &&
        maximum(abs, Mdiag_law_resid) <= tol &&
        maximum(abs, Bdiag_law_resid) <= tol &&
        maximum(abs, G_law_resid) <= tol &&
        maximum(abs, D_law_resid) <= tol
    end

    return (
            R_phys = Rv,
            rho = rho,
            r = r_scaled,
            Mdiag = Mdiag_scaled,
            Bdiag = Bdiag_scaled,
            M = M_scaled,
            B = B_scaled,
            G = G_scaled,
            D = D_scaled,
            checks = (
                      sbp_ok = sbp_ok,
                      sbp_residual = sbp_resid,
                      scale_laws_ok = scale_laws_ok,
                      scale_law_residuals = (
                                             r = r_law_resid,
                                             Mdiag = Mdiag_law_resid,
                                             Bdiag = Bdiag_law_resid,
                                             G = G_law_resid,
                                             D = D_law_resid,
                                            ),
                     ),
           )
end

function run_lhopital_gradient_repair(cfg::LHopitalGradientRepairConfig; R_phys = nothing)
    report = build_lhopital_repaired_canonical(cfg)

    println("L'Hopital Gradient Repair (canonical Δr=1, exact Rational{BigInt})")
    println("  accuracy_order = ", report.config.accuracy_order, ", inferred = ", report.q_infer)
    println("  Npoints[0,R] = ", report.Nhalf, ", Nhalf_index = ", report.Nhalf_index, ", Nh = ", length(report.r),
            ", spatial_dim = ", report.config.spatial_dim, ", p = ", report.p)
    println("  right_closure_width = ", report.right_closure_width)
    println("  constrained_divergence_rows_q1 = ", report.divergence_rows_q1)
    println("  w1_mode = ", report.config.w1_mode, ", w1 = ", _ratstr(report.w1))
    println("  stencil_mode = ", report.config.stencil_mode)
    println("  affected rows J_aff = ", report.J_aff)
    println("  solved rows J_solve = ", report.J_solve)

    println("\nChecks:")
    println("  Pole row retained exactly: ", report.checks.pole_row_match)
    println("  SBP residual identically zero: ", report.checks.sbp_all_zero,
            " (max abs = ", _ratstr(report.checks.sbp_max_abs), ")")
    println("  Repaired gradient even-monomial exactness: ", report.checks.grad_all_exact)
    println("  Repaired divergence odd-monomial exactness: ", report.checks.div_all_exact)
    println("  Constrained q=1 divergence-row exactness: ", report.checks.div_q1_constrained_all_exact)

    println("\nDiagnostics (j=2,3,4 when available):")
    for row in report.diagnostics.w1_rows_2_to_4
        w1raw = isnothing(row.w1_from_raw) ? "n/a" : _ratstr(row.w1_from_raw)
        w1new = isnothing(row.w1_from_new) ? "n/a" : _ratstr(row.w1_from_new)
        @printf("  j=%d  D1=%s  Graw=%s  Gnew=%s  Mjj=%s  w1(raw)=%s  w1(new)=%s\n",
                row.j, _ratstr(row.D1j), _ratstr(row.Grawj1), _ratstr(row.Gnewj1), _ratstr(row.Mjj), w1raw, w1new)
    end

    println("\nRow repair summary:")
    for rr in report.row_repairs
        @printf("  row=%d  stencil=%s  extra_points=%d  expanded=%s  grad_max_err=%s  div_max_err=%s\n",
                rr.row,
                string(rr.stencil_cols),
                rr.extra_points,
                string(rr.expanded),
                _ratstr(rr.grad_max_error),
                _ratstr(rr.div_max_error))
    end

    scaled = nothing
    if !isnothing(R_phys)
        scaled = scale_repaired_operators(report; R_phys = R_phys)
        println("\nPhysical scaling:")
        println("  R_phys = ", scaled.R_phys, ", rho = ", scaled.rho)
        println("  Scaled SBP check: ", scaled.checks.sbp_ok)
        println("  Entrywise scaling-law check: ", scaled.checks.scale_laws_ok)
    end

    return (canonical = report, scaled = scaled)
end

function _parse_rational_token(token::String)
    if occursin("//", token)
        parts = split(token, "//")
        length(parts) == 2 || throw(ArgumentError("Invalid rational literal `$token`."))
        return parse(BigInt, strip(parts[1])) // parse(BigInt, strip(parts[2]))
    elseif occursin("/", token)
        parts = split(token, "/")
        length(parts) == 2 || throw(ArgumentError("Invalid rational literal `$token`."))
        return parse(BigInt, strip(parts[1])) // parse(BigInt, strip(parts[2]))
    elseif occursin(r"[\.eE]", token)
        return parse(Float64, token)
    end
    return parse(BigInt, token) // 1
end

function print_lhopital_gradient_repair_cli_help(io::IO = stdout)
    println(io, "Usage: julia scripts/lhopital_gradient_repair/run_lhopital_gradient_repair.jl --accuracy-order <int> --nhalf <int> [options]")
    println(io, "")
    println(io, "Options:")
    println(io, "  --accuracy-order <int>              Required Cartesian SBP accuracy order")
    println(io, "  --nhalf <int>                       Required number of points on [0,R] (including r=0 and r=R)")
    println(io, "  --spatial-dim <int>                 Spatial dimension d (default: 3)")
    println(io, "  --w1-num <int> --w1-den <int>       User origin weight w1 = num//den (enables w1_mode=:user)")
    println(io, "  --w1-from-j2                        Set w1_mode=:from_j2_raw")
    println(io, "  --stencil-mode <raw_pattern_expand_right|fixed_cols>")
    println(io, "                                       Default: raw_pattern_expand_right")
    println(io, "  --stencil-cols <comma list>         Used when stencil-mode=fixed_cols (example: 2,3,4,5)")
    println(io, "  --R-phys <value>                    Optional physical radius for scaled operators")
    println(io, "                                       Value can be integer, fraction (20/3), or float")
    println(io, "  -h, --help                          Show this help")
end

function parse_lhopital_gradient_repair_cli_args(args::Vector{String})
    accuracy_order = nothing
    nhalf = nothing
    spatial_dim = 3

    w1_mode = :from_j2_raw
    w1_num = nothing
    w1_den = nothing

    stencil_mode = :raw_pattern_expand_right
    stencil_cols = nothing
    R_phys = nothing

    i = 1
    while i <= length(args)
        arg = args[i]
        val() = (i += 1; i <= length(args) || throw(ArgumentError("Missing value for $arg")); args[i])

        if arg in ("-h", "--help")
            print_lhopital_gradient_repair_cli_help()
            return nothing
        elseif arg == "--accuracy-order"
            accuracy_order = parse(Int, val())
        elseif arg == "--nhalf"
            nhalf = parse(Int, val())
        elseif arg == "--spatial-dim"
            spatial_dim = parse(Int, val())
        elseif arg == "--w1-num"
            w1_num = parse(BigInt, val())
            w1_mode = :user
        elseif arg == "--w1-den"
            w1_den = parse(BigInt, val())
            w1_mode = :user
        elseif arg == "--w1-from-j2"
            w1_mode = :from_j2_raw
        elseif arg == "--stencil-mode"
            sm = Symbol(val())
            sm in (:raw_pattern_expand_right, :fixed_cols) ||
                throw(ArgumentError("Invalid --stencil-mode $sm"))
            stencil_mode = sm
        elseif arg == "--stencil-cols"
            raw = split(val(), ",")
            stencil_cols = [parse(Int, strip(tok)) for tok in raw if !isempty(strip(tok))]
        elseif arg == "--R-phys"
            R_phys = _parse_rational_token(val())
        else
            throw(ArgumentError("Unknown argument: $arg"))
        end
        i += 1
    end

    isnothing(accuracy_order) && throw(ArgumentError("`--accuracy-order` is required."))
    isnothing(nhalf) && throw(ArgumentError("`--nhalf` is required."))

    w1_user = nothing
    if w1_mode == :user
        (w1_num === nothing || w1_den === nothing) &&
            throw(ArgumentError("`--w1-num` and `--w1-den` must both be provided for user w1 mode."))
        w1_den == 0 && throw(ArgumentError("`--w1-den` must be nonzero."))
        w1_user = w1_num // w1_den
    end

    cfg = LHopitalGradientRepairConfig(
                                       accuracy_order = accuracy_order,
                                       Nhalf = nhalf,
                                       spatial_dim = spatial_dim,
                                       w1_mode = w1_mode,
                                       w1_user = w1_user,
                                       stencil_mode = stencil_mode,
                                       custom_stencil_cols = stencil_cols,
                                      )
    return (cfg = cfg, R_phys = R_phys)
end
