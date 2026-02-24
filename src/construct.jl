function _extract_diagonal(H::SparseMatrixCSC{T, Ti}; atol::T) where {T <: Real, Ti <: Integer}
    n, m = size(H)
    n == m || throw(DimensionMismatch("Mass matrix must be square."))
    diag_entries = fill(zero(T), n)
    max_offdiag = zero(T)
    I, J, V = findnz(H)
    @inbounds for k in eachindex(V)
        i = I[k]
        j = J[k]
        v = V[k]
        if i == j
            diag_entries[i] = v
        else
            av = abs(v)
            if av > max_offdiag
                max_offdiag = av
            end
        end
    end

    if T <: AbstractFloat
        allowed = max(atol, T(128) * eps(T))
        max_offdiag <= allowed ||
            throw(ArgumentError("Metric mass matrix must be diagonal-norm; detected off-diagonal magnitude $max_offdiag."))
    else
        max_offdiag == zero(T) ||
            throw(ArgumentError("Exact arithmetic requires a strictly diagonal mass matrix."))
    end
    return diag_entries
end

function _set_divergence_rows!(D::SparseMatrixCSC{T, Ti},
                               RHS::SparseMatrixCSC{T, Ti},
                               Hdiag::Vector{T}) where {T <: Real, Ti <: Integer}
    n = size(D, 1)
    @inbounds for i in 2:n
        hi = Hdiag[i]
        hi == zero(T) &&
            throw(ArgumentError("Encountered H[$i,$i] = 0 while deriving divergence row from SBP."))
        for j in 1:n
            v = RHS[i, j]
            if v != zero(T)
                D[i, j] = v / hi
            end
        end
    end
    return D
end

function _set_origin_row!(D::SparseMatrixCSC{T, Ti},
                          Godd::SparseMatrixCSC{T, Ti},
                          p::Int) where {T <: Real, Ti <: Integer}
    n = size(D, 1)
    factor = convert(T, p + 1)
    @inbounds for j in 1:n
        v = factor * Godd[1, j]
        if v != zero(T)
            D[1, j] = v
        end
    end
    return D
end

@inline function _values_match(a::T, b::T; atol::T) where {T <: AbstractFloat}
    scale = max(one(T), abs(a), abs(b))
    return abs(a - b) <= max(atol, T(1024) * eps(T) * scale)
end

@inline function _values_match(a::T, b::T; atol::T) where {T <: Real}
    return a == b
end

@inline function _is_effective_nonzero(v::T; atol::T) where {T <: AbstractFloat}
    return abs(v) > max(atol, T(1024) * eps(T))
end

@inline function _is_effective_nonzero(v::T; atol::T) where {T <: Real}
    return v != zero(T)
end

function _rows_with_nonzero_first_column(G::SparseMatrixCSC{T, Ti}; atol::T) where {T <: Real, Ti <: Integer}
    n = size(G, 1)
    rows = Int[]
    @inbounds for i in 2:n
        _is_effective_nonzero(G[i, 1]; atol = atol) && push!(rows, i)
    end
    return rows
end

function _infer_interior_accuracy_order(Dfull)
    hasproperty(Dfull, :coefficients) ||
        throw(ArgumentError("Could not infer interior accuracy: operator has no `coefficients` field."))
    coeffs = getproperty(Dfull, :coefficients)
    hasproperty(coeffs, :accuracy_order) ||
        throw(ArgumentError("Could not infer interior accuracy: `coefficients.accuracy_order` is unavailable."))
    q = Int(getproperty(coeffs, :accuracy_order))
    q > 0 || throw(ArgumentError("Inferred non-positive interior accuracy order: $q."))
    return q
end

function _cartesian_half_bandwidth(Dfull)
    hasproperty(Dfull, :coefficients) ||
        throw(ArgumentError("Could not infer bandwidth: operator has no `coefficients` field."))
    coeffs = getproperty(Dfull, :coefficients)

    hasproperty(coeffs, :lower_coef) || hasproperty(coeffs, :upper_coef) ||
        throw(ArgumentError("Could not infer bandwidth: coefficients expose neither `lower_coef` nor `upper_coef`."))

    lower_bw = hasproperty(coeffs, :lower_coef) ? length(getproperty(coeffs, :lower_coef)) : 0
    upper_bw = hasproperty(coeffs, :upper_coef) ? length(getproperty(coeffs, :upper_coef)) : 0
    bw = max(lower_bw, upper_bw)
    bw > 0 || throw(ArgumentError("Inferred non-positive Cartesian half-bandwidth: $bw."))
    return bw
end

function _infer_boundary_accuracy_order(Gfull::SparseMatrixCSC{T, Ti},
                                        xfull::Vector{T},
                                        interior_accuracy::Int;
                                        atol::T) where {T <: Real, Ti <: Integer}
    n = length(xfull)
    n == size(Gfull, 1) == size(Gfull, 2) ||
        throw(DimensionMismatch("Incompatible matrix/grid dimensions while inferring boundary accuracy."))

    # Infer boundary exactness from the first Cartesian boundary row on monomials x^k.
    boundary_accuracy = -1
    row = 1
    @inbounds for degree in 0:interior_accuracy
        u = xfull .^ degree
        exact = degree == 0 ? zero(T) : convert(T, degree) * xfull[row]^(degree - 1)

        numerical = zero(T)
        for j in 1:n
            v = Gfull[row, j]
            if v != zero(T)
                numerical += v * u[j]
            end
        end

        if _values_match(numerical, exact; atol = atol)
            boundary_accuracy = degree
        else
            break
        end
    end

    boundary_accuracy >= 0 ||
        throw(ArgumentError("Failed to infer boundary accuracy from the base Cartesian operator."))

    return boundary_accuracy
end

@inline _num_even_constraints(boundary_accuracy::Int) = boundary_accuracy + 1

function _resolve_stencil_cols(custom_stencil_cols::Union{Nothing, Vector{Int}},
                               num_constraints::Int,
                               Nh::Int;
                               default_length::Int = num_constraints)
    default_length > 0 || throw(ArgumentError("`default_length` must be positive."))
    stencil_cols = isnothing(custom_stencil_cols) ? collect(2:(1 + default_length)) : copy(custom_stencil_cols)

    isempty(stencil_cols) && throw(ArgumentError("`stencil_cols` must be non-empty."))
    1 in stencil_cols && throw(ArgumentError("`stencil_cols` must not include index 1 (the origin)."))
    any(c -> c < 2 || c > Nh, stencil_cols) &&
        throw(ArgumentError("`stencil_cols` must lie in 2:$Nh."))
    length(unique(stencil_cols)) == length(stencil_cols) ||
        throw(ArgumentError("`stencil_cols` must not contain duplicate indices."))
    length(stencil_cols) >= num_constraints ||
        throw(ArgumentError("Stencil bandwidth ($(length(stencil_cols))) must be >= num_constraints ($num_constraints)."))

    return stencil_cols
end

"""
    compute_even_gradient_row_decoupled(row_index, stencil_indices, num_constraints)

Compute exact rational weights for an even-gradient row while decoupling stencil
bandwidth from accuracy constraints.

If `length(stencil_indices) > num_constraints`, the minimum L2-norm exact
solution is returned.
"""
function compute_even_gradient_row_decoupled(row_index::Int,
                                             stencil_indices::Vector{Int},
                                             num_constraints::Int)
    row_index >= 2 || throw(ArgumentError("`row_index` must satisfy row_index >= 2."))
    1 in stencil_indices && throw(ArgumentError("`stencil_indices` must not include 1 (the origin)."))

    n_weights = length(stencil_indices)
    n_weights >= num_constraints ||
        throw(ArgumentError("Bandwidth ($n_weights) must be >= num_constraints ($num_constraints)."))

    # Vandermonde-like system on even powers (in index-space coordinates).
    V = zeros(Rational{BigInt}, num_constraints, n_weights)
    b = zeros(Rational{BigInt}, num_constraints)
    r_i = big(row_index - 1)

    for m in 0:(num_constraints - 1)
        power = 2 * m
        b[m + 1] = m == 0 ? 0 // 1 : (power * r_i^(power - 1)) // 1

        for (col_idx, j) in enumerate(stencil_indices)
            r_j = big(j - 1)
            V[m + 1, col_idx] = (r_j^power) // 1
        end
    end

    if n_weights == num_constraints
        return V \ b
    end

    # Minimum-norm exact solution for underdetermined systems.
    return transpose(V) * ((V * transpose(V)) \ b)
end

"""
    verify_even_gradient_row_decoupled(row_index, stencil_indices, weights, num_constraints)

Return `true` only if the row differentiates even monomials exactly for all
constraints `r^0, r^2, ..., r^(2*(num_constraints-1))`.
"""
function verify_even_gradient_row_decoupled(row_index::Int,
                                            stencil_indices::Vector{Int},
                                            weights::Vector{Rational{BigInt}},
                                            num_constraints::Int)
    length(weights) == length(stencil_indices) ||
        throw(DimensionMismatch("`weights` and `stencil_indices` must have the same length."))

    r_i = big(row_index - 1)
    @inbounds for m in 0:(num_constraints - 1)
        power = 2 * m
        exact_deriv = m == 0 ? 0 // 1 : (power * r_i^(power - 1)) // 1
        num_deriv = sum(weights[idx] * (big(j - 1)^power) for (idx, j) in enumerate(stencil_indices))
        exact_deriv == num_deriv || return false
    end
    return true
end

function _uniform_spacing(r::Vector{T}; atol::T) where {T <: AbstractFloat}
    length(r) >= 2 || throw(ArgumentError("At least two grid points are required to infer spacing."))
    Δr = r[2] - r[1]
    Δr > zero(T) || throw(ArgumentError("Grid spacing must be strictly positive."))

    tol = max(atol, T(512) * eps(T) * max(one(T), abs(Δr)))
    @inbounds for i in 3:length(r)
        δ = r[i] - r[i - 1]
        abs(δ - Δr) <= tol ||
            throw(ArgumentError("Non-uniform half-grid detected while inferring scaling (|Δr[$i]-Δr[2]|=$(abs(δ - Δr)))."))
    end

    return Δr
end

function _uniform_spacing(r::Vector{T}; atol::T) where {T <: Real}
    length(r) >= 2 || throw(ArgumentError("At least two grid points are required to infer spacing."))
    Δr = r[2] - r[1]
    Δr > zero(T) || throw(ArgumentError("Grid spacing must be strictly positive."))

    @inbounds for i in 3:length(r)
        δ = r[i] - r[i - 1]
        δ == Δr || throw(ArgumentError("Non-uniform half-grid detected while inferring scaling."))
    end

    return Δr
end

function _solve_exact_linear_system(A::Matrix{Rational{BigInt}},
                                    b::Vector{Rational{BigInt}})
    n_eq, n_vars = size(A)
    n_eq == length(b) || throw(DimensionMismatch("Incompatible linear-system dimensions."))
    n_eq == 0 && return zeros(Rational{BigInt}, n_vars)

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
            throw(ArgumentError("Coupled row/column constraints are inconsistent; no exact solution exists for this stencil."))
        end
    end

    x = zeros(Rational{BigInt}, n_vars)
    for (r, col) in enumerate(pivot_cols)
        x[col] = M[r, n_vars + 1]
    end
    return x
end

function _divergence_constraint_degrees(interior_accuracy::Int, p::Int)
    max_odd_degree = interior_accuracy - p
    max_odd_degree >= 1 || return Int[]
    return collect(1:2:max_odd_degree)
end

function _verify_coupled_origin_repair(Geven::SparseMatrixCSC{T, Ti},
                                       rows_to_solve::Vector{Int},
                                       num_constraints::Int,
                                       divergence_degrees::Vector{Int},
                                       r::Vector{T},
                                       Hdiag::Vector{T},
                                       Bdiag::Vector{T},
                                       p::Int) where {T <: Real, Ti <: Integer}
    Nh = length(r)

    @inbounds for row in rows_to_solve
        Geven[row, 1] == zero(T) || return false
        for m in 0:(num_constraints - 1)
            power = 2 * m
            exact = m == 0 ? zero(T) : convert(T, power) * r[row]^(power - 1)
            num = zero(T)
            for col in 1:Nh
                num += Geven[row, col] * r[col]^power
            end
            num == exact || return false
        end
    end

    @inbounds for i in rows_to_solve
        hi = Hdiag[i]
        hi == zero(T) && return false

        for degree in divergence_degrees
            exact = convert(T, p + degree) * r[i]^(degree - 1)

            numer = Bdiag[i] * r[i]^degree
            for j in 1:Nh
                numer -= Geven[j, i] * Hdiag[j] * r[j]^degree
            end

            numer / hi == exact || return false
        end
    end

    return true
end

function _solve_coupled_geven_block(Geven::SparseMatrixCSC{T, Ti},
                                    rows_to_solve::Vector{Int},
                                    stencil_cols::Vector{Int},
                                    num_constraints::Int,
                                    divergence_degrees::Vector{Int},
                                    r::Vector{T},
                                    Hdiag::Vector{T},
                                    Bdiag::Vector{T},
                                    p::Int) where {T <: Real, Ti <: Integer}
    row_count = length(rows_to_solve)
    row_set = Set(rows_to_solve)
    stencil_set = Set(stencil_cols)
    all(i -> i in stencil_set, rows_to_solve) ||
        throw(ArgumentError("`stencil_cols` must include all `rows_to_solve` for coupled divergence constraints."))

    divergence_constraints = length(divergence_degrees)
    n_unknowns = row_count * length(stencil_cols)
    n_equations = row_count * (num_constraints + divergence_constraints)

    A = zeros(Rational{BigInt}, n_equations, n_unknowns)
    b = zeros(Rational{BigInt}, n_equations)
    var_index = Dict{Tuple{Int, Int}, Int}()

    idx = 1
    for row in rows_to_solve
        for col in stencil_cols
            var_index[(row, col)] = idx
            idx += 1
        end
    end

    eq = 1
    Nh = length(r)

    # Gradient constraints on modified rows (even monomials).
    @inbounds for row in rows_to_solve
        for m in 0:(num_constraints - 1)
            power = 2 * m
            target = m == 0 ? 0 // 1 : convert(Rational{BigInt}, power) * convert(Rational{BigInt}, r[row]^(power - 1))

            fixed = 0 // 1
            for col in 1:Nh
                col in stencil_set && continue
                gij = (col == 1) ? (0 // 1) : convert(Rational{BigInt}, Geven[row, col])
                fixed += gij * convert(Rational{BigInt}, r[col]^power)
            end

            b[eq] = target - fixed
            for col in stencil_cols
                A[eq, var_index[(row, col)]] = convert(Rational{BigInt}, r[col]^power)
            end
            eq += 1
        end
    end

    # Divergence constraints on near-origin rows (odd monomials), expressed via G^T H.
    @inbounds for i in rows_to_solve
        hi = convert(Rational{BigInt}, Hdiag[i])
        hi != 0 // 1 || throw(ArgumentError("Encountered zero mass diagonal at i=$i in coupled divergence constraints."))

        for degree in divergence_degrees
            exact = convert(Rational{BigInt}, p + degree) * convert(Rational{BigInt}, r[i]^(degree - 1))

            fixed_numer = convert(Rational{BigInt}, Bdiag[i]) * convert(Rational{BigInt}, r[i]^degree)
            for j in 1:Nh
                if (j in row_set) && (i in stencil_set)
                    continue
                end
                fixed_numer -= convert(Rational{BigInt}, Geven[j, i]) *
                               convert(Rational{BigInt}, Hdiag[j]) *
                               convert(Rational{BigInt}, r[j]^degree)
            end

            b[eq] = fixed_numer - hi * exact
            for j in rows_to_solve
                A[eq, var_index[(j, i)]] = convert(Rational{BigInt}, Hdiag[j]) *
                                           convert(Rational{BigInt}, r[j]^degree)
            end
            eq += 1
        end
    end

    x = _solve_exact_linear_system(A, b)
    return x, var_index, n_equations, n_unknowns
end

function _repair_even_gradient_first_column!(Geven::SparseMatrixCSC{T, Ti},
                                             Dfull,
                                             Gfull::SparseMatrixCSC{T, Ti},
                                             xfull::Vector{T},
                                             Hdiag::Vector{T},
                                             Bdiag::Vector{T},
                                             r::Vector{T};
                                             p::Int,
                                             atol::T,
                                             custom_stencil_cols::Union{Nothing, Vector{Int}} = nothing) where {T <: Real, Ti <: Integer}
    T <: Rational || throw(ArgumentError("Coupled origin repair requires exact rational arithmetic (`Rational{BigInt}` path)."))

    rows_to_solve = _rows_with_nonzero_first_column(Geven; atol = atol)
    cartesian_half_bandwidth = _cartesian_half_bandwidth(Dfull)
    interior_accuracy = _infer_interior_accuracy_order(Dfull)
    boundary_accuracy = _infer_boundary_accuracy_order(Gfull, xfull, interior_accuracy; atol = atol)
    num_constraints = _num_even_constraints(boundary_accuracy)
    requested_divergence_degrees = _divergence_constraint_degrees(interior_accuracy, p)
    divergence_degrees = copy(requested_divergence_degrees)
    divergence_constraints = length(divergence_degrees)

    Nh = length(r)
    default_stencil_len = num_constraints + divergence_constraints
    initial_stencil_len = min(default_stencil_len, Nh - 1)
    stencil_cols = _resolve_stencil_cols(
                                         custom_stencil_cols,
                                         num_constraints,
                                         Nh;
                                         default_length = initial_stencil_len
                                        )
    Δr = _uniform_spacing(r; atol = atol)

    if isempty(rows_to_solve)
        return (
                rows_to_solve = rows_to_solve,
                divergence_rows = Int[],
                cartesian_half_bandwidth = cartesian_half_bandwidth,
                interior_accuracy = interior_accuracy,
                boundary_accuracy = boundary_accuracy,
                num_constraints = num_constraints,
                divergence_constraints = divergence_constraints,
                divergence_constraint_degrees = divergence_degrees,
                requested_divergence_constraint_degrees = requested_divergence_degrees,
                stencil_cols = stencil_cols,
                n_equations = 0,
                n_unknowns = 0,
                Δr = Δr
               )
    end

    var_index = Dict{Tuple{Int, Int}, Int}()
    x = Rational{BigInt}[]
    n_equations = 0
    n_unknowns = 0

    if isnothing(custom_stencil_cols)
        solved = false
        while true
            divergence_constraints = length(divergence_degrees)
            start_len = min(num_constraints + divergence_constraints, Nh - 1)
            local_solved = false

            for len_try in start_len:(Nh - 1)
                stencil_try = _resolve_stencil_cols(
                                                   nothing,
                                                   num_constraints,
                                                   Nh;
                                                   default_length = len_try
                                                  )
                try
                    x_try, map_try, n_eq_try, n_unk_try = _solve_coupled_geven_block(
                                                                                      Geven,
                                                                                      rows_to_solve,
                                                                                      stencil_try,
                                                                                      num_constraints,
                                                                                      divergence_degrees,
                                                                                      r,
                                                                                      Hdiag,
                                                                                      Bdiag,
                                                                                      p
                                                                                     )
                    x = x_try
                    var_index = map_try
                    n_equations = n_eq_try
                    n_unknowns = n_unk_try
                    stencil_cols = stencil_try
                    solved = true
                    local_solved = true
                    break
                catch err
                    if err isa ArgumentError && occursin("no exact solution exists", sprint(showerror, err))
                        continue
                    end
                    rethrow()
                end
            end

            local_solved && break
            isempty(divergence_degrees) && break
            pop!(divergence_degrees)
        end

        if !solved
            throw(
                  ArgumentError(
                                "Could not find a consistent exact coupled solve up to stencil length $(Nh - 1) " *
                                "for requested divergence degrees $(requested_divergence_degrees)."
                               )
                 )
        end
    else
        divergence_constraints = length(divergence_degrees)
        x, var_index, n_equations, n_unknowns = _solve_coupled_geven_block(
                                                                          Geven,
                                                                          rows_to_solve,
                                                                          stencil_cols,
                                                                          num_constraints,
                                                                          divergence_degrees,
                                                                          r,
                                                                          Hdiag,
                                                                          Bdiag,
                                                                          p
                                                                         )
    end

    @inbounds for row in rows_to_solve
        Geven[row, 1] = zero(T)
        for col in stencil_cols
            Geven[row, col] = convert(T, x[var_index[(row, col)]])
        end
    end

    snap_sparse!(Geven)

    _verify_coupled_origin_repair(
                                  Geven,
                                  rows_to_solve,
                                  num_constraints,
                                  divergence_degrees,
                                  r,
                                  Hdiag,
                                  Bdiag,
                                  p
                                 ) ||
        throw(ArgumentError("Coupled row/column repair verification failed exactly."))

    return (
            rows_to_solve = rows_to_solve,
            divergence_rows = copy(rows_to_solve),
            cartesian_half_bandwidth = cartesian_half_bandwidth,
            interior_accuracy = interior_accuracy,
            boundary_accuracy = boundary_accuracy,
            num_constraints = num_constraints,
            divergence_constraints = divergence_constraints,
            divergence_constraint_degrees = divergence_degrees,
            requested_divergence_constraint_degrees = requested_divergence_degrees,
            stencil_cols = stencil_cols,
            n_equations = n_equations,
            n_unknowns = n_unknowns,
            Δr = Δr
           )
end

function _assert_global_even_interior_accuracy(Geven::SparseMatrixCSC{T, Ti},
                                               r::Vector{T},
                                               interior_accuracy::Int,
                                               rows_to_solve::Vector{Int},
                                               right_closure::Int;
                                               atol::T) where {T <: Real, Ti <: Integer}
    Nh = length(r)
    degree = interior_accuracy

    phi = r .^ degree
    dphi_exact = degree == 0 ? fill(zero(T), Nh) : convert(T, degree) .* (r .^ (degree - 1))
    dphi_num = Geven * phi

    left_boundary_end = isempty(rows_to_solve) ? 1 : max(1, maximum(rows_to_solve))
    start_idx = left_boundary_end + 1
    stop_idx = Nh - right_closure
    start_idx <= stop_idx || return nothing

    if T <: AbstractFloat
        err = maximum(abs.(view(dphi_num, start_idx:stop_idx) .- view(dphi_exact, start_idx:stop_idx)))
        scale = max(one(T), maximum(abs.(view(dphi_exact, start_idx:stop_idx))))
        tol = max(atol, T(2048) * eps(T) * scale)
        err <= tol ||
            throw(ArgumentError("Global interior accuracy check failed for even degree $degree (max error $err > $tol)."))
    else
        @inbounds for i in start_idx:stop_idx
            dphi_num[i] == dphi_exact[i] ||
                throw(ArgumentError("Global interior accuracy check failed at row $i for even degree $degree."))
        end
    end

    return nothing
end

function _default_scale_eltype(R)
    if R isa AbstractFloat
        return typeof(R)
    end
    if R isa Rational
        return Rational{BigInt}
    end
    if R isa Integer
        return Rational{BigInt}
    end
    return Float64
end

function _scale_sparse_matrix(A::SparseMatrixCSC{Ta, Ti},
                              factor,
                              ::Type{Tb};
                              snap_factor::Float64) where {Ta <: Real, Ti <: Integer, Tb <: Real}
    n, m = size(A)
    I, J, V = findnz(A)
    f = convert(Tb, factor)
    W = Vector{Tb}(undef, length(V))
    @inbounds for k in eachindex(V)
        W[k] = convert(Tb, V[k]) * f
    end
    B = sparse(I, J, W, n, m)
    snap_sparse!(B; snap_factor = snap_factor)
    return B
end

"""
    scale_spherical_operators(ops, R; target_eltype=nothing, atol=nothing)

Scale an SBP operator set from its current uniform spacing to a new physical radius `R`.

Scaling rules for `r = ρ r̂`:
- `Geven`, `Godd`, `D` scale by `1/ρ`
- `H` scales by `ρ^(p+1)`
- `B` scales by `ρ^p`
"""
function scale_spherical_operators(ops::SphericalOperators,
                                   R;
                                   target_eltype::Union{Nothing, Type} = nothing,
                                   atol = nothing)
    ops.Nh >= 2 || throw(ArgumentError("At least two half-grid points are required for scaling."))

    Tout = isnothing(target_eltype) ? _default_scale_eltype(R) : target_eltype
    Tout <: Real || throw(ArgumentError("`target_eltype` must be a subtype of `Real`."))

    N = ops.Nh - 1
    Rtarget = convert(Tout, R)
    Δr_target = Rtarget / convert(Tout, N)

    Tsrc = eltype(ops.r)
    atol_src = _resolve_atol(Tsrc, nothing)
    Δr_src = _uniform_spacing(ops.r; atol = atol_src)
    scale_ratio = Δr_target / convert(Tout, Δr_src)

    r_scaled = Vector{Tout}(undef, ops.Nh)
    @inbounds for i in eachindex(ops.r)
        r_scaled[i] = convert(Tout, ops.r[i]) * scale_ratio
    end

    H_scaled = _scale_sparse_matrix(ops.H, scale_ratio^(ops.p + 1), Tout; snap_factor = ops.snap_factor)
    B_scaled = _scale_sparse_matrix(ops.B, scale_ratio^ops.p, Tout; snap_factor = ops.snap_factor)
    Geven_scaled = _scale_sparse_matrix(ops.Geven, inv(scale_ratio), Tout; snap_factor = ops.snap_factor)
    Godd_scaled = _scale_sparse_matrix(ops.Godd, inv(scale_ratio), Tout; snap_factor = ops.snap_factor)
    D_scaled = _scale_sparse_matrix(ops.D, inv(scale_ratio), Tout; snap_factor = ops.snap_factor)

    atol_scaled = _resolve_atol(Tout, atol)

    return SphericalOperators(
                              r_scaled,
                              H_scaled,
                              B_scaled,
                              Geven_scaled,
                              Godd_scaled,
                              D_scaled,
                              ops.closure_width,
                              ops.accuracy_order,
                              ops.p,
                              Rtarget,
                              ops.source,
                              ops.mode,
                              atol_scaled,
                              ops.snap_factor,
                              ops.build_matrix,
                              ops.M_full,
                              ops.Nh
                             )
end

"""
    spherical_operators(source; accuracy_order, N, R, p=2, mode=FastMode(),
                        atol=nothing, snap_factor=64.0, build_matrix=:probe,
                        custom_stencil_cols=nothing, return_canonical=false,
                        target_eltype=nothing)

Construct folded spherical-symmetry SBP operators on `[0, R]` from a Cartesian first
derivative operator on `[-R, R]`.

Implementation details:
- construction is performed on a canonical grid with `Δr = 1` and `Rational{BigInt}`
  arithmetic to support exact row repair;
- optional scaling to simulation coordinates is applied at the end.
"""
function spherical_operators(source;
                             accuracy_order,
                             N,
                             R,
                             p::Int = 2,
                             mode = FastMode(),
                             atol = nothing,
                             snap_factor::Float64 = 64.0,
                             build_matrix::Symbol = :probe,
                             custom_stencil_cols::Union{Nothing, Vector{Int}} = nothing,
                             return_canonical::Bool = false,
                             target_eltype::Union{Nothing, Type} = nothing)
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    Nint = Int(N)
    Nint > 0 || throw(ArgumentError("`N` must be positive."))

    # Canonical construction grid: spacing = 1, domain [-N, N].
    R_canonical = big(Nint) // 1
    Dfull, xfull, Gfull, Hfull = _build_full_grid_objects(
                                                           source;
                                                           accuracy_order = Int(accuracy_order),
                                                           N = Nint,
                                                           R = R_canonical,
                                                           mode = mode,
                                                           build_matrix = build_matrix
                                                          )

    T = eltype(xfull)
    atol_construct = _resolve_atol(T, nothing)

    r, Rop, Eeven, Eodd = _build_folding_operators(xfull; atol = atol_construct)
    Nh = length(r)

    Geven = sparse(Rop * Gfull * Eeven)
    Godd = sparse(Rop * Gfull * Eodd)
    snap_sparse!(Geven; snap_factor = snap_factor)
    snap_sparse!(Godd; snap_factor = snap_factor)

    half_factor = convert(T, 1) / convert(T, 2)
    Hcart_half = sparse(half_factor * (transpose(Eeven) * Hfull * Eeven))
    metric = spdiagm(0 => r .^ p)
    H = sparse(Hcart_half * metric)
    snap_sparse!(H; snap_factor = snap_factor)

    B = spzeros(T, Nh, Nh)
    B[end, end] = r[end]^p
    snap_sparse!(B; snap_factor = snap_factor)

    Hdiag = _extract_diagonal(H; atol = atol_construct)
    Bdiag = fill(zero(T), Nh)
    Bdiag[end] = B[end, end]

    repair_info = _repair_even_gradient_first_column!(
                                                       Geven,
                                                       Dfull,
                                                       Gfull,
                                                       xfull,
                                                       Hdiag,
                                                       Bdiag,
                                                       r;
                                                       p = p,
                                                       atol = atol_construct,
                                                       custom_stencil_cols = custom_stencil_cols
                                                      )

    remaining_corrupted = _rows_with_nonzero_first_column(Geven; atol = atol_construct)
    isempty(remaining_corrupted) ||
        throw(ArgumentError("First column cleanup failed; nonzero rows remain: $remaining_corrupted"))

    closure_pattern = _closure_diagnostics(Geven)
    closure_from_operator = _boundary_closure_width_from_operator(Dfull)
    right_closure = isnothing(closure_from_operator) ?
                    closure_pattern.closure_width_right :
                    Int(closure_from_operator)

    _assert_global_even_interior_accuracy(
                                          Geven,
                                          r,
                                          repair_info.interior_accuracy,
                                          repair_info.rows_to_solve,
                                          right_closure;
                                          atol = atol_construct
                                         )

    RHS = sparse(B - transpose(Geven) * H)
    D = spzeros(T, Nh, Nh)
    _set_divergence_rows!(D, RHS, Hdiag)
    _set_origin_row!(D, Godd, p)
    snap_sparse!(D; snap_factor = snap_factor)

    closure_width = isnothing(closure_from_operator) ?
                    closure_pattern.closure_width_right :
                    max(closure_pattern.closure_width_right, closure_from_operator)

    ops_canonical = SphericalOperators(
                                       r,
                                       H,
                                       B,
                                       Geven,
                                       Godd,
                                       D,
                                       closure_width,
                                       repair_info.interior_accuracy,
                                       p,
                                       convert(T, R_canonical),
                                       source,
                                       mode,
                                       atol_construct,
                                       snap_factor,
                                       build_matrix,
                                       length(xfull),
                                       Nh
                                      )

    return_canonical && return ops_canonical

    Tout = isnothing(target_eltype) ? _default_scale_eltype(R) : target_eltype
    Tout <: Real || throw(ArgumentError("`target_eltype` must be a subtype of `Real`."))

    return scale_spherical_operators(
                                     ops_canonical,
                                     R;
                                     target_eltype = Tout,
                                     atol = atol
                                    )
end

@inline apply_even_gradient(ops::SphericalOperators, phi) = ops.Geven * phi
@inline apply_odd_derivative(ops::SphericalOperators, u) = ops.Godd * u
@inline apply_divergence(ops::SphericalOperators, u) = ops.D * u

function enforce_odd!(u::AbstractVector)
    isempty(u) && return u
    u[1] = zero(eltype(u))
    return u
end

function _default_odd_tol(::Type{T}) where {T <: AbstractFloat}
    return sqrt(eps(T))
end

function _default_odd_tol(::Type{T}) where {T}
    return zero(T)
end

function check_odd(u::AbstractVector; tol = _default_odd_tol(eltype(u)))
    isempty(u) && return true
    return abs(u[1]) <= tol
end
