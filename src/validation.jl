function _row_structure(G::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
    n, m = size(G)
    cols_by_row = [Int[] for _ in 1:n]
    vals_by_row = [T[] for _ in 1:n]
    @inbounds for j in 1:m
        for ptr in G.colptr[j]:(G.colptr[j + 1] - 1)
            i = G.rowval[ptr]
            push!(cols_by_row[i], j)
            push!(vals_by_row[i], G.nzval[ptr])
        end
    end
    return cols_by_row, vals_by_row
end

@inline function _relative_pattern(cols::Vector{Int}, i::Int)
    return cols .- i
end

function _row_closure_flag(i::Int,
                           patterns::Vector{Vector{Int}},
                           values::AbstractVector{<:AbstractVector};
                           reference_row::Int,
                           coeff_sensitive::Bool,
                           tol_coeff::Float64)
    pattern_i = patterns[i]
    pattern_ref = patterns[reference_row]
    if pattern_i != pattern_ref
        return true
    end

    if !coeff_sensitive
        return false
    end

    vals_i = values[i]
    vals_ref = values[reference_row]
    length(vals_i) == length(vals_ref) || return true

    maxdiff = 0.0
    @inbounds for k in eachindex(vals_i)
        d = abs(Float64(vals_i[k]) - Float64(vals_ref[k]))
        if d > maxdiff
            maxdiff = d
        end
    end
    return maxdiff > tol_coeff
end

function _closure_diagnostics(G::SparseMatrixCSC{T, Ti};
                              coeff_sensitive::Bool = true,
                              tol_coeff::Union{Nothing, Float64} = nothing) where {T <: Real, Ti <: Integer}
    n = size(G, 1)
    if n == 0
        return (
                row_nnz = Int[],
                interior_mode = 0,
                reference_row = 1,
                reference_pattern = Int[],
                coeff_sensitive = coeff_sensitive,
                tol_coeff = 0.0,
                closure_width_left = 0,
                closure_width_right = 0,
                safe_start = 1,
                safe_end = 0,
                idx_safe = Int[]
               )
    end

    reference_row = clamp(fld(n, 2), 1, n)
    cols_by_row, vals_by_row = _row_structure(G)
    patterns = [_relative_pattern(cols_by_row[i], i) for i in 1:n]
    row_nnz = [length(cols_by_row[i]) for i in 1:n]
    reference_pattern = patterns[reference_row]

    ref_inf = isempty(vals_by_row[reference_row]) ? 0.0 : maximum(abs.(Float64.(vals_by_row[reference_row])))
    tol_coeff_val = isnothing(tol_coeff) ? (1e3 * eps(Float64) * max(1.0, ref_inf)) : tol_coeff

    left = 0
    for i in 1:n
        _row_closure_flag(
                          i,
                          patterns,
                          vals_by_row;
                          reference_row = reference_row,
                          coeff_sensitive = coeff_sensitive,
                          tol_coeff = tol_coeff_val
                         ) || break
        left += 1
    end

    right = 0
    for i in n:-1:1
        _row_closure_flag(
                          i,
                          patterns,
                          vals_by_row;
                          reference_row = reference_row,
                          coeff_sensitive = coeff_sensitive,
                          tol_coeff = tol_coeff_val
                         ) || break
        right += 1
    end

    safe_start = 1 + left
    safe_end = n - right
    idx_safe = safe_start <= safe_end ? collect(safe_start:safe_end) : Int[]

    return (
            row_nnz = row_nnz,
            interior_mode = length(reference_pattern),
            reference_row = reference_row,
            reference_pattern = reference_pattern,
            coeff_sensitive = coeff_sensitive,
            tol_coeff = tol_coeff_val,
            closure_width_left = min(left, max(0, n - 1)),
            closure_width_right = min(right, max(0, n - 1)),
            safe_start = safe_start,
            safe_end = safe_end,
            idx_safe = idx_safe
           )
end

"""
    _estimate_right_closure_width(G)

Estimate right boundary closure width from row-pattern mismatch (and optional
coefficient mismatch) relative to an interior reference row.
"""
function _estimate_right_closure_width(G::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
    diag = _closure_diagnostics(G)
    return diag.closure_width_right
end

function _maxabs_sparse(A::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
    vals = findnz(A)[3]
    isempty(vals) && return zero(T)
    return _maxabs(vals)
end

function _maxabs_sparse_no_origin(A::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
    I, _, V = findnz(A)
    m = zero(T)
    @inbounds for k in eachindex(V)
        if I[k] != 1
            av = abs(V[k])
            if av > m
                m = av
            end
        end
    end
    return m
end

function _safe_max(errors::AbstractVector, idx_safe::Vector{Int})
    isempty(errors) && return zero(eltype(errors))
    isempty(idx_safe) && return zero(eltype(errors))
    return _maxabs(view(errors, idx_safe))
end

function _near_boundary_max(errors::AbstractVector, points::Int)
    n = length(errors)
    n == 0 && return zero(eltype(errors))
    m = max(1, points)
    first_idx = max(1, n - m + 1)
    return _maxabs(view(errors, first_idx:n))
end

function _integral_monomial(R::T, q::Int) where {T <: Real}
    return R^(q + 1) / convert(T, q + 1)
end

"""
    validate(ops; max_monomial_degree=ops.accuracy_order, verbose=true)

Run validation checks for:
- weighted quadrature on monomials;
- gradient accuracy on even monomials;
- divergence accuracy on odd monomials;
- SBP residual norms (full and excluding origin row).

Returns a `NamedTuple`.
"""
function validate(ops::SphericalOperators;
                  max_monomial_degree = ops.accuracy_order,
                  verbose::Bool = true,
                  near_boundary_points::Int = 8)
    T = eltype(ops.r)
    Nh = length(ops.r)
    ones_h = fill(one(T), Nh)
    maxdeg = Int(max_monomial_degree)
    closure = _closure_diagnostics(ops.Geven)
    closure_right = max(closure.closure_width_right, ops.closure_width)
    # Divergence rows can be influenced by the modified near-origin gradient columns;
    # use a conservative left-safe offset tied to right closure width as well.
    safe_left = max(closure.closure_width_left, closure_right + 1)
    safe_start = 1 + safe_left
    safe_end = Nh - closure_right
    idx_safe = safe_start <= safe_end ? collect(safe_start:safe_end) : Int[]

    quadrature = NamedTuple[]
    for k in 0:maxdeg
        u = ops.r .^ k
        numerical = dot(u, ops.H * ones_h)
        exact = _integral_monomial(ops.R, k + ops.p)
        err = abs(numerical - exact)
        push!(quadrature, (degree = k, numerical = numerical, exact = exact, abs_error = err))
    end

    gradient_even = NamedTuple[]
    for deg in 0:2:maxdeg
        phi = ops.r .^ deg
        dphi_exact = deg == 0 ? fill(zero(T), Nh) : convert(T, deg) .* (ops.r .^ (deg - 1))
        dphi_num = ops.Geven * phi
        err = abs.(dphi_num .- dphi_exact)
        push!(
              gradient_even,
              (
               degree = deg,
               max_error = _maxabs(err),
               max_error_safe = _safe_max(err, idx_safe),
               interior_max_error = _safe_max(err, idx_safe),
               max_error_near_boundary = _near_boundary_max(err, near_boundary_points)
              )
             )
    end

    divergence_odd = NamedTuple[]
    max_odd_degree = maxdeg - ops.p
    if max_odd_degree >= 1
        for deg in 1:2:max_odd_degree
            u = ops.r .^ deg
            coeff = convert(T, ops.p + deg)
            div_exact = coeff .* (ops.r .^ (deg - 1))
            div_num = ops.D * u
            err = abs.(div_num .- div_exact)
            push!(
                  divergence_odd,
                  (
                   degree = deg,
                   max_error = _maxabs(err),
                   max_error_safe = _safe_max(err, idx_safe),
                   interior_max_error = _safe_max(err, idx_safe),
                   max_error_near_boundary = _near_boundary_max(err, near_boundary_points)
                  )
                 )
        end
    end

    R_sbp = sparse(ops.H * ops.D + transpose(ops.Geven) * ops.H - ops.B)
    sbp = (
           sbp_full = _maxabs_sparse(R_sbp),
           sbp_no_origin = _maxabs_sparse_no_origin(R_sbp)
          )

    r0 = ops.r[1]
    r0_ok = T <: AbstractFloat ? abs(r0) <= ops.atol : r0 == zero(T)
    diagnostics = (
                   M = ops.M_full,
                   Nh = ops.Nh,
                   r0 = r0,
                   r0_ok = r0_ok,
                   closure_width = closure_right,
                   closure_width_left = safe_left,
                   closure_width_right = closure_right,
                   closure_width_right_pattern = closure.closure_width_right,
                   closure_width_right_operator = ops.closure_width,
                   idx_safe = idx_safe,
                   safe_count = length(idx_safe),
                   near_boundary_points = near_boundary_points,
                   accuracy_order = ops.accuracy_order,
                   p = ops.p,
                   R = ops.R,
                   max_monomial_degree = maxdeg
                  )

    report = (
              quadrature = quadrature,
              gradient_even = gradient_even,
              divergence_odd = divergence_odd,
              sbp = sbp,
              diagnostics = diagnostics
             )

    if verbose
        println("Validation summary")
        println("  M_full = $(diagnostics.M), Nh = $(diagnostics.Nh), closure_left = $(diagnostics.closure_width_left), closure_right = $(diagnostics.closure_width_right), safe_count = $(diagnostics.safe_count)")
        println("  r[1] = $(diagnostics.r0), r0_ok = $(diagnostics.r0_ok)")
        println("  sbp_full = $(sbp.sbp_full), sbp_no_origin = $(sbp.sbp_no_origin)")
        println("\n  Quadrature moments (u=r^k):")
        println("    k    abs_error")
        for row in quadrature
            println("    ", lpad(row.degree, 2), "    ", row.abs_error)
        end

        println("\n  Gradient on even moments (phi=r^k):")
        println("    k    max_error               max_error_safe          max_error_near_boundary")
        for row in gradient_even
            println(
                    "    ",
                    lpad(row.degree, 2),
                    "    ",
                    lpad(string(row.max_error), 22),
                    "    ",
                    lpad(string(row.max_error_safe), 22),
                    "    ",
                    lpad(string(row.max_error_near_boundary), 22)
                   )
        end

        println("\n  Divergence on odd moments (u=r^k):")
        println("    k    max_error               max_error_safe          max_error_near_boundary")
        for row in divergence_odd
            println(
                    "    ",
                    lpad(row.degree, 2),
                    "    ",
                    lpad(string(row.max_error), 22),
                    "    ",
                    lpad(string(row.max_error_safe), 22),
                    "    ",
                    lpad(string(row.max_error_near_boundary), 22)
                   )
        end
    end

    return report
end
