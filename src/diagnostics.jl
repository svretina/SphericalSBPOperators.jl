function _tofloat(x)
    return Float64(x)
end

function _tofloat_vec(v)
    return Float64.(v)
end

function _maxabs_float(v)
    isempty(v) && return 0.0
    m = 0.0
    @inbounds for x in v
        ax = abs(_tofloat(x))
        if ax > m
            m = ax
        end
    end
    return m
end

function _maxabs_sparse_float(A::SparseMatrixCSC)
    return _maxabs_float(findnz(A)[3])
end

function _diag_and_offdiag_max_float(A::SparseMatrixCSC)
    n, m = size(A)
    n == m || throw(DimensionMismatch("Expected square matrix."))
    diagv = zeros(Float64, n)
    offdiag_max = 0.0
    I, J, V = findnz(A)
    @inbounds for k in eachindex(V)
        i = I[k]
        j = J[k]
        v = _tofloat(V[k])
        if i == j
            diagv[i] = v
        else
            av = abs(v)
            if av > offdiag_max
                offdiag_max = av
            end
        end
    end
    return diagv, offdiag_max
end

function _maxabs_sparse_rows_float(A::SparseMatrixCSC, row_mask::AbstractVector{Bool})
    I, _, V = findnz(A)
    m = 0.0
    @inbounds for k in eachindex(V)
        if row_mask[I[k]]
            av = abs(_tofloat(V[k]))
            if av > m
                m = av
            end
        end
    end
    return m
end

function _argmax_sparse_float(A::SparseMatrixCSC; row_mask::Union{Nothing, AbstractVector{Bool}} = nothing)
    I, J, V = findnz(A)
    best = 0.0
    best_i = 0
    best_j = 0
    best_val = 0.0
    @inbounds for k in eachindex(V)
        i = I[k]
        row_mask !== nothing && !row_mask[i] && continue
        v = _tofloat(V[k])
        av = abs(v)
        if av > best
            best = av
            best_i = i
            best_j = J[k]
            best_val = v
        end
    end
    return (i = best_i, j = best_j, value = best_val, abs_value = best)
end

function _safe_max_and_arg(err::Vector{Float64}, idx::Vector{Int})
    if isempty(idx)
        return (max_error = NaN, argmax_index = 0, argmax_r = NaN)
    end
    local_vals = err[idx]
    local_max, local_arg = findmax(local_vals)
    i = idx[local_arg]
    return (max_error = local_max, argmax_index = i)
end

function _region_indices(Nh::Int, region_points::Int)
    np = max(1, region_points)
    idx_origin = collect(1:min(np, Nh))

    center = cld(Nh, 2)
    half = fld(np, 2)
    mid_start = max(1, center - half)
    mid_end = min(Nh, mid_start + np - 1)
    idx_mid = collect(mid_start:mid_end)

    bstart = max(1, Nh - np + 1)
    idx_boundary = collect(bstart:Nh)
    return idx_origin, idx_mid, idx_boundary
end

function _region_max(err::Vector{Float64}, idx::Vector{Int})
    isempty(idx) && return (max_error = NaN, argmax_index = 0)
    vals = err[idx]
    vmax, arg = findmax(vals)
    return (max_error = vmax, argmax_index = idx[arg])
end

function _pointwise_error_summary(err::Vector{Float64},
                                  r::Vector{Float64},
                                  idx_safe::Vector{Int},
                                  idx_origin::Vector{Int},
                                  idx_mid::Vector{Int},
                                  idx_boundary::Vector{Int})
    vmax, arg = findmax(err)
    safe = _safe_max_and_arg(err, idx_safe)
    origin = _region_max(err, idx_origin)
    mid = _region_max(err, idx_mid)
    boundary = _region_max(err, idx_boundary)
    return (
            max_error = vmax,
            argmax_index = arg,
            argmax_r = r[arg],
            safe_max_error = safe.max_error,
            safe_argmax_index = safe.argmax_index,
            safe_argmax_r = safe.argmax_index == 0 ? NaN : r[safe.argmax_index],
            near_origin_max_error = origin.max_error,
            near_origin_argmax_index = origin.argmax_index,
            mid_max_error = mid.max_error,
            mid_argmax_index = mid.argmax_index,
            near_boundary_max_error = boundary.max_error,
            near_boundary_argmax_index = boundary.argmax_index
           )
end

function _table_rows(r::Vector{Float64},
                     numerical::Vector{Float64},
                     exact::Vector{Float64},
                     err::Vector{Float64},
                     table_points::Int)
    n = length(r)
    np = max(1, table_points)
    idx = unique(vcat(collect(1:min(np, n)), collect(max(1, n - np + 1):n)))
    rows = NamedTuple[]
    for i in idx
        push!(
              rows,
              (
               i = i,
               r = r[i],
               numerical = numerical[i],
               exact = exact[i],
               error = err[i]
              )
             )
    end
    return rows
end

function _print_table(title::AbstractString, rows)
    println(title)
    println("  i         r                  numerical            exact                error")
    for row in rows
        println(
                "  ",
                lpad(row.i, 3),
                "  ",
                lpad(string(round(row.r, sigdigits = 10)), 14),
                "  ",
                lpad(string(round(row.numerical, sigdigits = 10)), 18),
                "  ",
                lpad(string(round(row.exact, sigdigits = 10)), 18),
                "  ",
                lpad(string(round(row.error, sigdigits = 6)), 12)
               )
    end
end

"""
    interpret_diagnostics(diag; tol=1e-10, poly_tol=1e-8)

Interpret a diagnostics report and produce concise conclusions about likely error
sources (folding maps, closure effects, operator-family expectations, origin handling).
"""
function interpret_diagnostics(diag; tol::Float64 = 1e-10, poly_tol::Float64 = 1e-8)
    conclusions = String[]
    flags = Dict{Symbol, Bool}()

    grid_bad = !(diag.grid.symmetry_ok && diag.grid.uniform_ok && diag.grid.r0_ok && diag.grid.rend_ok)
    flags[:grid_issue] = grid_bad
    if grid_bad
        push!(
              conclusions,
              "Grid symmetry/uniformity checks failed. Polynomial exactness expectations are unreliable until grid construction is fixed."
             )
    end

    extension_fail = !(diag.folding.rop_ok && diag.folding.even_even_ok && diag.folding.odd_signed_ok && diag.folding.odd_matches_xk_for_odd_k_ok)
    operator_mismatch = (diag.operator_consistency.geven_diff_max > tol ||
                         diag.operator_consistency.godd_diff_max > tol)
    flags[:folding_issue] = extension_fail || operator_mismatch
    if flags[:folding_issue]
        push!(
              conclusions,
              "Folding/extension consistency checks indicate a map or pairing issue (Rop/Eeven/Eodd or tolerance/permutation/sign)."
             )
    end

    if !diag.polynomial.skipped
        full_safe = diag.fullgrid_comparison.max_full_vs_exact_safe
        fold_safe = diag.polynomial.gradient.global_safe_max

        if full_safe > poly_tol
            flags[:underlying_full_operator_issue] = true
            push!(
                  conclusions,
                  "The full-grid derivative already shows non-negligible safe-interior polynomial error. This is likely operator-family behavior/expectation mismatch, not folding."
                 )
        else
            flags[:underlying_full_operator_issue] = false
        end

        if (full_safe <= poly_tol) && (fold_safe > max(poly_tol, 10 * full_safe))
            flags[:folding_degrades_accuracy] = true
            push!(
                  conclusions,
                  "Folding degrades safe-interior accuracy relative to the full-grid operator. Inspect pairing and extension maps."
                 )
        else
            flags[:folding_degrades_accuracy] = false
        end

        if fold_safe <= poly_tol && diag.polynomial.gradient.global_max > 50 * max(poly_tol, fold_safe)
            flags[:boundary_closure_dominated] = true
            push!(
                  conclusions,
                  "Gradient errors are boundary-closure dominated. Increase closure exclusion when quoting interior accuracy."
                 )
        else
            flags[:boundary_closure_dominated] = false
        end

        div_worst = diag.polynomial.divergence.worst_entry
        if div_worst !== nothing
            origin_dom = (div_worst.argmax_index == 1) ||
                         (div_worst.near_origin_max_error > 10 * max(poly_tol, div_worst.mid_max_error, div_worst.near_boundary_max_error))
            boundary_dom = div_worst.near_boundary_max_error > 10 * max(poly_tol, div_worst.mid_max_error, div_worst.near_origin_max_error)

            flags[:divergence_origin_dominated] = origin_dom
            flags[:divergence_boundary_dominated] = boundary_dom
            if origin_dom
                push!(
                      conclusions,
                      "Divergence error is origin-dominated; verify `(Du)(0)=(p+1)u'(0)`, parity of test input, and `Godd` origin row."
                     )
            elseif boundary_dom
                push!(
                      conclusions,
                      "Divergence error is far-boundary dominated, consistent with closure effects near r=R."
                     )
            end
        else
            flags[:divergence_origin_dominated] = false
            flags[:divergence_boundary_dominated] = false
        end
    else
        flags[:underlying_full_operator_issue] = false
        flags[:folding_degrades_accuracy] = false
        flags[:boundary_closure_dominated] = false
        flags[:divergence_origin_dominated] = false
        flags[:divergence_boundary_dominated] = false
    end

    if isempty(conclusions)
        push!(
              conclusions,
              "No structural inconsistency detected; observed errors are consistent with expected SBP closure/operator-family behavior."
             )
    end

    return (conclusions = conclusions, flags = flags)
end

"""
    diagnose(ops, report=nothing; Ktest=8, region_points=10, table_points=6,
             tol=nothing, verbose=true)

Run a detailed diagnostic suite for folded spherical SBP operators.

Returns a `NamedTuple` with:
- grid sanity (`grid`)
- folding map checks (`folding`)
- operator consistency checks (`operator_consistency`)
- closure diagnostics (`closure`)
- localized polynomial error diagnostics (`polynomial`)
- full-grid baseline comparison (`fullgrid_comparison`)
- SBP residual localization (`sbp`)
- interpreted conclusions (`interpretation`)
"""
function diagnose(ops::SphericalOperators,
                  report = nothing;
                  Ktest::Int = 8,
                  region_points::Int = 10,
                  table_points::Int = 6,
                  tol = nothing,
                  verbose::Bool = true)
    if report === nothing
        report = validate(ops; verbose = false)
    end

    atol_diag = tol === nothing ? max(1e-12, abs(_tofloat(ops.atol))) : _tofloat(tol)
    maxdeg = hasproperty(report, :diagnostics) && hasproperty(report.diagnostics, :max_monomial_degree) ?
             Int(report.diagnostics.max_monomial_degree) : ops.accuracy_order

    Nhalf = ops.Nh - 1
    Dfull, xfull, Gfull, Hfull = _build_full_grid_objects(
                                                          ops.source;
                                                          accuracy_order = ops.accuracy_order,
                                                          N = Nhalf,
                                                          R = ops.R,
                                                          mode = ops.mode,
                                                          build_matrix = ops.build_matrix
                                                         )

    atolT = _resolve_atol(eltype(xfull), ops.atol)
    r_fold, Rop, Eeven, Eodd = _build_folding_operators(xfull; atol = atolT)

    xfullf = _tofloat_vec(xfull)
    r = _tofloat_vec(ops.r)
    rf = _tofloat_vec(r_fold)
    M = length(xfullf)
    Nh = length(r)

    symmetry_error = _maxabs_float(xfullf .+ reverse(xfullf))
    dx = diff(xfullf)
    dx_min = isempty(dx) ? NaN : minimum(dx)
    dx_max = isempty(dx) ? NaN : maximum(dx)
    dx_mean = isempty(dx) ? NaN : sum(dx) / length(dx)
    dx_dev = isempty(dx) ? NaN : _maxabs_float(dx .- dx_mean)

    r_match_error = Nh == length(rf) ? _maxabs_float(r .- rf) : Inf
    r0_error = Nh > 0 ? abs(r[1]) : Inf
    rend_error = Nh > 0 ? abs(r[end] - _tofloat(ops.R)) : Inf

    symmetry_ok = symmetry_error <= atol_diag
    uniform_ok = isnan(dx_dev) || dx_dev <= atol_diag
    r0_ok = r0_error <= atol_diag
    rend_ok = rend_error <= atol_diag
    grid_ok = symmetry_ok && uniform_ok && r0_ok && rend_ok

    grid_diag = (
                 M = M,
                 Nh = Nh,
                 xfirst = xfullf[1],
                 xlast = xfullf[end],
                 symmetry_error = symmetry_error,
                 symmetry_ok = symmetry_ok,
                 dx_min = dx_min,
                 dx_max = dx_max,
                 dx_mean = dx_mean,
                 dx_max_deviation = dx_dev,
                 uniform_ok = uniform_ok,
                 r0 = Nh > 0 ? r[1] : NaN,
                 r_end = Nh > 0 ? r[end] : NaN,
                 r0_error = r0_error,
                 r_end_error = rend_error,
                 r0_ok = r0_ok,
                 rend_ok = rend_ok,
                 r_match_error = r_match_error,
                 grid_ok = grid_ok,
                 atol_used = atol_diag
                )

    # Folding map sanity
    row_counts = zeros(Int, Nh)
    selected_idx = fill(0, Nh)
    value_ok = true
    Irop, Jrop, Vrop = findnz(Rop)
    for k in eachindex(Vrop)
        i = Irop[k]
        row_counts[i] += 1
        selected_idx[i] = Jrop[k]
        value_ok &= abs(_tofloat(Vrop[k]) - 1.0) <= atol_diag
    end
    one_per_row = all(==(1), row_counts)
    expected_half_idx = sort(findall(x -> x >= -atol_diag, xfullf), by = i -> xfullf[i])
    index_match = one_per_row && selected_idx == expected_half_idx

    coord_match_max = 0.0
    nonnegative_ok = true
    coord_ok = true
    if one_per_row
        for ih in 1:Nh
            jf = selected_idx[ih]
            xj = xfullf[jf]
            nonnegative_ok &= xj >= -atol_diag
            err = abs(xj - r[ih])
            if err > coord_match_max
                coord_match_max = err
            end
            coord_ok &= err <= atol_diag
        end
    else
        coord_ok = false
    end
    rop_ok = one_per_row && value_ok && index_match && nonnegative_ok && coord_ok

    even_extension = NamedTuple[]
    odd_extension = NamedTuple[]
    even_even_errs = Float64[]
    odd_signed_errs = Float64[]
    odd_xk_odd_errs = Float64[]

    for k in 0:Ktest
        f_half = r_fold .^ k
        f_full_exact = xfull .^ k
        f_full_even = Eeven * f_half
        err_even_vs_xk = _maxabs_float(f_full_even .- f_full_exact)
        push!(
              even_extension,
              (
               degree = k,
               expected_match_xk = iseven(k),
               max_error_vs_xk = err_even_vs_xk
              )
             )
        if iseven(k)
            push!(even_even_errs, err_even_vs_xk)
        end

        g_half = r_fold .^ k
        g_full_odd = Eodd * g_half
        signed_abs = similar(xfull)
        for i in eachindex(xfull)
            xi = xfull[i]
            signed_abs[i] = ifelse(xi > zero(xi), abs(xi)^k,
                                   ifelse(xi < zero(xi), -abs(xi)^k, zero(xi)))
        end
        err_odd_vs_signed = _maxabs_float(g_full_odd .- signed_abs)
        err_odd_vs_xk = _maxabs_float(g_full_odd .- f_full_exact)
        push!(
              odd_extension,
              (
               degree = k,
               expected_match_xk = isodd(k),
               max_error_vs_xk = err_odd_vs_xk,
               max_error_vs_signabs = err_odd_vs_signed
              )
             )
        push!(odd_signed_errs, err_odd_vs_signed)
        if isodd(k)
            push!(odd_xk_odd_errs, err_odd_vs_xk)
        end
    end

    even_even_ok = isempty(even_even_errs) ? true : maximum(even_even_errs) <= atol_diag
    odd_signed_ok = isempty(odd_signed_errs) ? true : maximum(odd_signed_errs) <= atol_diag
    odd_matches_xk_for_odd_k_ok = isempty(odd_xk_odd_errs) ? true : maximum(odd_xk_odd_errs) <= atol_diag

    folding_diag = (
                   rop_ok = rop_ok,
                   rop_one_nonzero_per_row = one_per_row,
                   rop_values_ok = value_ok,
                   rop_index_match = index_match,
                   rop_nonnegative_ok = nonnegative_ok,
                   rop_coordinate_match_ok = coord_ok,
                   rop_coordinate_match_max = coord_match_max,
                   even_extension = even_extension,
                   odd_extension = odd_extension,
                   even_even_ok = even_even_ok,
                   odd_signed_ok = odd_signed_ok,
                   odd_matches_xk_for_odd_k_ok = odd_matches_xk_for_odd_k_ok
                  )

    # Operator consistency checks
    Gfold_even_direct = sparse(Rop * Gfull * Eeven)
    Gfold_odd_direct = sparse(Rop * Gfull * Eodd)
    snap_sparse!(Gfold_even_direct; snap_factor = ops.snap_factor)
    snap_sparse!(Gfold_odd_direct; snap_factor = ops.snap_factor)

    atol_ops = _resolve_atol(eltype(r_fold), ops.atol)
    repaired_rows = _rows_with_nonzero_first_column(Gfold_even_direct; atol = atol_ops)
    Geven_diff = sparse(ops.Geven - Gfold_even_direct)
    for row in repaired_rows
        Geven_diff[row, :] .= zero(eltype(Geven_diff))
    end
    snap_sparse!(Geven_diff; snap_factor = ops.snap_factor)

    Geven_diff_max = _maxabs_sparse_float(Geven_diff)
    Geven_diff_max_all_rows = _maxabs_sparse_float(ops.Geven - Gfold_even_direct)
    Godd_diff_max = _maxabs_sparse_float(ops.Godd - Gfold_odd_direct)

    half_factor = convert(eltype(xfull), 1) / convert(eltype(xfull), 2)
    Hcart_half_direct = sparse(half_factor * (transpose(Eeven) * Hfull * Eeven))
    Hdiag, Hoffdiag_max = _diag_and_offdiag_max_float(ops.H)
    Hcart_diag = _diag_and_offdiag_max_float(Hcart_half_direct)[1]

    mass_ratio_errors = Float64[]
    for i in 2:Nh
        rp = r[i]^ops.p
        rp == 0.0 && continue
        inferred = Hdiag[i] / rp
        push!(mass_ratio_errors, abs(inferred - Hcart_diag[i]))
    end
    mass_diag_ratio_max = isempty(mass_ratio_errors) ? 0.0 : maximum(mass_ratio_errors)

    operator_diag = (
                    geven_diff_max = Geven_diff_max,
                    geven_diff_max_all_rows = Geven_diff_max_all_rows,
                    geven_repaired_rows = repaired_rows,
                    godd_diff_max = Godd_diff_max,
                    H_offdiag_max = Hoffdiag_max,
                    Hcart_diag_ratio_max_error = mass_diag_ratio_max
                   )

    # Closure diagnostics
    closure_info = _closure_diagnostics(ops.Geven)
    closure_right_from_operator = _boundary_closure_width_from_operator(Dfull)
    closure_right_used = isnothing(closure_right_from_operator) ?
                         closure_info.closure_width_right :
                         max(closure_info.closure_width_right, closure_right_from_operator)
    safe_left = max(closure_info.closure_width_left, closure_right_used + 1)
    safe_start = 1 + safe_left
    safe_end = Nh - closure_right_used
    idx_safe = safe_start <= safe_end ? collect(safe_start:safe_end) : Int[]
    safe_mask = falses(Nh)
    for i in idx_safe
        safe_mask[i] = true
    end
    closure_diag = (
                    row_nnz = closure_info.row_nnz,
                    interior_mode = closure_info.interior_mode,
                    closure_width_left = safe_left,
                    closure_width_right = closure_right_used,
                    closure_width_right_pattern = closure_info.closure_width_right,
                    closure_width_right_operator = closure_right_from_operator,
                    idx_safe = idx_safe,
                    safe_start = safe_start,
                    safe_end = safe_end,
                    safe_count = length(idx_safe)
                   )

    idx_origin, idx_mid, idx_boundary = _region_indices(Nh, region_points)

    polynomial_skipped = !grid_ok
    gradient_entries = NamedTuple[]
    divergence_entries = NamedTuple[]
    grad_worst_table = NamedTuple[]
    div_worst_table = NamedTuple[]
    grad_worst_entry = nothing
    div_worst_entry = nothing
    grad_global_max = NaN
    grad_safe_max = NaN
    div_global_max = NaN
    div_safe_max = NaN

    full_comp_entries = NamedTuple[]
    full_max_full_vs_exact = NaN
    full_max_full_vs_exact_safe = NaN
    full_max_fold_vs_exact = NaN
    full_max_fold_vs_exact_safe = NaN
    full_max_fold_vs_full = NaN

    if !polynomial_skipped
        grad_degrees = collect(0:2:maxdeg)
        div_max_degree = maxdeg - ops.p
        div_degrees = div_max_degree >= 1 ? collect(1:2:div_max_degree) : Int[]

        grad_global_max = 0.0
        grad_safe_max = 0.0
        for deg in grad_degrees
            phi = ops.r .^ deg
            numerical = _tofloat_vec(ops.Geven * phi)
            exact = deg == 0 ? zeros(Float64, Nh) : _tofloat_vec(convert(eltype(ops.r), deg) .* (ops.r .^ (deg - 1)))
            err = abs.(numerical .- exact)
            stats = _pointwise_error_summary(err, r, idx_safe, idx_origin, idx_mid, idx_boundary)
            entry = (
                     degree = deg,
                     numerical = numerical,
                     exact = exact,
                     error = err,
                     max_error = stats.max_error,
                     argmax_index = stats.argmax_index,
                     argmax_r = stats.argmax_r,
                     safe_max_error = stats.safe_max_error,
                     safe_argmax_index = stats.safe_argmax_index,
                     safe_argmax_r = stats.safe_argmax_r,
                     near_origin_max_error = stats.near_origin_max_error,
                     mid_max_error = stats.mid_max_error,
                     near_boundary_max_error = stats.near_boundary_max_error
                    )
            push!(gradient_entries, entry)
            if entry.max_error > grad_global_max
                grad_global_max = entry.max_error
                grad_worst_entry = entry
                grad_worst_table = _table_rows(r, numerical, exact, err, table_points)
            end
            if !isnan(entry.safe_max_error)
                grad_safe_max = max(grad_safe_max, entry.safe_max_error)
            end
        end

        div_global_max = 0.0
        div_safe_max = 0.0
        for deg in div_degrees
            u = ops.r .^ deg
            numerical = _tofloat_vec(ops.D * u)
            exact = _tofloat_vec(convert(eltype(ops.r), ops.p + deg) .* (ops.r .^ (deg - 1)))
            err = abs.(numerical .- exact)
            stats = _pointwise_error_summary(err, r, idx_safe, idx_origin, idx_mid, idx_boundary)
            entry = (
                     degree = deg,
                     numerical = numerical,
                     exact = exact,
                     error = err,
                     max_error = stats.max_error,
                     argmax_index = stats.argmax_index,
                     argmax_r = stats.argmax_r,
                     safe_max_error = stats.safe_max_error,
                     safe_argmax_index = stats.safe_argmax_index,
                     safe_argmax_r = stats.safe_argmax_r,
                     near_origin_max_error = stats.near_origin_max_error,
                     mid_max_error = stats.mid_max_error,
                     near_boundary_max_error = stats.near_boundary_max_error
                    )
            push!(divergence_entries, entry)
            if entry.max_error > div_global_max
                div_global_max = entry.max_error
                div_worst_entry = entry
                div_worst_table = _table_rows(r, numerical, exact, err, table_points)
            end
            if !isnan(entry.safe_max_error)
                div_safe_max = max(div_safe_max, entry.safe_max_error)
            end
        end

        # Full-grid baseline comparison for even monomials
        full_max_full_vs_exact = 0.0
        full_max_full_vs_exact_safe = 0.0
        full_max_fold_vs_exact = 0.0
        full_max_fold_vs_exact_safe = 0.0
        full_max_fold_vs_full = 0.0

        for deg in grad_degrees
            phi_half = ops.r .^ deg
            phi_full_exact = xfull .^ deg
            dphi_full_num = Gfull * phi_full_exact
            dphi_half_from_full = Rop * dphi_full_num
            dphi_half_folded = ops.Geven * phi_half
            dphi_exact = deg == 0 ? fill(zero(eltype(ops.r)), Nh) : convert(eltype(ops.r), deg) .* (ops.r .^ (deg - 1))

            e_full_exact = abs.(_tofloat_vec(dphi_half_from_full) .- _tofloat_vec(dphi_exact))
            e_fold_exact = abs.(_tofloat_vec(dphi_half_folded) .- _tofloat_vec(dphi_exact))
            e_fold_full = abs.(_tofloat_vec(dphi_half_folded) .- _tofloat_vec(dphi_half_from_full))

            safe_full = _safe_max_and_arg(e_full_exact, idx_safe).max_error
            safe_fold = _safe_max_and_arg(e_fold_exact, idx_safe).max_error
            safe_fold_full = _safe_max_and_arg(e_fold_full, idx_safe).max_error

            push!(
                  full_comp_entries,
                  (
                   degree = deg,
                   max_full_vs_exact = _maxabs_float(e_full_exact),
                   max_full_vs_exact_safe = safe_full,
                   max_fold_vs_exact = _maxabs_float(e_fold_exact),
                   max_fold_vs_exact_safe = safe_fold,
                   max_fold_vs_full = _maxabs_float(e_fold_full),
                   max_fold_vs_full_safe = safe_fold_full
                  )
                 )

            full_max_full_vs_exact = max(full_max_full_vs_exact, _maxabs_float(e_full_exact))
            full_max_fold_vs_exact = max(full_max_fold_vs_exact, _maxabs_float(e_fold_exact))
            full_max_fold_vs_full = max(full_max_fold_vs_full, _maxabs_float(e_fold_full))
            if !isnan(safe_full)
                full_max_full_vs_exact_safe = max(full_max_full_vs_exact_safe, safe_full)
            end
            if !isnan(safe_fold)
                full_max_fold_vs_exact_safe = max(full_max_fold_vs_exact_safe, safe_fold)
            end
        end
    end

    polynomial_diag = (
                       skipped = polynomial_skipped,
                       reason = polynomial_skipped ? "grid_not_symmetric_or_uniform" : "",
                       regions = (
                                  near_origin = idx_origin,
                                  mid = idx_mid,
                                  near_boundary = idx_boundary
                                 ),
                       gradient = (
                                   entries = gradient_entries,
                                   global_max = grad_global_max,
                                   global_safe_max = grad_safe_max,
                                   worst_entry = grad_worst_entry,
                                   worst_table = grad_worst_table
                                  ),
                       divergence = (
                                     entries = divergence_entries,
                                     global_max = div_global_max,
                                     global_safe_max = div_safe_max,
                                     worst_entry = div_worst_entry,
                                     worst_table = div_worst_table
                                    )
                      )

    fullgrid_diag = (
                     entries = full_comp_entries,
                     max_full_vs_exact = full_max_full_vs_exact,
                     max_full_vs_exact_safe = full_max_full_vs_exact_safe,
                     max_fold_vs_exact = full_max_fold_vs_exact,
                     max_fold_vs_exact_safe = full_max_fold_vs_exact_safe,
                     max_fold_vs_full = full_max_fold_vs_full
                    )

    # SBP residual diagnostics
    Rsbp = sparse(ops.H * ops.D + transpose(ops.Geven) * ops.H - ops.B)
    mask_no_origin = trues(Nh)
    if Nh >= 1
        mask_no_origin[1] = false
    end
    mask_safe = copy(safe_mask)
    if Nh >= 1
        mask_safe[1] = false
    end
    sbp_no_origin = _maxabs_sparse_rows_float(Rsbp, mask_no_origin)
    sbp_safe = _maxabs_sparse_rows_float(Rsbp, mask_safe)
    largest_no_origin = _argmax_sparse_float(Rsbp; row_mask = mask_no_origin)
    largest_all = _argmax_sparse_float(Rsbp)
    sbp_diag = (
                max_no_origin = sbp_no_origin,
                max_safe_rows = sbp_safe,
                largest_entry_no_origin = (
                                           i = largest_no_origin.i,
                                           j = largest_no_origin.j,
                                           value = largest_no_origin.value,
                                           abs_value = largest_no_origin.abs_value,
                                           ri = largest_no_origin.i == 0 ? NaN : r[largest_no_origin.i],
                                           rj = largest_no_origin.j == 0 ? NaN : r[largest_no_origin.j]
                                          ),
                largest_entry_all = (
                                     i = largest_all.i,
                                     j = largest_all.j,
                                     value = largest_all.value,
                                     abs_value = largest_all.abs_value,
                                     ri = largest_all.i == 0 ? NaN : r[largest_all.i],
                                     rj = largest_all.j == 0 ? NaN : r[largest_all.j]
                                    )
               )

    base_diag = (
                 grid = grid_diag,
                 folding = folding_diag,
                 operator_consistency = operator_diag,
                 closure = closure_diag,
                 polynomial = polynomial_diag,
                 fullgrid_comparison = fullgrid_diag,
                 sbp = sbp_diag
                )
    interpretation = interpret_diagnostics(base_diag)

    diag = merge(base_diag, (interpretation = interpretation,))

    if verbose
        println("Diagnostics summary")
        println("  Grid: M=$(diag.grid.M), Nh=$(diag.grid.Nh), symmetry_error=$(diag.grid.symmetry_error), uniform_dev=$(diag.grid.dx_max_deviation)")
        println("  Grid checks: symmetry_ok=$(diag.grid.symmetry_ok), uniform_ok=$(diag.grid.uniform_ok), r0_ok=$(diag.grid.r0_ok), rend_ok=$(diag.grid.rend_ok)")
        println("  Folding: rop_ok=$(diag.folding.rop_ok), even_even_ok=$(diag.folding.even_even_ok), odd_signed_ok=$(diag.folding.odd_signed_ok)")
        println("  Consistency: max|Geven-direct|=$(diag.operator_consistency.geven_diff_max), max|Godd-direct|=$(diag.operator_consistency.godd_diff_max)")
        println("  Closure: left=$(diag.closure.closure_width_left), right=$(diag.closure.closure_width_right), safe_count=$(diag.closure.safe_count)")
        if !diag.polynomial.skipped
            println("  Gradient error: global=$(diag.polynomial.gradient.global_max), safe=$(diag.polynomial.gradient.global_safe_max)")
            println("  Divergence error: global=$(diag.polynomial.divergence.global_max), safe=$(diag.polynomial.divergence.global_safe_max)")
            if diag.polynomial.gradient.worst_entry !== nothing
                _print_table("  Gradient worst-case degree = $(diag.polynomial.gradient.worst_entry.degree)", diag.polynomial.gradient.worst_table)
            end
            if diag.polynomial.divergence.worst_entry !== nothing
                _print_table("  Divergence worst-case degree = $(diag.polynomial.divergence.worst_entry.degree)", diag.polynomial.divergence.worst_table)
            end
        else
            println("  Polynomial diagnostics skipped: $(diag.polynomial.reason)")
        end
        println("  SBP residual: max_no_origin=$(diag.sbp.max_no_origin), max_safe_rows=$(diag.sbp.max_safe_rows)")
        println("  Interpretation:")
        for c in diag.interpretation.conclusions
            println("    - ", c)
        end
    end

    return diag
end
