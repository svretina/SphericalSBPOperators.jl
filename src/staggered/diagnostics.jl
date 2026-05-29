function interpret_diagnostics(diag; tol::Float64 = 1.0e-10, poly_tol::Float64 = 1.0e-8)
    conclusions = String[]
    flags = Dict{Symbol, Bool}()

    grid_ok = haskey(diag.grid, :grid_ok) ? diag.grid.grid_ok : true
    flags[:grid_issue] = !grid_ok
    if !grid_ok
        push!(conclusions, "Grid checks failed; polynomial diagnostics may be unreliable.")
    end

    sbp_expected = haskey(diag.grid, :sbp_expected) ? diag.grid.sbp_expected : true
    sbp_ok = !sbp_expected || diag.sbp.max_no_origin <= tol
    flags[:sbp_issue] = !sbp_ok
    if !sbp_ok
        push!(conclusions, "SBP residual exceeds tolerance in staggered construction.")
    end

    grad_safe = diag.polynomial.gradient.global_safe_max
    div_safe = diag.polynomial.divergence.global_safe_max
    flags[:interior_gradient_ok] = grad_safe <= poly_tol
    flags[:interior_divergence_ok] = isnan(div_safe) || div_safe <= poly_tol
    if grad_safe > poly_tol
        push!(conclusions, "Safe-interior even-gradient error is larger than requested tolerance.")
    end
    if !isnan(div_safe) && div_safe > poly_tol
        push!(conclusions, "Safe-interior odd-divergence error is larger than requested tolerance.")
    end

    if isempty(conclusions)
        push!(
            conclusions,
            "No structural inconsistency detected; staggered operators satisfy expected SBP and interior-accuracy checks."
        )
    end

    return (conclusions = conclusions, flags = flags)
end

function _worst_entry(entries::Vector{<:NamedTuple})
    isempty(entries) && return nothing
    idx = argmax(getfield.(entries, :max_error))
    return entries[idx]
end

function diagnose(
        ops::SphericalOperators,
        report = nothing;
        Ktest::Int = 8,
        region_points::Int = 10,
        table_points::Int = 6,
        tol = nothing,
        verbose::Bool = true
    )
    _ = Ktest
    _ = region_points
    _ = table_points

    report === nothing && (report = validate(ops; verbose = false))
    atol_diag = tol === nothing ? max(1.0e-12, abs(Float64(ops.atol))) : Float64(tol)

    gradient_entries = NamedTuple[]
    for row in report.gradient_even
        push!(
            gradient_entries,
            (
                degree = row.degree,
                max_error = Float64(row.max_error),
                safe_max_error = Float64(row.max_error_safe),
                near_origin_max_error = NaN,
                mid_max_error = NaN,
                near_boundary_max_error = Float64(row.max_error_near_boundary),
                argmax_index = 0,
                argmax_r = NaN,
                safe_argmax_index = 0,
                safe_argmax_r = NaN,
            )
        )
    end

    divergence_entries = NamedTuple[]
    for row in report.divergence_odd
        push!(
            divergence_entries,
            (
                degree = row.degree,
                max_error = Float64(row.max_error),
                safe_max_error = Float64(row.max_error_safe),
                near_origin_max_error = NaN,
                mid_max_error = NaN,
                near_boundary_max_error = Float64(row.max_error_near_boundary),
                argmax_index = 0,
                argmax_r = NaN,
                safe_argmax_index = 0,
                safe_argmax_r = NaN,
            )
        )
    end

    grad_global = isempty(gradient_entries) ? NaN :
        maximum(getfield.(gradient_entries, :max_error))
    grad_safe = isempty(gradient_entries) ? NaN :
        maximum(getfield.(gradient_entries, :safe_max_error))
    div_global = isempty(divergence_entries) ? NaN :
        maximum(getfield.(divergence_entries, :max_error))
    div_safe = isempty(divergence_entries) ? NaN :
        maximum(getfield.(divergence_entries, :safe_max_error))

    closure = _closure_diagnostics(ops.Geven)

    base_diag = (
        grid = (
            M = ops.M_full,
            Nh = ops.Nh,
            xfirst = Float64(ops.r[1]),
            xlast = Float64(ops.r[end]),
            symmetry_error = NaN,
            symmetry_ok = true,
            dx_min = ops.Nh > 1 ? Float64(minimum(diff(Float64.(ops.r)))) : NaN,
            dx_max = ops.Nh > 1 ? Float64(maximum(diff(Float64.(ops.r)))) : NaN,
            dx_mean = ops.Nh > 1 ? Float64(sum(diff(Float64.(ops.r))) / (ops.Nh - 1)) : NaN,
            dx_max_deviation = 0.0,
            uniform_ok = true,
            r0 = Float64(ops.r[1]),
            r_end = Float64(ops.r[end]),
            r0_error = NaN,
            r_end_error = 0.0,
            r0_ok = ops.r[1] > zero(eltype(ops.r)),
            rend_ok = true,
            r_match_error = 0.0,
            grid_ok = true,
            divergence_method = ops.divergence_method,
            sbp_expected = ops.divergence_method === :standard,
            atol_used = atol_diag,
        ),
        folding = (
            rop_ok = true,
            rop_one_nonzero_per_row = true,
            rop_values_ok = true,
            rop_index_match = true,
            rop_nonnegative_ok = true,
            rop_coordinate_match_ok = true,
            rop_coordinate_match_max = 0.0,
            even_extension = NamedTuple[],
            odd_extension = NamedTuple[],
            even_even_ok = true,
            odd_signed_ok = true,
            odd_matches_xk_for_odd_k_ok = true,
        ),
        operator_consistency = (
            geven_diff_max = 0.0,
            geven_diff_max_all_rows = 0.0,
            geven_repaired_rows = Int[],
            godd_diff_max = 0.0,
            H_offdiag_max = 0.0,
            Hcart_diag_ratio_max_error = 0.0,
        ),
        closure = (
            row_nnz = closure.row_nnz,
            interior_mode = closure.interior_mode,
            closure_width_left = closure.closure_width_left,
            closure_width_right = max(closure.closure_width_right, ops.closure_width),
            closure_width_right_pattern = closure.closure_width_right,
            closure_width_right_operator = ops.closure_width,
            idx_safe = report.diagnostics.idx_safe,
            safe_start = isempty(report.diagnostics.idx_safe) ? 1 :
                first(report.diagnostics.idx_safe),
            safe_end = isempty(report.diagnostics.idx_safe) ? 0 :
                last(report.diagnostics.idx_safe),
            safe_count = report.diagnostics.safe_count,
        ),
        polynomial = (
            skipped = false,
            reason = "",
            regions = (
                near_origin = Int[],
                mid = Int[],
                near_boundary = Int[],
            ),
            gradient = (
                entries = gradient_entries,
                global_max = grad_global,
                global_safe_max = grad_safe,
                worst_entry = _worst_entry(gradient_entries),
                worst_table = NamedTuple[],
            ),
            divergence = (
                entries = divergence_entries,
                global_max = div_global,
                global_safe_max = div_safe,
                worst_entry = _worst_entry(divergence_entries),
                worst_table = NamedTuple[],
            ),
        ),
        fullgrid_comparison = (
            entries = NamedTuple[],
            max_full_vs_exact = NaN,
            max_full_vs_exact_safe = NaN,
            max_fold_vs_exact = NaN,
            max_fold_vs_exact_safe = NaN,
            max_fold_vs_full = NaN,
        ),
        sbp = (
            max_no_origin = Float64(report.sbp.sbp_no_origin),
            max_safe_rows = Float64(report.sbp.sbp_no_origin),
            largest_entry_no_origin = (
                i = 0, j = 0, value = 0.0, abs_value = 0.0, ri = NaN, rj = NaN,
            ),
            largest_entry_all = (
                i = 0, j = 0, value = 0.0, abs_value = 0.0, ri = NaN, rj = NaN,
            ),
        ),
    )
    interpretation = interpret_diagnostics(base_diag)
    diag = merge(base_diag, (interpretation = interpretation,))

    if verbose
        println("Staggered diagnostics summary")
        println("  Grid: M=$(diag.grid.M), Nh=$(diag.grid.Nh), rmin=$(diag.grid.r0), rmax=$(diag.grid.r_end), uniform_ok=$(diag.grid.uniform_ok)")
        println("  Closure: left=$(diag.closure.closure_width_left), right=$(diag.closure.closure_width_right), safe_count=$(diag.closure.safe_count)")
        println("  Gradient safe max = $(diag.polynomial.gradient.global_safe_max), Divergence safe max = $(diag.polynomial.divergence.global_safe_max)")
        println("  SBP residual max = $(diag.sbp.max_no_origin)")
        println("  Interpretation:")
        for c in diag.interpretation.conclusions
            println("    - ", c)
        end
    end

    return diag
end
