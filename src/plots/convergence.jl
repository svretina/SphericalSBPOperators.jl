@inline function _extract_wave_solution(simulation)
    return hasproperty(simulation, :sol) ? getproperty(simulation, :sol) : simulation
end

function _load_glmakie()
    if !isdefined(@__MODULE__, :GLMakie)
        @eval import GLMakie
    end
    return Base.invokelatest(getfield, @__MODULE__, :GLMakie)
end

function _canonical_wave_field(field::Symbol)
    if field in (:Π, :pi, :Pi)
        return :Π
    elseif field in (:Ψ, :psi, :Psi)
        return :Ψ
    end
    throw(ArgumentError("`field` must be one of :Π/:pi/:Pi or :Ψ/:psi/:Psi."))
end

function _wave_field_history(simulation, field::Symbol)
    sol = _extract_wave_solution(simulation)
    field_sym = _canonical_wave_field(field)
    return field_sym === :Π ? sol.Π : sol.Ψ
end

function _nested_common_sampling(n_coarse::Int, n_medium::Int, n_fine::Int)
    n_coarse > 0 || throw(ArgumentError("Coarse data must be non-empty."))
    n_medium > 0 || throw(ArgumentError("Medium data must be non-empty."))
    n_fine > 0 || throw(ArgumentError("Fine data must be non-empty."))

    n_common = min(n_coarse, 1 + fld(n_medium - 1, 2), 1 + fld(n_fine - 1, 4))
    n_common > 0 || throw(ArgumentError("No common nested samples were found."))

    return (coarse = Base.collect(1:n_common),
            medium = Base.collect(1:2:(2 * n_common - 1)),
            fine = Base.collect(1:4:(4 * n_common - 3)))
end

function _time_match_tolerance(t::Real, atol::Real, rtol::Real)
    return max(Float64(atol), Float64(rtol) * max(abs(Float64(t)), 1.0))
end

function _advance_to_time_index(v::AbstractVector,
                                start::Int,
                                target::Float64;
                                atol::Real,
                                rtol::Real)
    i = start
    while i <= length(v) &&
          Float64(v[i]) < target - _time_match_tolerance(target, atol, rtol)
        i += 1
    end
    if i <= length(v) && isapprox(Float64(v[i]), target; atol = atol, rtol = rtol)
        return i
    end
    return nothing
end

function _common_time_sampling(t_coarse::AbstractVector,
                               t_medium::AbstractVector,
                               t_fine::AbstractVector;
                               atol::Real = 1.0e-10,
                               rtol::Real = 1.0e-8)
    isempty(t_coarse) && throw(ArgumentError("Coarse time data must be non-empty."))
    isempty(t_medium) && throw(ArgumentError("Medium time data must be non-empty."))
    isempty(t_fine) && throw(ArgumentError("Fine time data must be non-empty."))

    coarse = Int[]
    medium = Int[]
    fine = Int[]
    im = 1
    ifi = 1

    for ic in eachindex(t_coarse)
        tc = Float64(t_coarse[ic])
        imatch = _advance_to_time_index(t_medium, im, tc; atol = atol, rtol = rtol)
        fmatch = _advance_to_time_index(t_fine, ifi, tc; atol = atol, rtol = rtol)

        if imatch !== nothing && fmatch !== nothing
            push!(coarse, Int(ic))
            push!(medium, imatch)
            push!(fine, fmatch)
            im = imatch
            ifi = fmatch
        end
    end

    isempty(coarse) &&
        throw(ArgumentError("No common saved times were found between the three simulations."))

    return ((coarse = coarse, medium = medium, fine = fine), Float64.(t_coarse[coarse]))
end

function _validate_nested_vector_alignment(v_coarse::AbstractVector,
                                           v_medium::AbstractVector,
                                           v_fine::AbstractVector,
                                           sampling;
                                           name::AbstractString,
                                           atol::Real = 1.0e-10,
                                           rtol::Real = 1.0e-8)
    coarse = Base.collect(v_coarse[sampling.coarse])
    medium = Base.collect(v_medium[sampling.medium])
    fine = Base.collect(v_fine[sampling.fine])

    mismatch_medium = findfirst(eachindex(coarse, medium)) do idx
        coarse[idx] != medium[idx]
    end
    mismatch_fine = findfirst(eachindex(coarse, fine)) do idx
        coarse[idx] != fine[idx]
    end

    @assert isnothing(mismatch_medium) ("The provided simulations do not share identical nested `$name` samples between coarse and medium resolutions. "*
                                        "First mismatch at sampled index $(mismatch_medium): coarse=$(coarse[mismatch_medium]), medium=$(medium[mismatch_medium]).")
    @assert isnothing(mismatch_fine) ("The provided simulations do not share identical nested `$name` samples between coarse and fine resolutions. "*
                                      "First mismatch at sampled index $(mismatch_fine): coarse=$(coarse[mismatch_fine]), fine=$(fine[mismatch_fine]).")

    return Float64.(coarse)
end

function _error_norm(values::AbstractVector, norm_kind::Symbol)
    if norm_kind === :l2
        return norm(values) / sqrt(length(values))
    elseif norm_kind === :linf
        return maximum(abs.(values))
    end
    throw(ArgumentError("`norm_kind` must be :l2 or :linf."))
end

function _time_limits_with_padding(t::AbstractVector;
                                   left_pad_frac::Float64 = 0.0,
                                   right_pad_frac::Float64 = 0.10)
    tf = Float64.(t)
    isempty(tf) && throw(ArgumentError("Time vector must be non-empty."))

    tmin, tmax = extrema(tf)
    span = tmax - tmin
    ref_span = span > 0 ? span : max(abs(tmin), abs(tmax), 1.0)

    return (tmin - left_pad_frac * ref_span,
            tmax + right_pad_frac * ref_span)
end

function _convergence_order_curve(simulation_h,
                                  simulation_h2,
                                  simulation_h4;
                                  field::Symbol,
                                  norm_kind::Symbol = :l2)
    sol_h = _extract_wave_solution(simulation_h)
    sol_h2 = _extract_wave_solution(simulation_h2)
    sol_h4 = _extract_wave_solution(simulation_h4)

    time_sampling, common_t = _common_time_sampling(sol_h.t, sol_h2.t, sol_h4.t)
    spatial_sampling = _nested_common_sampling(length(sol_h.r), length(sol_h2.r),
                                               length(sol_h4.r))

    common_r = _validate_nested_vector_alignment(sol_h.r, sol_h2.r, sol_h4.r,
                                                 spatial_sampling;
                                                 name = "r")

    U_h = _wave_field_history(sol_h, field)
    U_h2 = _wave_field_history(sol_h2, field)
    U_h4 = _wave_field_history(sol_h4, field)

    orders = fill(NaN, length(common_t))
    err_h_h2 = zeros(Float64, length(common_t))
    err_h2_h4 = zeros(Float64, length(common_t))

    for k in eachindex(common_t)
        u_h = Float64.(U_h[spatial_sampling.coarse, time_sampling.coarse[k]])
        u_h2 = Float64.(U_h2[spatial_sampling.medium, time_sampling.medium[k]])
        u_h4 = Float64.(U_h4[spatial_sampling.fine, time_sampling.fine[k]])

        err12 = u_h .- u_h2
        err24 = u_h2 .- u_h4

        err_h_h2[k] = _error_norm(err12, norm_kind)
        err_h2_h4[k] = _error_norm(err24, norm_kind)

        if err_h_h2[k] > 0 && err_h2_h4[k] > 0
            orders[k] = log2(err_h_h2[k] / err_h2_h4[k])
        end
    end

    return (t = common_t,
            r = common_r,
            order = orders,
            err_h_h2 = err_h_h2,
            err_h2_h4 = err_h2_h4,
            field = _canonical_wave_field(field),
            norm_kind = norm_kind)
end

function _pointwise_convergence_order_history(simulation_h,
                                              simulation_h2,
                                              simulation_h4;
                                              field::Symbol)
    sol_h = _extract_wave_solution(simulation_h)
    sol_h2 = _extract_wave_solution(simulation_h2)
    sol_h4 = _extract_wave_solution(simulation_h4)

    time_sampling, common_t = _common_time_sampling(sol_h.t, sol_h2.t, sol_h4.t)
    spatial_sampling = _nested_common_sampling(length(sol_h.r), length(sol_h2.r),
                                               length(sol_h4.r))

    common_r = _validate_nested_vector_alignment(sol_h.r, sol_h2.r, sol_h4.r,
                                                 spatial_sampling;
                                                 name = "r")

    U_h = _wave_field_history(sol_h, field)
    U_h2 = _wave_field_history(sol_h2, field)
    U_h4 = _wave_field_history(sol_h4, field)

    nr = length(common_r)
    nt = length(common_t)
    order = fill(NaN, nr, nt)
    err_h_h2 = fill(NaN, nr, nt)
    err_h2_h4 = fill(NaN, nr, nt)

    for k in eachindex(common_t)
        u_h = Float64.(U_h[spatial_sampling.coarse, time_sampling.coarse[k]])
        u_h2 = Float64.(U_h2[spatial_sampling.medium, time_sampling.medium[k]])
        u_h4 = Float64.(U_h4[spatial_sampling.fine, time_sampling.fine[k]])

        diff12 = u_h .- u_h2
        diff24 = u_h2 .- u_h4
        err12 = abs.(diff12)
        err24 = abs.(diff24)

        err_h_h2[:, k] .= diff12
        err_h2_h4[:, k] .= diff24

        for i in eachindex(common_r)
            if err12[i] > 0 && err24[i] > 0
                order[i, k] = log2(err12[i] / err24[i])
            end
        end
    end

    return (t = common_t,
            r = common_r,
            order = order,
            err_h_h2 = err_h_h2,
            err_h2_h4 = err_h2_h4,
            field = _canonical_wave_field(field))
end

function _finite_limits_with_padding(values::AbstractArray;
                                     pad_frac::Float64 = 0.08,
                                     absolute_pad::Float64 = 1.0e-12)
    finite_vals = Float64[]
    for x in values
        xf = Float64(x)
        isfinite(xf) && push!(finite_vals, xf)
    end

    isempty(finite_vals) && return (-1.0, 1.0)

    vmin, vmax = extrema(finite_vals)
    span = vmax - vmin
    ref_scale = span > 0 ? span : max(abs(vmin), abs(vmax), 1.0)
    pad = max(Float64(absolute_pad), Float64(pad_frac) * ref_scale)
    return (vmin - pad, vmax + pad)
end

function _positive_limits_with_padding(values::AbstractArray;
                                       pad_frac::Float64 = 0.08,
                                       floor_value::Float64 = 1.0e-16)
    positive_vals = Float64[]
    for x in values
        xf = Float64(x)
        isfinite(xf) && xf > 0 && push!(positive_vals, xf)
    end

    isempty(positive_vals) && return (floor_value, 1.0)

    vmin, vmax = extrema(positive_vals)
    ylo = max(floor_value, (1 - pad_frac) * vmin)
    yhi = max(ylo * (1 + pad_frac), (1 + pad_frac) * vmax)
    return (ylo, yhi)
end

function _expected_order_from_simulation(simulation)
    if hasproperty(simulation, :ops)
        ops = getproperty(simulation, :ops)
        if hasproperty(ops, :accuracy_order)
            return Int(getproperty(ops, :accuracy_order))
        end
    end
    return nothing
end

function _resolve_expected_order(simulation_h,
                                 simulation_h2,
                                 simulation_h4,
                                 expected_order)
    if !isnothing(expected_order)
        q = Int(expected_order)
        q > 0 || throw(ArgumentError("`expected_order` must be positive."))
        return q
    end

    for sim in (simulation_h, simulation_h2, simulation_h4)
        q = _expected_order_from_simulation(sim)
        isnothing(q) || return q
    end

    throw(ArgumentError("Could not infer the convergence order from the supplied simulations. " *
                        "Pass `expected_order = q` explicitly."))
end

function _resolve_scale_order(expected_order::Real, scale_order)
    if isnothing(scale_order)
        return Float64(expected_order)
    end

    s = Float64(scale_order)
    s > 0 || throw(ArgumentError("`scale_order` must be positive."))
    return s
end

function _coerce_bundle_metadata(metadata_source)
    if isnothing(metadata_source)
        return nothing
    elseif metadata_source isa AbstractString
        return JLD2.load(metadata_source, "metadata")
    elseif metadata_source isa AbstractDict
        if haskey(metadata_source, "metadata")
            return metadata_source["metadata"]
        elseif haskey(metadata_source, :metadata)
            return metadata_source[:metadata]
        end
        return metadata_source
    end
    return metadata_source
end

function _simulation_boundary_condition(simulation)
    if hasproperty(simulation, :wave_config) &&
       hasproperty(getproperty(simulation, :wave_config), :boundary_condition)
        return getproperty(getproperty(simulation, :wave_config), :boundary_condition)
    elseif hasproperty(simulation, :diagnostics) &&
           hasproperty(getproperty(simulation, :diagnostics), :boundary_condition)
        return getproperty(getproperty(simulation, :diagnostics), :boundary_condition)
    end

    sol = _extract_wave_solution(simulation)
    if hasproperty(sol, :boundary_condition)
        return getproperty(sol, :boundary_condition)
    end

    return nothing
end

function _simulation_initial_data_info(simulation)
    if hasproperty(simulation, :initial_data)
        init = getproperty(simulation, :initial_data)
        if hasproperty(init, :kind) && hasproperty(init, :summary)
            return (kind = getproperty(init, :kind), summary = getproperty(init, :summary))
        end
    end
    return nothing
end

function _analytic_reference_from_kind_summary(kind::Symbol,
                                               summary,
                                               boundary_condition)
    hasproperty(summary, :amplitude) ||
        throw(ArgumentError("Initial-data summary is missing `amplitude`."))
    hasproperty(summary, :d) ||
        throw(ArgumentError("Initial-data summary is missing `d`."))

    if kind === :even_gaussian_pi_zero_psi
        hasproperty(summary, :r0) ||
            throw(ArgumentError("Initial-data summary is missing `r0`."))
        return (kind = kind,
                amplitude = Float64(getproperty(summary, :amplitude)),
                r0 = Float64(getproperty(summary, :r0)),
                d = Float64(getproperty(summary, :d)),
                boundary_condition = boundary_condition)
    elseif kind === :regular_left_moving_spherical_gaussian
        hasproperty(summary, :R) ||
            throw(ArgumentError("Initial-data summary is missing `R`."))
        rc = hasproperty(summary, :rc) ? Float64(getproperty(summary, :rc)) :
             0.5 * Float64(getproperty(summary, :R))
        return (kind = kind,
                amplitude = Float64(getproperty(summary, :amplitude)),
                R = Float64(getproperty(summary, :R)),
                d = Float64(getproperty(summary, :d)),
                rc = rc,
                boundary_condition = boundary_condition)
    end

    throw(ArgumentError("No analytic reference is defined for initial-data kind `$kind`."))
end

function _analytic_reference_from_metadata(metadata)
    md = _coerce_bundle_metadata(metadata)
    isnothing(md) && return nothing
    hasproperty(md, :shared) ||
        throw(ArgumentError("Bundle metadata does not contain a `shared` section."))
    shared = getproperty(md, :shared)
    hasproperty(shared, :initial_data) ||
        throw(ArgumentError("Bundle metadata shared section does not contain `initial_data`."))
    hasproperty(shared, :wave_config) ||
        throw(ArgumentError("Bundle metadata shared section does not contain `wave_config`."))

    init = getproperty(shared, :initial_data)
    wave = getproperty(shared, :wave_config)
    hasproperty(init, :kind) || throw(ArgumentError("Metadata initial_data is missing `kind`."))
    hasproperty(init, :summary) || throw(ArgumentError("Metadata initial_data is missing `summary`."))
    hasproperty(wave, :boundary_condition) ||
        throw(ArgumentError("Metadata wave_config is missing `boundary_condition`."))

    return _analytic_reference_from_kind_summary(getproperty(init, :kind),
                                                 getproperty(init, :summary),
                                                 getproperty(wave, :boundary_condition))
end

function _resolve_analytic_reference(simulation; analytic_reference = nothing, metadata = nothing)
    if !isnothing(analytic_reference)
        return analytic_reference
    end

    sim_info = _simulation_initial_data_info(simulation)
    bc = _simulation_boundary_condition(simulation)
    if !isnothing(sim_info) && !isnothing(bc)
        return _analytic_reference_from_kind_summary(sim_info.kind, sim_info.summary, bc)
    end

    md_ref = _analytic_reference_from_metadata(metadata)
    isnothing(md_ref) ||
        return md_ref

    throw(ArgumentError("Could not infer an analytic reference from the supplied simulation. " *
                        "Pass `analytic_reference=...` explicitly or provide bundle `metadata`."))
end

function _pointwise_analytic_error_history(simulation_h,
                                           simulation_h2,
                                           simulation_h4;
                                           field::Symbol,
                                           expected_order = nothing,
                                           analytic_reference = nothing,
                                           metadata = nothing)
    sol_h = _extract_wave_solution(simulation_h)
    sol_h2 = _extract_wave_solution(simulation_h2)
    sol_h4 = _extract_wave_solution(simulation_h4)

    time_sampling, common_t = _common_time_sampling(sol_h.t, sol_h2.t, sol_h4.t)
    spatial_sampling = _nested_common_sampling(length(sol_h.r), length(sol_h2.r),
                                               length(sol_h4.r))

    common_r = _validate_nested_vector_alignment(sol_h.r, sol_h2.r, sol_h4.r,
                                                 spatial_sampling;
                                                 name = "r")

    U_h = _wave_field_history(sol_h, field)
    U_h2 = _wave_field_history(sol_h2, field)
    U_h4 = _wave_field_history(sol_h4, field)

    q = _resolve_expected_order(simulation_h, simulation_h2, simulation_h4, expected_order)
    reference = _resolve_analytic_reference(simulation_h;
                                            analytic_reference = analytic_reference,
                                            metadata = metadata)

    h = _gundlach_grid_spacing(sol_h.r)
    h2 = _gundlach_grid_spacing(sol_h2.r)
    h4 = _gundlach_grid_spacing(sol_h4.r)
    h0 = Float64(h)
    scale_h = 1.0
    scale_h2 = (Float64(h2) / h0)^(-Float64(q))
    scale_h4 = (Float64(h4) / h0)^(-Float64(q))

    nr = length(common_r)
    nt = length(common_t)
    err_h = fill(NaN, nr, nt)
    err_h2 = fill(NaN, nr, nt)
    err_h4 = fill(NaN, nr, nt)
    scaled_err_h = fill(NaN, nr, nt)
    scaled_err_h2 = fill(NaN, nr, nt)
    scaled_err_h4 = fill(NaN, nr, nt)

    field_sym = _canonical_wave_field(field)
    for k in eachindex(common_t)
        t = common_t[k]
        exact = analytic_wave_solution(t, common_r, reference)
        u_exact = field_sym === :Π ? Float64.(exact.Π) : Float64.(exact.Ψ)
        u_h = Float64.(U_h[spatial_sampling.coarse, time_sampling.coarse[k]])
        u_h2 = Float64.(U_h2[spatial_sampling.medium, time_sampling.medium[k]])
        u_h4 = Float64.(U_h4[spatial_sampling.fine, time_sampling.fine[k]])

        diff_h = u_h .- u_exact
        diff_h2 = u_h2 .- u_exact
        diff_h4 = u_h4 .- u_exact

        err_h[:, k] .= diff_h
        err_h2[:, k] .= diff_h2
        err_h4[:, k] .= diff_h4
        scaled_err_h[:, k] .= scale_h .* diff_h
        scaled_err_h2[:, k] .= scale_h2 .* diff_h2
        scaled_err_h4[:, k] .= scale_h4 .* diff_h4
    end

    return (t = common_t,
            r = common_r,
            err_h = err_h,
            err_h2 = err_h2,
            err_h4 = err_h4,
            scaled_err_h = scaled_err_h,
            scaled_err_h2 = scaled_err_h2,
            scaled_err_h4 = scaled_err_h4,
            h = Float64(h),
            h2 = Float64(h2),
            h4 = Float64(h4),
            h0 = h0,
            expected_order = q,
            scale_h = scale_h,
            scale_h2 = scale_h2,
            scale_h4 = scale_h4,
            analytic_reference = reference,
            field = field_sym)
end

function _ops_from_simulation(simulation)
    if hasproperty(simulation, :ops)
        return getproperty(simulation, :ops)
    end
    throw(ArgumentError("The supplied simulation object does not provide an `ops` field needed for the Gundlach SBP norm diagnostic."))
end

function _gundlach_pointwise_error_history(simulation_h,
                                           simulation_h2,
                                           simulation_h4;
                                           field::Symbol,
                                           expected_order = nothing,
                                           scale_order = nothing,
                                           h0::Real = 0.1)
    sol_h = _extract_wave_solution(simulation_h)
    sol_h2 = _extract_wave_solution(simulation_h2)
    sol_h4 = _extract_wave_solution(simulation_h4)
    ops_h = _ops_from_simulation(simulation_h)
    ops_h2 = _ops_from_simulation(simulation_h2)

    time_sampling, common_t = _common_time_sampling(sol_h.t, sol_h2.t, sol_h4.t)
    common_r = Float64.(sol_h.r)

    U_h = _wave_field_history(sol_h, field)
    U_h2 = _wave_field_history(sol_h2, field)
    U_h4 = _wave_field_history(sol_h4, field)

    s_expected = _resolve_expected_order(simulation_h,
                                         simulation_h2,
                                         simulation_h4,
                                         expected_order)
    s_scale = _resolve_scale_order(s_expected, scale_order)
    h = _gundlach_grid_spacing(sol_h.r)
    h2 = _gundlach_grid_spacing(sol_h2.r)
    variable = _canonical_wave_field(field) === :Π ? Val(:Π) : Val(:Ψ)
    H_h = _gundlach_norm_matrix(ops_h, variable)
    H_h2 = _gundlach_norm_matrix(ops_h2, variable)
    R_h = _gundlach_radius(ops_h, common_r)
    R_h2 = _gundlach_radius(ops_h2, sol_h2.r)

    nr = length(common_r)
    nt = length(common_t)
    err_h_h2 = fill(NaN, nr, nt)
    err_h2_h4 = fill(NaN, nr, nt)
    norm_h_h2 = fill(NaN, nt)
    norm_h2_h4 = fill(NaN, nt)

    for k in eachindex(common_t)
        u_h = Float64.(U_h[:, time_sampling.coarse[k]])
        u_h2 = _align_reference_to_grid(common_r,
                                        sol_h2.r,
                                        U_h2[:, time_sampling.medium[k]])
        u_h4 = _align_reference_to_grid(common_r,
                                        sol_h4.r,
                                        U_h4[:, time_sampling.fine[k]])
        u_h2_native = Float64.(U_h2[:, time_sampling.medium[k]])
        u_h4_on_h2 = _align_reference_to_grid(sol_h2.r,
                                              sol_h4.r,
                                              U_h4[:, time_sampling.fine[k]])

        err12 = gundlach_error(u_h, u_h2; h = h, s = s_scale, h0 = h0)
        err24 = gundlach_error(u_h2, u_h4; h = h2, s = s_scale, h0 = h0)

        err_h_h2[:, k] .= err12
        err_h2_h4[:, k] .= err24
        norm_h_h2[k] = gundlach_error_norm(u_h, u_h2, H_h;
                                           h = h,
                                           s = s_scale,
                                           R = R_h,
                                           h0 = h0)
        norm_h2_h4[k] = gundlach_error_norm(u_h2_native, u_h4_on_h2, H_h2;
                                            h = h2,
                                            s = s_scale,
                                            R = R_h2,
                                            h0 = h0)
    end

    return (t = common_t,
            r = common_r,
            err_h_h2 = err_h_h2,
            err_h2_h4 = err_h2_h4,
            norm_h_h2 = norm_h_h2,
            norm_h2_h4 = norm_h2_h4,
            expected_order = s_expected,
            scale_order = s_scale,
            h0 = Float64(h0),
            field = _canonical_wave_field(field))
end

function _gundlach_pointwise_error_history_with_reference(simulation_h,
                                                          simulation_h2,
                                                          simulation_ref;
                                                          field::Symbol,
                                                          expected_order = nothing,
                                                          scale_order = nothing,
                                                          h0::Real = 0.1)
    sol_h = _extract_wave_solution(simulation_h)
    sol_h2 = _extract_wave_solution(simulation_h2)
    sol_ref = _extract_wave_solution(simulation_ref)
    ops_h = _ops_from_simulation(simulation_h)

    time_sampling, common_t = _common_time_sampling(sol_h.t, sol_h2.t, sol_ref.t)
    common_r = Float64.(sol_h.r)

    U_h = _wave_field_history(sol_h, field)
    U_h2 = _wave_field_history(sol_h2, field)
    U_ref = _wave_field_history(sol_ref, field)

    s_expected = _resolve_expected_order(simulation_h,
                                         simulation_h2,
                                         simulation_ref,
                                         expected_order)
    s_scale = _resolve_scale_order(s_expected, scale_order)
    h = _gundlach_grid_spacing(sol_h.r)
    h2 = _gundlach_grid_spacing(sol_h2.r)
    variable = _canonical_wave_field(field) === :Π ? Val(:Π) : Val(:Ψ)
    H_h = _gundlach_norm_matrix(ops_h, variable)
    R_h = _gundlach_radius(ops_h, common_r)

    nr = length(common_r)
    nt = length(common_t)
    err_h_ref = fill(NaN, nr, nt)
    err_h2_ref = fill(NaN, nr, nt)
    norm_h_ref = fill(NaN, nt)
    norm_h2_ref = fill(NaN, nt)

    for k in eachindex(common_t)
        u_h = Float64.(U_h[:, time_sampling.coarse[k]])
        u_h2_on_h = _align_reference_to_grid(common_r,
                                             sol_h2.r,
                                             U_h2[:, time_sampling.medium[k]])
        u_ref_on_h = _align_reference_to_grid(common_r,
                                              sol_ref.r,
                                              U_ref[:, time_sampling.fine[k]])

        err_h = gundlach_error(u_h, u_ref_on_h; h = h, s = s_scale, h0 = h0)
        err_h2 = gundlach_error(u_h2_on_h, u_ref_on_h; h = h2, s = s_scale, h0 = h0)

        err_h_ref[:, k] .= err_h
        err_h2_ref[:, k] .= err_h2
        norm_h_ref[k] = gundlach_error_norm(u_h, u_ref_on_h, H_h;
                                            h = h,
                                            s = s_scale,
                                            R = R_h,
                                            h0 = h0)
        norm_h2_ref[k] = gundlach_error_norm(u_h2_on_h, u_ref_on_h, H_h;
                                             h = h2,
                                             s = s_scale,
                                             R = R_h,
                                             h0 = h0)
    end

    return (t = common_t,
            r = common_r,
            err_h_ref = err_h_ref,
            err_h2_ref = err_h2_ref,
            norm_h_ref = norm_h_ref,
            norm_h2_ref = norm_h2_ref,
            expected_order = s_expected,
            scale_order = s_scale,
            h0 = Float64(h0),
            field = _canonical_wave_field(field))
end

function _nearest_common_time_index(t::AbstractVector, time::Real)
    isempty(t) && throw(ArgumentError("Time vector must be non-empty."))
    target = Float64(time)
    return argmin(abs.(Float64.(t) .- target))
end

function _gundlach_pointwise_error_snapshot(simulation_h,
                                            simulation_h2,
                                            simulation_h4;
                                            field::Symbol,
                                            time::Real,
                                            expected_order = nothing,
                                            h0::Real = 0.1)
    sol_h = _extract_wave_solution(simulation_h)
    sol_h2 = _extract_wave_solution(simulation_h2)
    sol_h4 = _extract_wave_solution(simulation_h4)
    ops_h = _ops_from_simulation(simulation_h)
    ops_h2 = _ops_from_simulation(simulation_h2)

    time_sampling, common_t = _common_time_sampling(sol_h.t, sol_h2.t, sol_h4.t)
    time_index = _nearest_common_time_index(common_t, time)
    common_r = Float64.(sol_h.r)

    U_h = _wave_field_history(sol_h, field)
    U_h2 = _wave_field_history(sol_h2, field)
    U_h4 = _wave_field_history(sol_h4, field)

    s = _resolve_expected_order(simulation_h, simulation_h2, simulation_h4, expected_order)
    h = _gundlach_grid_spacing(sol_h.r)
    h2 = _gundlach_grid_spacing(sol_h2.r)
    variable = _canonical_wave_field(field) === :Π ? Val(:Π) : Val(:Ψ)
    H_h = _gundlach_norm_matrix(ops_h, variable)
    H_h2 = _gundlach_norm_matrix(ops_h2, variable)
    R_h = _gundlach_radius(ops_h, common_r)
    R_h2 = _gundlach_radius(ops_h2, sol_h2.r)

    u_h = Float64.(U_h[:, time_sampling.coarse[time_index]])
    u_h2 = _align_reference_to_grid(common_r,
                                    sol_h2.r,
                                    U_h2[:, time_sampling.medium[time_index]])
    u_h4 = _align_reference_to_grid(common_r,
                                    sol_h4.r,
                                    U_h4[:, time_sampling.fine[time_index]])
    u_h2_native = Float64.(U_h2[:, time_sampling.medium[time_index]])
    u_h4_on_h2 = _align_reference_to_grid(sol_h2.r,
                                          sol_h4.r,
                                          U_h4[:, time_sampling.fine[time_index]])

    err_h_h2 = gundlach_error(u_h, u_h2; h = h, s = s, h0 = h0)
    err_h2_h4 = gundlach_error(u_h2, u_h4; h = h2, s = s, h0 = h0)
    norm_h_h2 = gundlach_error_norm(u_h, u_h2, H_h;
                                    h = h,
                                    s = s,
                                    R = R_h,
                                    h0 = h0)
    norm_h2_h4 = gundlach_error_norm(u_h2_native, u_h4_on_h2, H_h2;
                                     h = h2,
                                     s = s,
                                     R = R_h2,
                                     h0 = h0)

    return (t = common_t[time_index],
            requested_time = Float64(time),
            time_index = time_index,
            r = common_r,
            err_h_h2 = err_h_h2,
            err_h2_h4 = err_h2_h4,
            norm_h_h2 = norm_h_h2,
            norm_h2_h4 = norm_h2_h4,
            expected_order = q,
            h0 = Float64(h0),
            field = _canonical_wave_field(field))
end

function _gundlach_pointwise_convergence_order_history(simulation_h,
                                                       simulation_h2,
                                                       simulation_h4;
                                                       field::Symbol,
                                                       expected_order = nothing,
                                                       h0::Real = 0.1)
    data = _gundlach_pointwise_error_history(simulation_h, simulation_h2, simulation_h4;
                                             field = field,
                                             expected_order = expected_order,
                                             h0 = h0)

    nr, nt = size(data.err_h_h2)
    order = fill(NaN, nr, nt)
    norm_order = fill(NaN, nt)

    for k in 1:nt
        for i in 1:nr
            err12 = abs(data.err_h_h2[i, k])
            err24 = abs(data.err_h2_h4[i, k])
            if err12 > 0 && err24 > 0
                order[i, k] = data.expected_order + log2(err12 / err24)
            end
        end

        if data.norm_h_h2[k] > 0 && data.norm_h2_h4[k] > 0
            norm_order[k] = data.expected_order +
                            log2(data.norm_h_h2[k] / data.norm_h2_h4[k])
        end
    end

    return merge(data, (order = order, norm_order = norm_order))
end

"""
    plot_convergence_order(simulation_h, simulation_h2, simulation_h4; field=:Π, norm_kind=:l2)

Plot the observed convergence order versus time using three nested simulations with
resolutions `dr`, `dr/2`, and `dr/4`.

The routine samples only the common nested grid/time locations:
- coarse: `i`
- medium: `2i - 1`
- fine: `4i - 3`

and computes the observed order
`log2(||u_h - u_h/2|| / ||u_h/2 - u_h/4||)`.
"""
function plot_convergence_order(simulation_h,
                                simulation_h2,
                                simulation_h4;
                                field::Symbol = :Π,
                                norm_kind::Symbol = :l2,
                                title::Union{Nothing, AbstractString} = nothing)
    data = _convergence_order_curve(simulation_h, simulation_h2, simulation_h4;
                                    field = field,
                                    norm_kind = norm_kind)

    field_tex = data.field === :Π ? "\\Pi" : "\\Psi"
    plot_title = isnothing(title) ?
                 latexstring("\\mathrm{Observed\\ Convergence\\ Order}:\\ ", field_tex) :
                 title
    ylabel = norm_kind === :l2 ?
             L"\mathrm{observed\ order}_{L_2}" :
             L"\mathrm{observed\ order}_{L_\infty}"

    fig = _with_spectrum_theme() do
        fig = Figure(size = (880, 480))
        ax = Axis(fig[1, 1],
                  xlabel = L"t",
                  ylabel = ylabel,
                  title = plot_title)
        lines!(ax, data.t, data.order; linewidth = 2.5, color = :royalblue4)
        # scatter!(ax, data.t, data.order; markersize = 7)
        xlims!(ax, _time_limits_with_padding(data.t)...)
        fig
    end

    return (fig = fig, data = data)
end

"""
    plot_wave_convergence_orders(simulation_h, simulation_h2, simulation_h4; norm_kind=:l2)

Convenience wrapper that plots the time-dependent observed convergence order for both
`Π` and `Ψ` in a two-panel figure.
"""
function plot_wave_convergence_orders(simulation_h,
                                      simulation_h2,
                                      simulation_h4;
                                      norm_kind::Symbol = :l2)
    pi_data = _convergence_order_curve(simulation_h, simulation_h2, simulation_h4;
                                       field = :Π,
                                       norm_kind = norm_kind)
    psi_data = _convergence_order_curve(simulation_h, simulation_h2, simulation_h4;
                                        field = :Ψ,
                                        norm_kind = norm_kind)

    ylabel = norm_kind === :l2 ?
             L"\mathrm{observed\ order\ (RMS)}" :
             L"\mathrm{observed\ order\ (Linf)}"

    fig = _with_spectrum_theme() do
        fig = Figure(size = (980, 780))

        ax_pi = Axis(fig[1, 1],
                     xlabel = L"t",
                     ylabel = ylabel,
                     title = L"\mathrm{Observed\ Convergence\ Order}: \Pi")
        lines!(ax_pi, pi_data.t, pi_data.order; linewidth = 2.5, color = :royalblue4)
        scatter!(ax_pi, pi_data.t, pi_data.order; markersize = 7)
        xlims!(ax_pi, _time_limits_with_padding(pi_data.t)...)

        ax_psi = Axis(fig[2, 1],
                      xlabel = L"t",
                      ylabel = ylabel,
                      title = L"\mathrm{Observed\ Convergence\ Order}: \Psi")
        lines!(ax_psi, psi_data.t, psi_data.order; linewidth = 2.5, color = :seagreen4)
        scatter!(ax_psi, psi_data.t, psi_data.order; markersize = 7)
        xlims!(ax_psi, _time_limits_with_padding(psi_data.t)...)

        fig
    end

    return (fig = fig, pi = pi_data, psi = psi_data)
end

"""
    interactive_pointwise_wave_convergence_orders(simulation_h, simulation_h2, simulation_h4;
                                                  start_index=1, display_figure=true)

Open an interactive GLMakie viewer for the pointwise observed convergence order on the
common nested coarse grid.

For each common time snapshot and each common grid point `r_i`, the viewer displays

`log2(abs(u_h - u_h/2) / abs(u_h/2 - u_h/4))`

for both `Π` and `Ψ` in synchronized subplots. A shared time slider selects the
snapshot shown in both panels.
"""
function interactive_pointwise_wave_convergence_orders(simulation_h,
                                                       simulation_h2,
                                                       simulation_h4;
                                                       start_index::Int = 1,
                                                       display_figure::Bool = true)
    GLM = Base.invokelatest(_load_glmakie)
    return Base.invokelatest(_interactive_pointwise_wave_convergence_orders_impl,
                             GLM,
                             simulation_h,
                             simulation_h2,
                             simulation_h4,
                             start_index,
                             display_figure)
end

function _interactive_pointwise_wave_convergence_orders_impl(GLM,
                                                             simulation_h,
                                                             simulation_h2,
                                                             simulation_h4,
                                                             start_index::Int,
                                                             display_figure::Bool)
    pi_data = _pointwise_convergence_order_history(simulation_h, simulation_h2,
                                                   simulation_h4;
                                                   field = :Π)
    psi_data = _pointwise_convergence_order_history(simulation_h, simulation_h2,
                                                    simulation_h4;
                                                    field = :Ψ)

    nt = length(pi_data.t)
    1 <= start_index <= nt ||
        throw(ArgumentError("`start_index` must satisfy 1 <= start_index <= $(nt)."))

    GLM.activate!()

    r = Float64.(pi_data.r)
    fig = GLM.with_theme(mytheme_aps()) do
        fig = GLM.Figure(size = (1500, 840), figure_padding = (28, 24, 22, 18))

        GLM.Label(fig[0, 1:2],
                  L"\mathrm{Pointwise\ Observed\ Convergence\ Order}";
                  fontsize = 20,
                  font = :bold)

        ax_pi = GLM.Axis(fig[1, 1];
                         xlabel = L"r",
                         ylabel = L"\mathrm{pointwise\ order}",
                         title = L"\Pi")
        ax_psi = GLM.Axis(fig[1, 2];
                          xlabel = L"r",
                          ylabel = L"\mathrm{pointwise\ order}",
                          title = L"\Psi")

        for ax in (ax_pi, ax_psi)
            GLM.hlines!(ax, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
            GLM.xlims!(ax, minimum(r), maximum(r))
            GLM.ylims!(ax, 0.0, 10.0)
        end

        slider = GLM.Slider(fig[3, 1:2];
                            range = 1:nt,
                            startvalue = start_index,
                            snap = true)
        snapshot_index = slider.value

        time_label = GLM.lift(snapshot_index) do idx
            return latexstring("\\mathrm{t = ", @sprintf("%.6g", Float64(pi_data.t[idx])),
                               "\\quad snapshot = ", string(idx), "/", string(nt), "}")
        end
        GLM.Label(fig[2, 1:2], time_label; fontsize = 15)

        pi_obs = GLM.lift(idx -> pi_data.order[:, idx], snapshot_index)
        psi_obs = GLM.lift(idx -> psi_data.order[:, idx], snapshot_index)

        GLM.lines!(ax_pi, r, pi_obs; color = :royalblue4, linewidth = 2.5)
        GLM.scatter!(ax_pi, r, pi_obs; color = :royalblue4, markersize = 8)

        GLM.lines!(ax_psi, r, psi_obs; color = :seagreen4, linewidth = 2.5)
        GLM.scatter!(ax_psi, r, psi_obs; color = :seagreen4, markersize = 8)

        fig
    end

    screen = display_figure ? GLM.display(fig) : nothing
    return (fig = fig,
            screen = screen,
            pi = pi_data,
            psi = psi_data)
end

"""
    interactive_pointwise_wave_scaled_errors(simulation_h, simulation_h2, simulation_h4;
                                             start_index=1, expected_order=nothing,
                                             display_figure=true)

Open an interactive GLMakie viewer for the pointwise nested scaled-difference curves
on the common coarse grid, before the pointwise convergence-order ratio is formed.

For each common time snapshot and each common grid point `r_i`, the viewer displays

- `u_h - u_{h/2}`
- `2^q (u_{h/2} - u_{h/4})`

for both `Π` and `Ψ` in synchronized subplots. A shared time slider selects the
snapshot shown in both panels, where `q` is the expected convergence order.

If the leading error behaves like `C h^q`, then

- `u_h - u_{h/2} ~ C (1 - 2^{-q}) h^q`
- `u_{h/2} - u_{h/4} ~ C (1 - 2^{-q}) (h/2)^q`

so multiplying the finer-resolution difference by `2^q` should align the two curves.
"""
function interactive_pointwise_wave_scaled_errors(simulation_h,
                                                  simulation_h2,
                                                  simulation_h4;
                                                  start_index::Int = 1,
                                                  expected_order = nothing,
                                                  display_figure::Bool = true)
    GLM = Base.invokelatest(_load_glmakie)
    return Base.invokelatest(_interactive_pointwise_wave_scaled_errors_impl,
                             GLM,
                             simulation_h,
                             simulation_h2,
                             simulation_h4,
                             start_index,
                             expected_order,
                             display_figure)
end

function _interactive_pointwise_wave_scaled_errors_impl(GLM,
                                                        simulation_h,
                                                        simulation_h2,
                                                        simulation_h4,
                                                        start_index::Int,
                                                        expected_order,
                                                        display_figure::Bool)
    pi_data = _pointwise_convergence_order_history(simulation_h, simulation_h2,
                                                   simulation_h4;
                                                   field = :Π)
    psi_data = _pointwise_convergence_order_history(simulation_h, simulation_h2,
                                                    simulation_h4;
                                                    field = :Ψ)
    q = _resolve_expected_order(simulation_h, simulation_h2, simulation_h4, expected_order)
    scale_factor = 2.0^q

    nt = length(pi_data.t)
    1 <= start_index <= nt ||
        throw(ArgumentError("`start_index` must satisfy 1 <= start_index <= $(nt)."))

    GLM.activate!()
    r = Float64.(pi_data.r)
    pi_diff12 = pi_data.err_h_h2
    pi_diff24 = scale_factor .* pi_data.err_h2_h4
    psi_diff12 = psi_data.err_h_h2
    psi_diff24 = scale_factor .* psi_data.err_h2_h4

    pi_limits = _finite_limits_with_padding(vcat(vec(pi_diff12),
                                                 vec(pi_diff24));
                                            pad_frac = 0.1)
    psi_limits = _finite_limits_with_padding(vcat(vec(psi_diff12),
                                                  vec(psi_diff24));
                                             pad_frac = 0.1)

    fig = GLM.with_theme(mytheme_aps()) do
        fig = GLM.Figure(size = (1500, 920), figure_padding = (28, 24, 22, 18))

        GLM.Label(fig[0, 1:2],
                  latexstring("\\mathrm{Pointwise\\ Convergence\\text{-}Scaled\\ Difference\\ Curves}\\quad(q=",
                              string(q), ")");
                  fontsize = 20,
                  font = :bold)

        ax_pi = GLM.Axis(fig[1, 1];
                         xlabel = L"r",
                         ylabel = latexstring("\\Delta u\\ \\mathrm{and}\\ 2^{", string(q),
                                              "}\\Delta u_{1/2}"),
                         title = L"\Pi")
        ax_psi = GLM.Axis(fig[1, 2];
                          xlabel = L"r",
                          ylabel = latexstring("\\Delta u\\ \\mathrm{and}\\ 2^{", string(q),
                                               "}\\Delta u_{1/2}"),
                          title = L"\Psi")

        for (ax, limits) in ((ax_pi, pi_limits), (ax_psi, psi_limits))
            GLM.hlines!(ax, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
            GLM.xlims!(ax, minimum(r), maximum(r))
            GLM.ylims!(ax, limits[1], limits[2])
        end

        scaling_label = latexstring("\\mathrm{Scaling:}\\quad",
                                    "\\Delta_h = u_h - u_{h/2},\\quad",
                                    "\\Delta_{h/2} = u_{h/2} - u_{h/4},\\quad",
                                    "2^{", string(q), "}\\Delta_{h/2}\\ \\mathrm{should\\ align\\ with}\\ \\Delta_h")
        GLM.Label(fig[2, 1:2], scaling_label; fontsize = 15)

        slider = GLM.Slider(fig[4, 1:2];
                            range = 1:nt,
                            startvalue = start_index,
                            snap = true)
        snapshot_index = slider.value

        time_label = GLM.lift(snapshot_index) do idx
            return latexstring("\\mathrm{t = ", @sprintf("%.6g", Float64(pi_data.t[idx])),
                               "\\quad snapshot = ", string(idx), "/", string(nt), "}")
        end
        GLM.Label(fig[3, 1:2], time_label; fontsize = 15)

        pi_err12_obs = GLM.lift(idx -> pi_diff12[:, idx], snapshot_index)
        pi_err24_obs = GLM.lift(idx -> pi_diff24[:, idx], snapshot_index)
        psi_err12_obs = GLM.lift(idx -> psi_diff12[:, idx], snapshot_index)
        psi_err24_obs = GLM.lift(idx -> psi_diff24[:, idx], snapshot_index)

        GLM.lines!(ax_pi, r, pi_err12_obs; color = :royalblue4, linewidth = 2.5,
                   label = latexstring("u_h - u_{h/2}"))
        GLM.scatter!(ax_pi, r, pi_err12_obs; color = :royalblue4, markersize = 8)
        GLM.lines!(ax_pi, r, pi_err24_obs; color = :darkorange3, linewidth = 2.5,
                   linestyle = :dash,
                   label = latexstring("2^{", string(q), "} (u_{h/2} - u_{h/4})"))
        GLM.scatter!(ax_pi, r, pi_err24_obs; color = :darkorange3, markersize = 8,
                     marker = :rect)
        GLM.axislegend(ax_pi; position = :rb)

        GLM.lines!(ax_psi, r, psi_err12_obs; color = :seagreen4, linewidth = 2.5,
                   label = latexstring("u_h - u_{h/2}"))
        GLM.scatter!(ax_psi, r, psi_err12_obs; color = :seagreen4, markersize = 8)
        GLM.lines!(ax_psi, r, psi_err24_obs; color = :firebrick, linewidth = 2.5,
                   linestyle = :dash,
                   label = latexstring("2^{", string(q), "} (u_{h/2} - u_{h/4})"))
        GLM.scatter!(ax_psi, r, psi_err24_obs; color = :firebrick, markersize = 8,
                     marker = :rect)
        GLM.axislegend(ax_psi; position = :rb)

        fig
    end

    screen = display_figure ? GLM.display(fig) : nothing
    return (fig = fig,
            screen = screen,
            pi = pi_data,
            psi = psi_data,
            scaling = (expected_order = q,
                       coarse_factor = 1.0,
                       fine_factor = scale_factor,
                       formula = "Delta_h = u_h - u_{h/2}, 2^q * Delta_h/2 = 2^q * (u_{h/2} - u_{h/4})"))
end

"""
    interactive_pointwise_wave_scaled_errors_analytic(simulation_h, simulation_h2, simulation_h4;
                                                      start_index=1,
                                                      expected_order=nothing,
                                                      analytic_reference=nothing,
                                                      metadata=nothing,
                                                      display_figure=true)

Open an interactive GLMakie viewer for pointwise errors against an analytic solution.

The analytic reference is inferred from the saved simulation metadata when possible:

- `simulation.wave_config.boundary_condition`
- `simulation.initial_data.kind`
- `simulation.initial_data.summary`

If that information is unavailable, pass `metadata` as either the bundle metadata
NamedTuple/dictionary or the JLD2 bundle path containing a top-level `metadata`
entry. The viewer plots

- `(h/h0)^(-q) (u_h - u_exact)`
- `(h/2h0)^(-q) (u_{h/2} - u_exact)`
- `(h/4h0)^(-q) (u_{h/4} - u_exact)`

on the common coarse grid, with `h0 = h` by default, so the displayed scale factors
become `1`, `2^q`, and `4^q` for standard nested refinements.
"""
function interactive_pointwise_wave_scaled_errors_analytic(simulation_h,
                                                           simulation_h2,
                                                           simulation_h4;
                                                           start_index::Int = 1,
                                                           expected_order = nothing,
                                                           analytic_reference = nothing,
                                                           metadata = nothing,
                                                           display_figure::Bool = true)
    GLM = Base.invokelatest(_load_glmakie)
    return Base.invokelatest(_interactive_pointwise_wave_scaled_errors_analytic_impl,
                             GLM,
                             simulation_h,
                             simulation_h2,
                             simulation_h4,
                             start_index,
                             expected_order,
                             analytic_reference,
                             metadata,
                             display_figure)
end

function _interactive_pointwise_wave_scaled_errors_analytic_impl(GLM,
                                                                 simulation_h,
                                                                 simulation_h2,
                                                                 simulation_h4,
                                                                 start_index::Int,
                                                                 expected_order,
                                                                 analytic_reference,
                                                                 metadata,
                                                                 display_figure::Bool)
    pi_data = _pointwise_analytic_error_history(simulation_h, simulation_h2, simulation_h4;
                                                field = :Π,
                                                expected_order = expected_order,
                                                analytic_reference = analytic_reference,
                                                metadata = metadata)
    psi_data = _pointwise_analytic_error_history(simulation_h, simulation_h2, simulation_h4;
                                                 field = :Ψ,
                                                 expected_order = expected_order,
                                                 analytic_reference = analytic_reference,
                                                 metadata = metadata)

    nt = length(pi_data.t)
    1 <= start_index <= nt ||
        throw(ArgumentError("`start_index` must satisfy 1 <= start_index <= $(nt)."))

    GLM.activate!()

    r = Float64.(pi_data.r)
    pi_limits = _finite_limits_with_padding(vcat(vec(pi_data.scaled_err_h),
                                                 vec(pi_data.scaled_err_h2),
                                                 vec(pi_data.scaled_err_h4));
                                            pad_frac = 0.1)
    psi_limits = _finite_limits_with_padding(vcat(vec(psi_data.scaled_err_h),
                                                  vec(psi_data.scaled_err_h2),
                                                  vec(psi_data.scaled_err_h4));
                                             pad_frac = 0.1)

    q = pi_data.expected_order
    fig = GLM.with_theme(mytheme_aps()) do
        fig = GLM.Figure(size = (1500, 920), figure_padding = (28, 24, 22, 18))

        GLM.Label(fig[0, 1:2],
                  latexstring("\\mathrm{Pointwise\\ Analytic\\text{-}Reference\\ Scaled\\ Error\\ Curves}\\quad(q=",
                              string(q), ")");
                  fontsize = 20,
                  font = :bold)

        ax_pi = GLM.Axis(fig[1, 1];
                         xlabel = L"r",
                         ylabel = latexstring("(h/h_0)^{-", string(q), "}\\,(u-u_{\\mathrm{exact}})"),
                         title = L"\Pi")
        ax_psi = GLM.Axis(fig[1, 2];
                          xlabel = L"r",
                          ylabel = latexstring("(h/h_0)^{-", string(q), "}\\,(u-u_{\\mathrm{exact}})"),
                          title = L"\Psi")

        for (ax, limits) in ((ax_pi, pi_limits), (ax_psi, psi_limits))
            GLM.hlines!(ax, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
            GLM.xlims!(ax, minimum(r), maximum(r))
            GLM.ylims!(ax, limits[1], limits[2])
        end

        scaling_label = latexstring("\\mathrm{Scaling:}\\quad",
                                    "(h/h_0)^{-q}(u_h-u_{\\mathrm{exact}}),\\quad",
                                    "(h_{1/2}/h_0)^{-q}(u_{h/2}-u_{\\mathrm{exact}}),\\quad",
                                    "(h_{1/4}/h_0)^{-q}(u_{h/4}-u_{\\mathrm{exact}})")
        GLM.Label(fig[2, 1:2], scaling_label; fontsize = 15)

        factor_label = "Factors: h0=$( @sprintf("%.6g", pi_data.h0) ), " *
                       "coarse=$( @sprintf("%.6g", pi_data.scale_h) ), " *
                       "medium=$( @sprintf("%.6g", pi_data.scale_h2) ), " *
                       "fine=$( @sprintf("%.6g", pi_data.scale_h4) )"
        GLM.Label(fig[3, 1:2], factor_label; fontsize = 15)

        slider = GLM.Slider(fig[5, 1:2];
                            range = 1:nt,
                            startvalue = start_index,
                            snap = true)
        snapshot_index = slider.value

        time_label = GLM.lift(snapshot_index) do idx
            return latexstring("\\mathrm{t = ", @sprintf("%.6g", Float64(pi_data.t[idx])),
                               "\\quad snapshot = ", string(idx), "/", string(nt), "}")
        end
        GLM.Label(fig[4, 1:2], time_label; fontsize = 15)

        pi_h_obs = GLM.lift(idx -> pi_data.scaled_err_h[:, idx], snapshot_index)
        pi_h2_obs = GLM.lift(idx -> pi_data.scaled_err_h2[:, idx], snapshot_index)
        pi_h4_obs = GLM.lift(idx -> pi_data.scaled_err_h4[:, idx], snapshot_index)
        psi_h_obs = GLM.lift(idx -> psi_data.scaled_err_h[:, idx], snapshot_index)
        psi_h2_obs = GLM.lift(idx -> psi_data.scaled_err_h2[:, idx], snapshot_index)
        psi_h4_obs = GLM.lift(idx -> psi_data.scaled_err_h4[:, idx], snapshot_index)

        GLM.lines!(ax_pi, r, pi_h_obs; color = :royalblue4, linewidth = 2.5,
                   label = latexstring(@sprintf("%.6g", pi_data.scale_h), "(u_h-u_{\\mathrm{exact}})"))
        GLM.scatter!(ax_pi, r, pi_h_obs; color = :royalblue4, markersize = 8)
        GLM.lines!(ax_pi, r, pi_h2_obs; color = :darkorange3, linewidth = 2.5,
                   linestyle = :dash,
                   label = latexstring(@sprintf("%.6g", pi_data.scale_h2),
                                       "(u_{h/2}-u_{\\mathrm{exact}})"))
        GLM.scatter!(ax_pi, r, pi_h2_obs; color = :darkorange3, markersize = 8,
                     marker = :rect)
        GLM.lines!(ax_pi, r, pi_h4_obs; color = :mediumpurple4, linewidth = 2.5,
                   linestyle = :dot,
                   label = latexstring(@sprintf("%.6g", pi_data.scale_h4),
                                       "(u_{h/4}-u_{\\mathrm{exact}})"))
        GLM.scatter!(ax_pi, r, pi_h4_obs; color = :mediumpurple4, markersize = 8,
                     marker = :utriangle)
        GLM.axislegend(ax_pi; position = :rb)

        GLM.lines!(ax_psi, r, psi_h_obs; color = :seagreen4, linewidth = 2.5,
                   label = latexstring(@sprintf("%.6g", psi_data.scale_h), "(u_h-u_{\\mathrm{exact}})"))
        GLM.scatter!(ax_psi, r, psi_h_obs; color = :seagreen4, markersize = 8)
        GLM.lines!(ax_psi, r, psi_h2_obs; color = :firebrick, linewidth = 2.5,
                   linestyle = :dash,
                   label = latexstring(@sprintf("%.6g", psi_data.scale_h2),
                                       "(u_{h/2}-u_{\\mathrm{exact}})"))
        GLM.scatter!(ax_psi, r, psi_h2_obs; color = :firebrick, markersize = 8,
                     marker = :rect)
        GLM.lines!(ax_psi, r, psi_h4_obs; color = :goldenrod4, linewidth = 2.5,
                   linestyle = :dot,
                   label = latexstring(@sprintf("%.6g", psi_data.scale_h4),
                                       "(u_{h/4}-u_{\\mathrm{exact}})"))
        GLM.scatter!(ax_psi, r, psi_h4_obs; color = :goldenrod4, markersize = 8,
                     marker = :utriangle)
        GLM.axislegend(ax_psi; position = :rb)

        fig
    end

    screen = display_figure ? GLM.display(fig) : nothing
    return (fig = fig,
            screen = screen,
            pi = pi_data,
            psi = psi_data,
            analytic_reference = pi_data.analytic_reference,
            scaling = (expected_order = q,
                       h0 = pi_data.h0,
                       factors = (pi_data.scale_h, pi_data.scale_h2, pi_data.scale_h4),
                       formula = "(h/h0)^(-q) * (u_h - u_exact)"))
end

"""
    interactive_pointwise_wave_convergence_orders_gundlach(simulation_h, simulation_h2,
                                                           simulation_h4;
                                                           start_index=1,
                                                           expected_order=nothing,
                                                           h0=0.1,
                                                           display_figure=true)

Open an interactive GLMakie viewer for the pointwise observed convergence order on the
common nested coarse grid, computed from Gundlach-style weighted errors. For each
common time snapshot and each common grid point `r_i`, the viewer displays

`q + log2(|e_{u,q}(h)| / |e_{u,q}(h/2)|)`

for both `Π` and `Ψ`, where `e_{u,q}` is the Gundlach-style weighted error profile.
The figure also reports the corresponding Gundlach norm-based order at the selected
time snapshot.
"""
function interactive_pointwise_wave_convergence_orders_gundlach(simulation_h,
                                                                simulation_h2,
                                                                simulation_h4;
                                                                start_index::Int = 1,
                                                                expected_order = nothing,
                                                                h0::Real = 0.1,
                                                                display_figure::Bool = true)
    GLM = Base.invokelatest(_load_glmakie)
    return Base.invokelatest(_interactive_pointwise_wave_convergence_orders_gundlach_impl,
                             GLM,
                             simulation_h,
                             simulation_h2,
                             simulation_h4,
                             start_index,
                             expected_order,
                             h0,
                             display_figure)
end

function _interactive_pointwise_wave_convergence_orders_gundlach_impl(GLM,
                                                                      simulation_h,
                                                                      simulation_h2,
                                                                      simulation_h4,
                                                                      start_index::Int,
                                                                      expected_order,
                                                                      h0::Real,
                                                                      display_figure::Bool)
    pi_data = _gundlach_pointwise_convergence_order_history(simulation_h, simulation_h2,
                                                            simulation_h4;
                                                            field = :Π,
                                                            expected_order = expected_order,
                                                            h0 = h0)
    psi_data = _gundlach_pointwise_convergence_order_history(simulation_h, simulation_h2,
                                                             simulation_h4;
                                                             field = :Ψ,
                                                             expected_order = expected_order,
                                                             h0 = h0)

    nt = length(pi_data.t)
    1 <= start_index <= nt ||
        throw(ArgumentError("`start_index` must satisfy 1 <= start_index <= $(nt)."))

    GLM.activate!()

    r = Float64.(pi_data.r)
    fig = GLM.with_theme(mytheme_aps()) do
        fig = GLM.Figure(size = (1500, 900), figure_padding = (28, 24, 22, 18))

        GLM.Label(fig[0, 1:2],
                  latexstring("\\mathrm{Pointwise\\ Gundlach\\ Observed\\ Convergence\\ Order}\\quad(q=",
                              string(pi_data.expected_order),
                              ",\\ h_0=",
                              @sprintf("%.3g", pi_data.h0),
                              ")");
                  fontsize = 20,
                  font = :bold)

        ax_pi = GLM.Axis(fig[1, 1];
                         xlabel = L"r",
                         ylabel = L"\mathrm{pointwise\ order}",
                         title = L"\Pi")
        ax_psi = GLM.Axis(fig[1, 2];
                          xlabel = L"r",
                          ylabel = L"\mathrm{pointwise\ order}",
                          title = L"\Psi")

        for ax in (ax_pi, ax_psi)
            GLM.hlines!(ax, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
            GLM.hlines!(ax, [pi_data.expected_order];
                        color = :gray40,
                        linewidth = 1.0,
                        linestyle = :dot)
            GLM.xlims!(ax, minimum(r), maximum(r))
            GLM.ylims!(ax, 0.0, max(10.0, pi_data.expected_order + 2.0))
        end

        slider = GLM.Slider(fig[4, 1:2];
                            range = 1:nt,
                            startvalue = start_index,
                            snap = true)
        snapshot_index = slider.value

        time_label = GLM.lift(snapshot_index) do idx
            return latexstring("\\mathrm{t = ", @sprintf("%.6g", Float64(pi_data.t[idx])),
                               "\\quad snapshot = ", string(idx), "/", string(nt), "}")
        end
        GLM.Label(fig[2, 1:2], time_label; fontsize = 15)

        norm_label = GLM.lift(snapshot_index) do idx
            return "Pi: norm order=$( @sprintf("%.4f", Float64(pi_data.norm_order[idx])) )    " *
                   "Psi: norm order=$( @sprintf("%.4f", Float64(psi_data.norm_order[idx])) )"
        end
        GLM.Label(fig[3, 1:2], norm_label; fontsize = 14)

        pi_obs = GLM.lift(idx -> pi_data.order[:, idx], snapshot_index)
        psi_obs = GLM.lift(idx -> psi_data.order[:, idx], snapshot_index)

        GLM.lines!(ax_pi, r, pi_obs; color = :royalblue4, linewidth = 2.5)
        GLM.scatter!(ax_pi, r, pi_obs; color = :royalblue4, markersize = 8)

        GLM.lines!(ax_psi, r, psi_obs; color = :seagreen4, linewidth = 2.5)
        GLM.scatter!(ax_psi, r, psi_obs; color = :seagreen4, markersize = 8)

        fig
    end

    screen = display_figure ? GLM.display(fig) : nothing
    return (fig = fig,
            screen = screen,
            pi = pi_data,
            psi = psi_data)
end

"""
    plot_pointwise_wave_scaled_errors_gundlach(simulation_h, simulation_h2,
                                               simulation_h4; time,
                                               expected_order=nothing,
                                               scale_order=nothing,
                                               h0=0.1)

Plot the static CairoMakie version of
`interactive_pointwise_wave_scaled_errors_gundlach` at the common saved time nearest
to `time`. The plotted quantity is the same signed profile used by the interactive
viewer, `r^(p/2) e_{u,q}`, for the nested pairs `(h, h/2)` and `(h/2, h/4)`.
"""
function plot_pointwise_wave_scaled_errors_gundlach(simulation_h,
                                                    simulation_h2,
                                                    simulation_h4;
                                                    time::Real,
                                                    expected_order = nothing,
                                                    scale_order = nothing,
                                                    h0::Real = 0.1,
                                                    title = nothing)
    pi_data = _gundlach_pointwise_error_history(simulation_h, simulation_h2, simulation_h4;
                                                field = :Π,
                                                expected_order = expected_order,
                                                scale_order = scale_order,
                                                h0 = h0)
    psi_data = _gundlach_pointwise_error_history(simulation_h, simulation_h2, simulation_h4;
                                                 field = :Ψ,
                                                 expected_order = expected_order,
                                                 scale_order = scale_order,
                                                 h0 = h0)
    time_index = _nearest_common_time_index(pi_data.t, time)

    r = Float64.(pi_data.r)
    # ops_h = _ops_from_simulation(simulation_h)
    # hasproperty(ops_h, :p) ||
    #     throw(ArgumentError("The Gundlach scaled-error plot requires an operator with a `p` field."))
    # p = Float64(getproperty(ops_h, :p))
    # radial_weight = r .^ (p / 2)
    pi_err_h_h2_plot = pi_data.err_h_h2
    pi_err_h2_h4_plot = pi_data.err_h2_h4
    psi_err_h_h2_plot = psi_data.err_h_h2
    psi_err_h2_h4_plot = psi_data.err_h2_h4
    pi_err_h_h2_snapshot = pi_err_h_h2_plot[:, time_index]
    pi_err_h2_h4_snapshot = pi_err_h2_h4_plot[:, time_index]
    psi_err_h_h2_snapshot = psi_err_h_h2_plot[:, time_index]
    psi_err_h2_h4_snapshot = psi_err_h2_h4_plot[:, time_index]

    pi_limits = _finite_limits_with_padding(vcat(vec(pi_err_h_h2_plot),
                                                 vec(pi_err_h2_h4_plot)))
    psi_limits = _finite_limits_with_padding(vcat(vec(psi_err_h_h2_plot),
                                                  vec(psi_err_h2_h4_plot)))
    plot_title = isnothing(title) ?
                 latexstring("\\mathrm{Pointwise\\ Error}\\quad(s=",
                             @sprintf("%.6g", pi_data.scale_order),
                             ",\\ h_0=",
                             @sprintf("%.3g", pi_data.h0),
                             ",\\ t=",
                             @sprintf("%.6g", pi_data.t[time_index]),
                             ")") :
                 title

    fig = _with_spectrum_theme() do
        fig = Figure(size = (1020, 500), figure_padding = (92, 16, 8, 8))

        Label(fig[0, 1:2],
              plot_title;
              fontsize = 18,
              font = :bold)

        ax_pi = Axis(fig[1, 1],
                     xlabel = L"r",
                     ylabel = latexstring("e_{u,q}(r,t;h)"),
                     title = L"\Pi")
        ax_psi = Axis(fig[1, 2],
                      xlabel = L"r",
                      ylabel = latexstring("e_{u,q}(r,t;h)"),
                      title = L"\Psi")

        for (ax, limits) in ((ax_pi, pi_limits), (ax_psi, psi_limits))
            hlines!(ax, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
            xlims!(ax, minimum(r), maximum(r))
            ylims!(ax, limits[1], limits[2])
        end

        lines!(ax_pi, r, pi_err_h_h2_snapshot; color = :royalblue4, linewidth = 2.5,
               label = latexstring("e_{\\Pi,q}(h)"))
        scatter!(ax_pi, r, pi_err_h_h2_snapshot; color = :royalblue4, markersize = 7)
        lines!(ax_pi, r, pi_err_h2_h4_snapshot; color = :darkorange3, linewidth = 2.5,
               linestyle = :dash,
               label = latexstring("e_{\\Pi,q}(h/2)"))
        scatter!(ax_pi, r, pi_err_h2_h4_snapshot; color = :darkorange3, markersize = 7,
                 marker = :rect)
        axislegend(ax_pi; position = :rb)

        lines!(ax_psi, r, psi_err_h_h2_snapshot; color = :seagreen4, linewidth = 2.5,
               label = latexstring("e_{\\Psi,q}(h)"))
        scatter!(ax_psi, r, psi_err_h_h2_snapshot; color = :seagreen4, markersize = 7)
        lines!(ax_psi, r, psi_err_h2_h4_snapshot; color = :firebrick, linewidth = 2.5,
               linestyle = :dash,
               label = latexstring("e_{\\Psi,q}(h/2)"))
        scatter!(ax_psi, r, psi_err_h2_h4_snapshot; color = :firebrick, markersize = 7,
                 marker = :rect)
        axislegend(ax_psi; position = :rb)

        fig
    end

    return (fig = fig,
            pi = pi_data,
            psi = psi_data,
            t = pi_data.t[time_index],
            time_index = time_index)
end

"""
    interactive_pointwise_wave_scaled_errors_gundlach(simulation_h, simulation_h2,
                                                      simulation_h4;
                                                      start_index=1,
                                                      expected_order=nothing,
                                                      scale_order=nothing,
                                                      h0=0.1,
                                                      display_figure=true)

Open an interactive GLMakie viewer for the pointwise nested Gundlach error curves on
the common coarse grid. For each common time snapshot and each common grid point
`r_i`, the viewer displays the Gundlach-style weighted profiles

- `e_{u,q}(r_i, t; h)` built from `u_h - u_{h/2}`
- `e_{u,q}(r_i, t; h/2)` built from `u_{h/2} - u_{h/4}`

for both `Π` and `Ψ`. The figure also reports the corresponding Gundlach norms at the
selected time snapshot.
"""
function interactive_pointwise_wave_scaled_errors_gundlach(simulation_h,
                                                           simulation_h2,
                                                           simulation_h4;
                                                           start_index::Int = 1,
                                                           expected_order = nothing,
                                                           scale_order = nothing,
                                                           h0::Real = 0.1,
                                                           display_figure::Bool = true)
    GLM = Base.invokelatest(_load_glmakie)
    return Base.invokelatest(_interactive_pointwise_wave_scaled_errors_gundlach_impl,
                             GLM,
                             simulation_h,
                             simulation_h2,
                             simulation_h4,
                             start_index,
                             expected_order,
                             scale_order,
                             h0,
                             display_figure)
end

function _interactive_pointwise_wave_scaled_errors_gundlach_impl(GLM,
                                                                 simulation_h,
                                                                 simulation_h2,
                                                                 simulation_h4,
                                                                 start_index::Int,
                                                                 expected_order,
                                                                 scale_order,
                                                                 h0::Real,
                                                                 display_figure::Bool)
    pi_data = _gundlach_pointwise_error_history(simulation_h, simulation_h2, simulation_h4;
                                                field = :Π,
                                                expected_order = expected_order,
                                                scale_order = scale_order,
                                                h0 = h0)
    psi_data = _gundlach_pointwise_error_history(simulation_h, simulation_h2, simulation_h4;
                                                 field = :Ψ,
                                                 expected_order = expected_order,
                                                 scale_order = scale_order,
                                                 h0 = h0)

    nt = length(pi_data.t)
    1 <= start_index <= nt ||
        throw(ArgumentError("`start_index` must satisfy 1 <= start_index <= $(nt)."))

    GLM.activate!()

    r = Float64.(pi_data.r)
    # ops_h = _ops_from_simulation(simulation_h)
    # hasproperty(ops_h, :p) ||
    #     throw(ArgumentError("The Gundlach scaled-error viewer requires an operator with a `p` field."))
    # p = Float64(getproperty(ops_h, :p))
    # radial_weight = r .^ (p / 2)
    pi_err_h_h2_plot = pi_data.err_h_h2
    pi_err_h2_h4_plot = pi_data.err_h2_h4
    psi_err_h_h2_plot = psi_data.err_h_h2
    psi_err_h2_h4_plot = psi_data.err_h2_h4

    pi_limits = _finite_limits_with_padding(vcat(vec(pi_err_h_h2_plot),
                                                 vec(pi_err_h2_h4_plot)))
    psi_limits = _finite_limits_with_padding(vcat(vec(psi_err_h_h2_plot),
                                                  vec(psi_err_h2_h4_plot)))

    fig = GLM.with_theme(mytheme_aps()) do
        fig = GLM.Figure(size = (1500, 920), figure_padding = (28, 24, 22, 18))

        GLM.Label(fig[0, 1:2],
                  latexstring("\\mathrm{Pointwise\\ Gundlach\\ Error}\\quad(s=",
                              @sprintf("%.6g", pi_data.scale_order),
                              ",\\ h_0=",
                              @sprintf("%.3g", pi_data.h0),
                              ")");
                  fontsize = 20,
                  font = :bold)

        ax_pi = GLM.Axis(fig[1, 1];
                         xlabel = L"r",
                         ylabel = latexstring("e_{u,q}(r,t;h)"),
                         title = L"\Pi")
        ax_psi = GLM.Axis(fig[1, 2];
                          xlabel = L"r",
                          ylabel = latexstring("e_{u,q}(r,t;h)"),
                          title = L"\Psi")

        for (ax, limits) in ((ax_pi, pi_limits), (ax_psi, psi_limits))
            GLM.hlines!(ax, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
            GLM.xlims!(ax, minimum(r), maximum(r))
            GLM.ylims!(ax, limits[1], limits[2])
        end

        slider = GLM.Slider(fig[4, 1:2];
                            range = 1:nt,
                            startvalue = start_index,
                            snap = true)
        snapshot_index = slider.value

        time_label = GLM.lift(snapshot_index) do idx
            return latexstring("\\mathrm{t = ", @sprintf("%.6g", Float64(pi_data.t[idx])),
                               "\\quad snapshot = ", string(idx), "/", string(nt), "}")
        end
        GLM.Label(fig[2, 1:2], time_label; fontsize = 15)

        pi_err12_obs = GLM.lift(idx -> pi_err_h_h2_plot[:, idx], snapshot_index)
        pi_err24_obs = GLM.lift(idx -> pi_err_h2_h4_plot[:, idx], snapshot_index)
        psi_err12_obs = GLM.lift(idx -> psi_err_h_h2_plot[:, idx], snapshot_index)
        psi_err24_obs = GLM.lift(idx -> psi_err_h2_h4_plot[:, idx], snapshot_index)

        GLM.lines!(ax_pi, r, pi_err12_obs; color = :royalblue4, linewidth = 2.5,
                   label = latexstring("e_{\\Pi,q}(h)"))
        GLM.scatter!(ax_pi, r, pi_err12_obs; color = :royalblue4, markersize = 8)
        GLM.lines!(ax_pi, r, pi_err24_obs; color = :darkorange3, linewidth = 2.5,
                   linestyle = :dash,
                   label = latexstring("e_{\\Pi,q}(h/2)"))
        GLM.scatter!(ax_pi, r, pi_err24_obs; color = :darkorange3, markersize = 8,
                     marker = :rect)
        GLM.axislegend(ax_pi; position = :rb)

        GLM.lines!(ax_psi, r, psi_err12_obs; color = :seagreen4, linewidth = 2.5,
                   label = latexstring("e_{\\Psi,q}(h)"))
        GLM.scatter!(ax_psi, r, psi_err12_obs; color = :seagreen4, markersize = 8)
        GLM.lines!(ax_psi, r, psi_err24_obs; color = :firebrick, linewidth = 2.5,
                   linestyle = :dash,
                   label = latexstring("e_{\\Psi,q}(h/2)"))
        GLM.scatter!(ax_psi, r, psi_err24_obs; color = :firebrick, markersize = 8,
                     marker = :rect)
        GLM.axislegend(ax_psi; position = :rb)

        fig
    end

    screen = display_figure ? GLM.display(fig) : nothing
    return (fig = fig,
            screen = screen,
            pi = pi_data,
            psi = psi_data)
end

"""
    interactive_pointwise_wave_scaled_errors_gundlach_reference(simulation_h,
                                                                simulation_h2,
                                                                simulation_ref;
                                                                start_index=1,
                                                                expected_order=nothing,
                                                                scale_order=nothing,
                                                                h0=0.1,
                                                                display_figure=true)

Open an interactive GLMakie viewer for Gundlach-style pointwise error curves built
against a shared fine reference solution instead of consecutive resolution
differences.

For each common saved time and each coarse-grid node `r_i`, the viewer displays

- `e_{u,q}(r_i, t; h)` built from `u_h - u_ref`
- `e_{u,q}(r_i, t; h/2)` built from `u_{h/2} - u_ref`

with both `u_{h/2}` and `u_ref` restricted exactly onto the coarse grid.
"""
function interactive_pointwise_wave_scaled_errors_gundlach_reference(simulation_h,
                                                                     simulation_h2,
                                                                     simulation_ref;
                                                                     start_index::Int = 1,
                                                                     expected_order = nothing,
                                                                     scale_order = nothing,
                                                                     h0::Real = 0.1,
                                                                     display_figure::Bool = true)
    GLM = Base.invokelatest(_load_glmakie)
    return Base.invokelatest(_interactive_pointwise_wave_scaled_errors_gundlach_reference_impl,
                             GLM,
                             simulation_h,
                             simulation_h2,
                             simulation_ref,
                             start_index,
                             expected_order,
                             scale_order,
                             h0,
                             display_figure)
end

function _interactive_pointwise_wave_scaled_errors_gundlach_reference_impl(GLM,
                                                                           simulation_h,
                                                                           simulation_h2,
                                                                           simulation_ref,
                                                                           start_index::Int,
                                                                           expected_order,
                                                                           scale_order,
                                                                           h0::Real,
                                                                           display_figure::Bool)
    pi_data = _gundlach_pointwise_error_history_with_reference(simulation_h,
                                                               simulation_h2,
                                                               simulation_ref;
                                                               field = :Π,
                                                               expected_order = expected_order,
                                                               scale_order = scale_order,
                                                               h0 = h0)
    psi_data = _gundlach_pointwise_error_history_with_reference(simulation_h,
                                                                simulation_h2,
                                                                simulation_ref;
                                                                field = :Ψ,
                                                                expected_order = expected_order,
                                                                scale_order = scale_order,
                                                                h0 = h0)

    nt = length(pi_data.t)
    1 <= start_index <= nt ||
        throw(ArgumentError("`start_index` must satisfy 1 <= start_index <= $(nt)."))

    GLM.activate!()

    ops_h = _ops_from_simulation(simulation_h)
    hasproperty(ops_h, :p) ||
        throw(ArgumentError("The Gundlach scaled-error viewer requires an operator with a `p` field."))

    r = Float64.(pi_data.r)
    p = Float64(getproperty(ops_h, :p))
    radial_weight = r .^ (p / 2)
    pi_err_h_ref_plot = pi_data.err_h_ref .* radial_weight
    pi_err_h2_ref_plot = pi_data.err_h2_ref .* radial_weight
    psi_err_h_ref_plot = psi_data.err_h_ref .* radial_weight
    psi_err_h2_ref_plot = psi_data.err_h2_ref .* radial_weight

    pi_limits = _finite_limits_with_padding(vcat(vec(pi_err_h_ref_plot),
                                                 vec(pi_err_h2_ref_plot)))
    psi_limits = _finite_limits_with_padding(vcat(vec(psi_err_h_ref_plot),
                                                  vec(psi_err_h2_ref_plot)))

    fig = GLM.with_theme(mytheme_aps()) do
        fig = GLM.Figure(size = (1500, 920), figure_padding = (28, 24, 22, 18))

        GLM.Label(fig[0, 1:2],
                  latexstring("\\mathrm{Pointwise\\ Gundlach\\ Error\\ vs\\ Fine\\ Reference}\\quad(s=",
                              @sprintf("%.6g", pi_data.scale_order),
                              ",\\ h_0=",
                              @sprintf("%.3g", pi_data.h0),
                              ",\\ p=",
                              @sprintf("%.3g", p),
                              ")");
                  fontsize = 20,
                  font = :bold)

        ax_pi = GLM.Axis(fig[1, 1];
                         xlabel = L"r",
                         ylabel = latexstring("r^{p/2} e_{u,q}(r,t;h)"),
                         title = L"\Pi")
        ax_psi = GLM.Axis(fig[1, 2];
                          xlabel = L"r",
                          ylabel = latexstring("r^{p/2} e_{u,q}(r,t;h)"),
                          title = L"\Psi")

        for (ax, limits) in ((ax_pi, pi_limits), (ax_psi, psi_limits))
            GLM.hlines!(ax, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
            GLM.xlims!(ax, minimum(r), maximum(r))
            GLM.ylims!(ax, limits[1], limits[2])
        end

        slider = GLM.Slider(fig[4, 1:2];
                            range = 1:nt,
                            startvalue = start_index,
                            snap = true)
        snapshot_index = slider.value

        time_label = GLM.lift(snapshot_index) do idx
            return latexstring("\\mathrm{t = ", @sprintf("%.6g", Float64(pi_data.t[idx])),
                               "\\quad snapshot = ", string(idx), "/", string(nt), "}")
        end
        GLM.Label(fig[2, 1:2], time_label; fontsize = 15)

        norm_label = GLM.lift(snapshot_index) do idx
            return "Pi: coarse-grid norms vs ref = (" *
                   @sprintf("%.4e", Float64(pi_data.norm_h_ref[idx])) *
                   ", " *
                   @sprintf("%.4e", Float64(pi_data.norm_h2_ref[idx])) *
                   ")    Psi: coarse-grid norms vs ref = (" *
                   @sprintf("%.4e", Float64(psi_data.norm_h_ref[idx])) *
                   ", " *
                   @sprintf("%.4e", Float64(psi_data.norm_h2_ref[idx])) *
                   ")"
        end
        GLM.Label(fig[3, 1:2], norm_label; fontsize = 14)

        pi_err_h_obs = GLM.lift(idx -> pi_err_h_ref_plot[:, idx], snapshot_index)
        pi_err_h2_obs = GLM.lift(idx -> pi_err_h2_ref_plot[:, idx], snapshot_index)
        psi_err_h_obs = GLM.lift(idx -> psi_err_h_ref_plot[:, idx], snapshot_index)
        psi_err_h2_obs = GLM.lift(idx -> psi_err_h2_ref_plot[:, idx], snapshot_index)

        GLM.lines!(ax_pi, r, pi_err_h_obs; color = :royalblue4, linewidth = 2.5,
                   label = latexstring("r^{p/2} e_{\\Pi,q}(h;\\mathrm{ref})"))
        GLM.scatter!(ax_pi, r, pi_err_h_obs; color = :royalblue4, markersize = 8)
        GLM.lines!(ax_pi, r, pi_err_h2_obs; color = :darkorange3, linewidth = 2.5,
                   linestyle = :dash,
                   label = latexstring("r^{p/2} e_{\\Pi,q}(h/2;\\mathrm{ref})"))
        GLM.scatter!(ax_pi, r, pi_err_h2_obs; color = :darkorange3, markersize = 8,
                     marker = :rect)
        GLM.axislegend(ax_pi; position = :rb)

        GLM.lines!(ax_psi, r, psi_err_h_obs; color = :seagreen4, linewidth = 2.5,
                   label = latexstring("r^{p/2} e_{\\Psi,q}(h;\\mathrm{ref})"))
        GLM.scatter!(ax_psi, r, psi_err_h_obs; color = :seagreen4, markersize = 8)
        GLM.lines!(ax_psi, r, psi_err_h2_obs; color = :firebrick, linewidth = 2.5,
                   linestyle = :dash,
                   label = latexstring("r^{p/2} e_{\\Psi,q}(h/2;\\mathrm{ref})"))
        GLM.scatter!(ax_psi, r, psi_err_h2_obs; color = :firebrick, markersize = 8,
                     marker = :rect)
        GLM.axislegend(ax_psi; position = :rb)

        fig
    end

    screen = display_figure ? GLM.display(fig) : nothing
    return (fig = fig,
            screen = screen,
            pi = pi_data,
            psi = psi_data)
end
