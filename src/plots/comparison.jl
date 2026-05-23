const COMPARISON_AXIS_LABEL_SIZE = 24
const COMPARISON_TICK_LABEL_SIZE = 20
const COMPARISON_PANEL_TITLE_SIZE = 24
const COMPARISON_FIGURE_TITLE_SIZE = 28
const COMPARISON_LEGEND_LABEL_SIZE = 24

@inline _odd_monomial(r::AbstractVector, power::Int) = Float64.(r) .^ power

@inline _odd_monomial_divergence_exact(r::AbstractVector, power::Int, p::Int) = (power + p) .*
                                                                                (Float64.(r) .^
                                                                                 (power - 1))

@inline _analytic_profile(r::AbstractVector) = Float64.(r) .* exp.(-(Float64.(r) .^ 2))

@inline _analytic_profile_divergence_exact(r::AbstractVector) = (3 .-
                                                                 2 .* (Float64.(r) .^ 2)) .*
                                                                exp.(-(Float64.(r) .^ 2))

const DIVERGENCE_COMPARISON_DIR = joinpath(@__DIR__, "..", "..", "plots", "div_comparison")

function _comparison_profile_data(r::AbstractVector, profile::Symbol, power::Int, p::Int)
    if profile === :monomial
        return (u = _odd_monomial(r, power),
                exact = _odd_monomial_divergence_exact(r, power, p))
    elseif profile === :analytic
        return (u = _analytic_profile(r),
                exact = _analytic_profile_divergence_exact(r))
    end
    throw(ArgumentError("Unsupported comparison profile `$profile`. Use `:monomial` or `:analytic`."))
end

function _comparison_title(profile::Symbol, power::Int, h::Real, p::Int)
    if profile === :monomial
        return latexstring("\\mathrm{D}(r^{$(power)}),\\ dr = ", string(Float64(h)),
                           ",\\ p = ", string(p))
    elseif profile === :analytic
        return latexstring("\\mathrm{D}(r e^{-r^2}) = (3 - 2r^2)e^{-r^2},\\ dr = ",
                           string(Float64(h)), ",\\ p = ", string(p))
    end
    throw(ArgumentError("Unsupported comparison profile `$profile`. Use `:monomial` or `:analytic`."))
end

function _comparison_value_ylabel(profile::Symbol, power::Int)
    if profile === :monomial
        return latexstring("\\mathrm{D}\\, r^{$(power)}")
    elseif profile === :analytic
        return L"\mathrm{D}\!\left(r e^{-r^2}\right)"
    end
    throw(ArgumentError("Unsupported comparison profile `$profile`. Use `:monomial` or `:analytic`."))
end

function _comparison_error_ylabel(profile::Symbol, power::Int, p::Int)
    if profile === :monomial
        return latexstring("\\mathrm{D}\\, r^{$(power)} - ", string(power + p), "r^{$(power - 1)}")
    elseif profile === :analytic
        return L"\mathrm{D}\!\left(r e^{-r^2}\right) - (3 - 2r^2)e^{-r^2}"
    end
    throw(ArgumentError("Unsupported comparison profile `$profile`. Use `:monomial` or `:analytic`."))
end

function _comparison_panel_title(label::AbstractString)
    latex_label = replace(label, " " => "\\ ")
    return latexstring("\\mathrm{", latex_label, "}")
end

function _comparison_filename(profile::Symbol,
                              power::Int,
                              h::Real,
                              p::Int,
                              accuracy_order::Int,
                              npoints::Int)
    profile_tag = profile === :monomial ? "monomial_r$(power)" : "analytic"
    h_token = replace(string(Float64(h)), "." => "p")
    return "divergence_comparison_$(profile_tag)_order$(accuracy_order)_h$(h_token)_p$(p)_n$(npoints).pdf"
end

function _padded_plot_limits(values::AbstractVector{<:Real}; pad_frac::Float64 = 0.10)
    vmin, vmax = extrema(values)
    span = vmax - vmin
    scale = max(abs(vmin), abs(vmax), 1.0)
    pad = pad_frac * max(span, scale)
    return (vmin - pad, vmax + pad)
end

function _error_plot_limits(values::AbstractVector{<:Real}; pad_frac::Float64 = 0.10)
    vmin, vmax = extrema(values)
    span = vmax - vmin
    scale = max(span, abs(vmin), abs(vmax), eps(Float64))
    pad = pad_frac * scale
    return (vmin - pad, vmax + pad)
end

function _nonnegative_plot_limits(values::AbstractVector{<:Real}; pad_frac::Float64 = 0.04)
    vmin, vmax = extrema(values)
    span = vmax - vmin
    scale = max(span, abs(vmax), eps(Float64))
    pad = pad_frac * scale
    return (max(0.0, vmin), vmax + pad)
end

function _nice_tick_step(span::Real, nticks::Int)
    nticks >= 2 || throw(ArgumentError("`nticks` must be at least 2."))
    raw = Float64(span) / (nticks - 1)
    raw > 0 || return 1.0

    magnitude = 10.0 ^ floor(log10(raw))
    scaled = raw / magnitude
    nice_scaled = scaled <= 1 ? 1.0 :
                  scaled <= 2 ? 2.0 :
                  scaled <= 2.5 ? 2.5 :
                  scaled <= 5 ? 5.0 : 10.0
    return nice_scaled * magnitude
end

function _nice_shared_ticks(vmin::Real, vmax::Real; nticks::Int = 5)
    span = Float64(vmax) - Float64(vmin)
    step = _nice_tick_step(span, nticks)

    lo = floor(Float64(vmin) / step) * step
    hi = ceil(Float64(vmax) / step) * step

    # Force 0 into the common ticks and prefer a compact 4-6 tick set.
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)

    ticks = collect(lo:step:hi)
    if 0.0 ∉ ticks
        push!(ticks, 0.0)
        sort!(ticks)
    end

    while length(ticks) > 6
        step *= 2
        lo = floor(Float64(vmin) / step) * step
        hi = ceil(Float64(vmax) / step) * step
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        ticks = collect(lo:step:hi)
        if 0.0 ∉ ticks
            push!(ticks, 0.0)
            sort!(ticks)
        end
    end

    while length(ticks) < 4
        step /= 2
        lo = floor(Float64(vmin) / step) * step
        hi = ceil(Float64(vmax) / step) * step
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        ticks = collect(lo:step:hi)
        if 0.0 ∉ ticks
            push!(ticks, 0.0)
            sort!(ticks)
        end
    end

    return unique(round.(ticks; digits = 12))
end

function _comparison_shared_top_row_axis(data; plot_points, nticks::Int = 5)
    nticks >= 4 || throw(ArgumentError("`nticks` must be at least 4."))

    xvals = reduce(vcat, [entry.r[_plot_indices(entry.r, plot_points)] for entry in data])
    yvals = reduce(vcat,
                   [vcat(entry.exact[_plot_indices(entry.r, plot_points)],
                         entry.approx[_plot_indices(entry.r, plot_points)]) for entry in data])

    xlo, xhi = _nonnegative_plot_limits(xvals)
    ylo_raw, yhi = _padded_plot_limits(yvals)
    ylo = min(ylo_raw, 0.0)
    ticks = _nice_shared_ticks(ylo, yhi; nticks = nticks)

    return (xlims = (xlo, xhi), ylims = (ylo, yhi), yticks = ticks)
end

@inline function _plot_indices(values::AbstractVector, plot_points::Union{Nothing, Int})
    plot_points === nothing && return eachindex(values)
    n = min(length(values), plot_points)
    return Base.OneTo(n)
end

function _error_curve_xy(rvals::AbstractVector{<:Real}, errvals::AbstractVector{<:Real})
    isempty(rvals) && return (Float64[], Float64[])

    rplot = Float64.(rvals)
    eplot = Float64.(errvals)
    first(rplot) > 0 || return (rplot, eplot)
    return (vcat(0.0, rplot), vcat(first(eplot), eplot))
end

function _try_family_construction(builder::Function, label::AbstractString, color)
    try
        ops = builder()
        return (label = label, color = color, ops = ops, skipped = false,
                error_message = nothing)
    catch err
        @warn "Skipping $label divergence comparison because operator construction failed." exception=(err,
                                                                                                       catch_backtrace())
        return (label = label, color = color, ops = nothing, skipped = true,
                error_message = sprint(showerror, err))
    end
end

"""
    plot_divergence_comparison(; power, h, p=2, accuracy_order=4, npoints=30, plot_points=5, profile=:monomial, source=MattssonNordström2004(), mode=SafeMode())

Compare the diagonal, mixed-order diagonal, staggered, and non-diagonal divergence
operators on either the odd monomial `u(r) = r^power` or the analytic profile
`u(r) = r e^{-r^2}`.

The comparison constructs operators with `npoints` grid points in each family, then
plots only the first `plot_points` near-origin samples by default. The supplied
spacing `h` therefore determines the corresponding domain size in each constructor.
"""
function plot_divergence_comparison(;
                                    power::Int,
                                    h::Real,
                                    p::Int = 2,
                                    accuracy_order::Int = 4,
                                    npoints::Int = 30,
                                    plot_points::Union{Nothing, Int} = 5,
                                    profile::Symbol = :monomial,
                                    source = MattssonNordström2004(),
                                    mode = SafeMode(),
                                    display_figure::Bool = true,
                                    save_figure::Bool = true,
                                    out_dir::AbstractString = DIVERGENCE_COMPARISON_DIR,
                                    filename::Union{Nothing, AbstractString} = nothing)
    if profile === :monomial
        isodd(power) ||
            throw(ArgumentError("`power` must be odd for `profile = :monomial`."))
        power >= 1 ||
            throw(ArgumentError("`power` must be positive for `profile = :monomial`."))
    elseif profile !== :analytic
        throw(ArgumentError("Unsupported comparison profile `$profile`. Use `:monomial` or `:analytic`."))
    end
    h > 0 || throw(ArgumentError("`h` must be positive."))
    # For diagonal and non-diagonal operators the folded grid includes the origin and
    # outer boundary, so `npoints` physical points means `R = (npoints - 1)h`.
    # For staggered operators the folded grid uses half-offset points
    # `h/2, 3h/2, ..., (2npoints-1)h/2`, so `npoints` points with spacing `h`
    # means `R = (npoints - 0.5)h`.
    R_collocated = (npoints - 1) * Float64(h)
    R_staggered = (npoints - 0.5) * Float64(h)

    attempted_families = (_try_family_construction("Diagonal", :royalblue4) do
                              diagonal_spherical_operators(source;
                                                  accuracy_order = accuracy_order,
                                                  N = npoints - 1,
                                                  R = R_collocated,
                                                  p = p,
                                                  mode = mode)
                          end,
                          _try_family_construction("Mixed-order diagonal", :mediumpurple4) do
                              mixed_order_diagonal_spherical_operators(source;
                                                                       accuracy_order = accuracy_order,
                                                                       N = npoints - 1,
                                                                       R = Float64(R_collocated),
                                                                       p = p,
                                                                       mode = mode)
                          end,
                          _try_family_construction("Staggered", :darkorange3) do
                              staggered_spherical_operators(source;
                                                            accuracy_order = accuracy_order,
                                                            N = npoints,
                                                            R = R_staggered,
                                                            p = p,
                                                            mode = mode)
                          end,
                          _try_family_construction("Non-diagonal", :seagreen4) do
                              non_diagonal_spherical_operators(source;
                                                               accuracy_order = accuracy_order,
                                                               N = npoints,
                                                               h = h,
                                                               p = p,
                                                               mode = mode)
                          end)

    families = collect(filter(family -> !family.skipped, attempted_families))
    isempty(families) &&
        throw(ArgumentError("All operator constructions failed for the requested divergence comparison."))

    data = map(families) do family
        r = Float64.(family.ops.r)
        profile_data = _comparison_profile_data(r, profile, power, p)
        u = profile_data.u
        approx = Float64.(apply_divergence(family.ops, u))
        exact = profile_data.exact
        error = approx .- exact
        idx = _plot_indices(r, plot_points)
        return (label = family.label,
                color = family.color,
                r = r,
                approx = approx,
                exact = exact,
                error = error,
                max_abs_error = maximum(abs.(error)),
                plotted_max_abs_error = maximum(abs.(error[idx])))
    end

    top_row_axis = _comparison_shared_top_row_axis(data; plot_points = plot_points)

    fig = _with_spectrum_theme() do
        nfamilies = length(data)
        fig = Figure(size = (420 * nfamilies, 760))

        for (j, entry) in enumerate(data)
            idx = _plot_indices(entry.r, plot_points)
            ax = Axis(fig[1, j],
                      xlabel = L"r",
                      ylabel = j == 1 ? _comparison_value_ylabel(profile, power) : "",
                      yticks = top_row_axis.yticks,
                      title = _comparison_panel_title(entry.label),
                      xlabelsize = COMPARISON_AXIS_LABEL_SIZE,
                      ylabelsize = COMPARISON_AXIS_LABEL_SIZE,
                      xticklabelsize = COMPARISON_TICK_LABEL_SIZE,
                      yticklabelsize = COMPARISON_TICK_LABEL_SIZE,
                      titlesize = COMPARISON_PANEL_TITLE_SIZE)
            lines!(ax, entry.r[idx], entry.exact[idx]; color = :black,
                   linewidth = 2.0,
                   linestyle = :dash, label = "exact")
            scatter!(ax, entry.r[idx], entry.approx[idx];
                     color = entry.color,
                     markersize = 10,
                     label = "discrete")
            lines!(ax, entry.r[idx], entry.approx[idx];
                   color = entry.color,
                   linewidth = 2.0)
            axislegend(ax;
                       position = :lt,
                       labelsize = COMPARISON_LEGEND_LABEL_SIZE,
                       margin = (18, 0, 0, 0))

            xlims!(ax, top_row_axis.xlims...)
            ylims!(ax, top_row_axis.ylims...)
        end

        ax_err = Axis(fig[2, 1:nfamilies],
                      xlabel = L"r",
                      ylabel = _comparison_error_ylabel(profile, power, p),
                      title = L"\mathrm{Error}",
                      xlabelsize = COMPARISON_AXIS_LABEL_SIZE,
                      ylabelsize = COMPARISON_AXIS_LABEL_SIZE,
                      xticklabelsize = COMPARISON_TICK_LABEL_SIZE,
                      yticklabelsize = COMPARISON_TICK_LABEL_SIZE,
                      titlesize = COMPARISON_PANEL_TITLE_SIZE)
        hlines!(ax_err, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
        for entry in data
            idx = _plot_indices(entry.r, plot_points)
            line_r, line_err = _error_curve_xy(entry.r[idx], entry.error[idx])
            lines!(ax_err, line_r, line_err;
                   linewidth = 2.5,
                   color = entry.color,
                   label = "$(entry.label) (max = $(round(entry.plotted_max_abs_error; sigdigits = 4)))")
            scatter!(ax_err, entry.r[idx], entry.error[idx];
                     color = entry.color,
                     markersize = 8)
        end
        legend = axislegend(ax_err;
                            position = :cb,
                            orientation = :horizontal,
                            labelsize = COMPARISON_LEGEND_LABEL_SIZE,
                            tellheight = true,
                            tellwidth = false,
                            framevisible = false,
                            padding = (0, 0, 0, 0),
                            margin = (0, 0, 12, 0))

        err_x = reduce(vcat,
                       [_error_curve_xy(entry.r[_plot_indices(entry.r, plot_points)],
                                        entry.error[_plot_indices(entry.r, plot_points)])[1]
                        for entry in data])
        err_y = reduce(vcat,
                       [entry.error[_plot_indices(entry.r, plot_points)] for entry in data])
        xlo, xhi = _nonnegative_plot_limits(err_x)
        ylo, yhi = _error_plot_limits(vcat(err_y, [0.0]))
        xlims!(ax_err, xlo, xhi)
        ylims!(ax_err, ylo, yhi)
        fig[3, 1:nfamilies] = legend

        Label(fig[0, 1:nfamilies],
              _comparison_title(profile, power, h, p);
              fontsize = COMPARISON_FIGURE_TITLE_SIZE,
              font = :bold)
        fig
    end

    display_figure && display(fig)
    save_path = nothing
    if save_figure
        mkpath(out_dir)
        plot_filename = isnothing(filename) ? _comparison_filename(profile,
                                                                   power,
                                                                   h,
                                                                   p,
                                                                   accuracy_order,
                                                                   npoints) :
                        String(filename)
        save_path = joinpath(out_dir, plot_filename)
        save(save_path, fig)
    end

    return (fig = fig, families = data, save_path = save_path)
end
