using CairoMakie
using LaTeXStrings: @L_str, latexstring
using Printf: @printf, @sprintf

# Reuse the exact same banded-operator construction and high-precision
# spectrum helpers used for the table generation pipeline.
include(joinpath(@__DIR__, "generate_banded_spectrum_tables.jl"))

if !isdefined(@__MODULE__, :SphericalSBPOperators)
    @eval using SphericalSBPOperators
end

const _APS_REVTeX_COLUMNWIDTH_PT = 246.0
const _APS_REVTeX_TEXTWIDTH_PT = 510.0
const _REF_BG = RGBf(1.0, 1.0, 1.0)
const _REF_GRID = RGBAf(0.0, 0.0, 0.0, 0.10)
const _REF_ZERO = RGBAf(0.0, 0.0, 0.0, 0.35)
const _DEFAULT_PAPER_PLOTS_DIR = "/home/svretina/PhD/mypapers/Spherical-SBP-Operators-Paper/plots"
const _REF_COLORS = [RGBAf(0.298039, 0.447059, 0.690196, 1.0),
                     RGBAf(0.866667, 0.517647, 0.321569, 1.0),
                     RGBAf(0.333333, 0.658824, 0.407843, 1.0),
                     RGBAf(0.768627, 0.305882, 0.321569, 1.0),
                     RGBAf(0.505882, 0.447059, 0.701961, 1.0),
                     RGBAf(0.576471, 0.470588, 0.376471, 1.0),
                     RGBAf(0.854902, 0.545098, 0.764706, 1.0),
                     RGBAf(0.54902, 0.54902, 0.54902, 1.0),
                     RGBAf(0.8, 0.72549, 0.454902, 1.0),
                     RGBAf(0.392157, 0.709804, 0.803922, 1.0)]
const _REF_MARKERS = [:circle, :rect, :dtriangle, :utriangle, :cross,
                      :diamond, :ltriangle, :rtriangle, :pentagon,
                      :xcross, :hexagon]

@inline function _with_custom_theme(f::Function; kwargs...)
    return with_theme(SphericalSBPOperators.mytheme_aps()) do
        with_theme(SphericalSBPOperators.mytheme_aps_spectrum(; kwargs...)) do
            f()
        end
    end
end

function _revtex_aps_figure_size(width_mode::Symbol, aspect_ratio::Float64)
    width_pt = if width_mode == :onecolumn
        _APS_REVTeX_COLUMNWIDTH_PT
    elseif width_mode == :twocolumn
        _APS_REVTeX_TEXTWIDTH_PT
    else
        throw(ArgumentError("Unsupported width_mode `$width_mode`; use `:onecolumn` or `:twocolumn`."))
    end

    height_pt = aspect_ratio * width_pt
    return (round(Int, width_pt), round(Int, height_pt))
end

@inline function _to_complex64(vals::AbstractVector{<:Complex};
                               tiny_zero_tol::Float64 = 1e-16)
    out = Vector{ComplexF64}(undef, length(vals))
    @inbounds for i in eachindex(vals)
        re = Float64(real(vals[i]))
        im = Float64(imag(vals[i]))
        if abs(re) < tiny_zero_tol
            re = 0.0
        end
        if abs(im) < tiny_zero_tol
            im = 0.0
        end
        out[i] = complex(re, im)
    end
    return out
end

function _axis_limits(series::AbstractVector; pad_frac::Float64 = 0.10)
    xall = Float64[]
    yall = Float64[]
    for row in series
        append!(xall, real.(row.eigvals))
        append!(yall, imag.(row.eigvals))
    end
    isempty(xall) && throw(ArgumentError("No eigenvalues were collected for plotting."))

    xmin, xmax = extrema(xall)
    ymin, ymax = extrema(yall)
    dx = xmax - xmin
    dy = ymax - ymin
    xpad = dx > 0 ? pad_frac * dx : pad_frac * max(abs(xmin), abs(xmax), 1.0)
    ypad = dy > 0 ? pad_frac * dy : pad_frac * max(abs(ymin), abs(ymax), 1.0)
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad
end

@inline _plot_short_name(label::String) = get(_SOURCE_TO_SHORT, label, label)
@inline _legend_entry(label::String) = latexstring("\\mathrm{", label, "}")

@inline function _series_style(idx::Int)
    return (color = _REF_COLORS[mod1(idx, length(_REF_COLORS))],
            marker = _REF_MARKERS[mod1(idx, length(_REF_MARKERS))])
end

function _sort_series!(series::Vector)
    sort!(series;
          by = row -> (get(_SHORT_SORT_INDEX, row.short_name, 10_000), row.source_label))
    return series
end

function _build_diagonal_ops(source, order::Int;
                             points::Int,
                             R::Rational{BigInt},
                             p::Int,
                             mode)
    return SphericalSBPOperators.diagonal_spherical_operators(source;
                                                     accuracy_order = order,
                                                     N = points,
                                                     R = R,
                                                     p = p,
                                                     mode = mode)
end

@inline function _method_tag(method::Symbol)
    method === :banded && return "banded"
    method === :diagonal && return "diagonal"
    throw(ArgumentError("Unsupported method `$method`; use `:banded` or `:diagonal`."))
end

@inline function _default_out_dir(method::Symbol)
    method === :banded && return joinpath("plots", "spectra", "non-diagonal")
    method === :diagonal && return joinpath("plots", "spectra", "diagonal")
    throw(ArgumentError("Unsupported method `$method`; use `:banded` or `:diagonal`."))
end

@inline function _method_subdir(method::Symbol)
    method === :banded && return joinpath("spectra", "non-diagonal")
    method === :diagonal && return joinpath("spectra", "diagonal")
    throw(ArgumentError("Unsupported method `$method`; use `:banded` or `:diagonal`."))
end

@inline function _R_token(R::Rational{BigInt})
    den = denominator(R)
    num = numerator(R)
    return den == 1 ? string(num) : "$(num)d$(den)"
end

function _export_plots_to_dir(paths::Vector{String},
                              method::Symbol,
                              export_plots_dir::Union{Nothing, AbstractString})
    isnothing(export_plots_dir) && return
    target_dir = joinpath(export_plots_dir, _method_subdir(method))
    mkpath(target_dir)
    for src in paths
        dst = joinpath(target_dir, basename(src))
        cp(src, dst; force = true)
        println("Copied: ", dst)
    end
end

function _legend_nbanks(labels::Vector{String};
                        figure_size::Tuple{Int, Int},
                        figure_padding::NTuple{4, Int},
                        labelsize::Real,
                        patchsize::Tuple{Int, Int},
                        colgap::Real)
    n = length(labels)
    n == 0 && return 1

    fig_width = figure_size[1]
    left_pad, right_pad = figure_padding[1], figure_padding[2]
    usable_width = max(1.0, fig_width - left_pad - right_pad)

    # Conservative text-width model in points for CM-like fonts.
    avg_char_width = 0.58 * Float64(labelsize)
    text_widths = [length(lbl) * avg_char_width for lbl in labels]
    widths_desc = sort(text_widths; rev = true)
    patch_w = Float64(patchsize[1])
    gap_w = Float64(colgap)
    max_items_per_row = 4

    for banks in 1:n
        items_per_row = cld(n, banks)
        items_per_row <= max_items_per_row || continue

        worst_n = min(items_per_row, n)
        worst_row_width = sum(@view widths_desc[1:worst_n]) +
                          worst_n * patch_w +
                          (worst_n - 1) * gap_w
        if worst_row_width <= usable_width
            return banks
        end
    end

    # Fallback: enforce at most `max_items_per_row` entries per row.
    return max(1, cld(n, max_items_per_row))
end

function _save_overlay_plot(series::AbstractVector,
                            path::AbstractString;
                            title,
                            width_mode::Symbol = :twocolumn,
                            aspect_ratio::Float64 = 0.70,
                            markersize::Real = 8,
                            axis_labelsize::Real = 13,
                            tick_labelsize::Real = 12,
                            title_size::Real = 16,
                            legend_labelsize::Real = 10)
    isempty(series) && throw(ArgumentError("Cannot plot empty spectrum series."))

    mkpath(dirname(path))
    fig_padding = (8, 8, 6, 6)

    _with_custom_theme(axis_labelsize = axis_labelsize,
                       tick_labelsize = tick_labelsize,
                       title_size = title_size,
                       legend_labelsize = legend_labelsize,
                       figure_padding = fig_padding,
                       backgroundcolor = _REF_BG,
                       gridcolor = _REF_GRID) do
        fig_size = _revtex_aps_figure_size(width_mode, aspect_ratio)
        fig = Figure(size = fig_size)
        ax = Axis(fig[1, 1],
                  xlabel = L"\mathrm{Re}(\lambda)",
                  ylabel = L"\mathrm{Im}(\lambda)",
                  title = title)

        seen_labels = Set{String}()
        for (idx, row) in enumerate(series)
            style = _series_style(idx)
            legend_label = if row.legend_label in seen_labels
                nothing
            else
                push!(seen_labels, row.legend_label)
                _legend_entry(row.legend_label)
            end

            if isnothing(legend_label)
                scatter!(ax, real.(row.eigvals), imag.(row.eigvals);
                         markersize = markersize,
                         color = style.color,
                         marker = style.marker)
            else
                scatter!(ax, real.(row.eigvals), imag.(row.eigvals);
                         markersize = markersize,
                         color = style.color,
                         marker = style.marker,
                         label = legend_label)
            end
        end

        vlines!(ax, [0.0]; linestyle = :dash, color = _REF_ZERO, linewidth = 1.6)
        hlines!(ax, [0.0]; linestyle = :dot, color = (:black, 0.55), linewidth = 1.5)

        xlo, xhi, ylo, yhi = _axis_limits(series)
        xlims!(ax, xlo, xhi)
        ylims!(ax, ylo, yhi)

        labels_sorted = sort!(collect(seen_labels))
        patch_size = (30, 16)
        col_gap = 12
        legend_nbanks = _legend_nbanks(labels_sorted;
                                       figure_size = fig_size,
                                       figure_padding = fig_padding,
                                       labelsize = legend_labelsize,
                                       patchsize = patch_size,
                                       colgap = col_gap)

        Legend(fig[2, 1],
               ax;
               orientation = :horizontal,
               nbanks = legend_nbanks,
               tellheight = true,
               tellwidth = false)
        rowsize!(fig.layout, 2, Auto(0.12))

        # Export in physical points for direct compatibility with REVTeX widths.
        CairoMakie.save(path, fig; pt_per_unit = 1.0)
    end
    return path
end

function _maybe_save_overlay_plot(series::AbstractVector,
                                  path::AbstractString;
                                  title,
                                  width_mode::Symbol = :twocolumn,
                                  axis_labelsize::Real = 13,
                                  tick_labelsize::Real = 12,
                                  title_size::Real = 16,
                                  legend_labelsize::Real = 10)
    if isempty(series)
        @warn "Skipping empty spectrum series." path = path title = title
        return nothing
    end

    return _save_overlay_plot(series,
                              path;
                              title = title,
                              width_mode = width_mode,
                              axis_labelsize = axis_labelsize,
                              tick_labelsize = tick_labelsize,
                              title_size = title_size,
                              legend_labelsize = legend_labelsize)
end

function _collect_method_spectra(order::Int;
                                 method::Symbol,
                                 points::Int,
                                 R::Rational{BigInt},
                                 p::Int,
                                 mode,
                                 tiny_zero_tol::Float64,
                                 boundary_model::Symbol = :sat)
    builder = if method === :banded
        _build_banded_ops
    elseif method === :diagonal
        _build_diagonal_ops
    else
        throw(ArgumentError("Unsupported method `$method`; use `:banded` or `:diagonal`."))
    end

    gathered = collect_sbp_sources()
    hyper_reflective = NamedTuple[]
    hyper_radiative = NamedTuple[]
    laplacian = NamedTuple[]
    failures = NamedTuple[]

    for source in gathered.sources
        source_label = _source_label(source)
        short_name = _plot_short_name(source_label)
        try
            ops = builder(source, order; points = points, R = R, p = p, mode = mode)

            L_ref = _assemble_hyperbolic_block(ops; bc = :reflecting,
                                               boundary_model = boundary_model)
            L_rad = _assemble_hyperbolic_block(ops; bc = :absorbing,
                                               boundary_model = boundary_model)
            L_lap = Matrix(ops.D * ops.Geven)

            λ_ref = _to_complex64(_high_precision_schur_values(L_ref);
                                  tiny_zero_tol = tiny_zero_tol)
            λ_rad = _to_complex64(_high_precision_schur_values(L_rad);
                                  tiny_zero_tol = tiny_zero_tol)
            λ_lap = _to_complex64(_high_precision_schur_values(L_lap);
                                  tiny_zero_tol = tiny_zero_tol)

            push!(hyper_reflective,
                  (;
                   source_label = source_label,
                   short_name = short_name,
                   legend_label = short_name,
                   eigvals = λ_ref))
            push!(hyper_radiative,
                  (;
                   source_label = source_label,
                   short_name = short_name,
                   legend_label = short_name,
                   eigvals = λ_rad))
            push!(laplacian,
                  (;
                   source_label = source_label,
                   short_name = short_name,
                   legend_label = short_name,
                   eigvals = λ_lap))
        catch err
            push!(failures, (; label = source_label, error = sprint(showerror, err)))
        end
    end

    _sort_series!(hyper_reflective)
    _sort_series!(hyper_radiative)
    _sort_series!(laplacian)

    return (hyper_reflective = hyper_reflective,
            hyper_radiative = hyper_radiative,
            laplacian = laplacian,
            failures = failures,
            total_sources = length(gathered.sources))
end

function _hyperbolic_plot_title(kind::Symbol, boundary_model::Symbol)
    kind_label = kind === :reflective ? "reflective" :
                 kind === :radiative ? "radiative" :
                 throw(ArgumentError("Unsupported hyperbolic spectrum kind `$kind`."))
    model_label = _boundary_model_tag(boundary_model)
    return "Hyperbolic block spectrum ($kind_label, $model_label BC)"
end

function generate_method_spectrum_plots(;
                                        method::Symbol = :banded,
                                        orders::Tuple{Vararg{Int}} = (4, 6),
                                        points::Int = 31,
                                        p::Int = 2,
                                        R::Rational{BigInt} = big(30) // big(1),
                                        tiny_zero_tol::Float64 = 1e-16,
                                        boundary_model::Symbol = :sat,
                                        mode = SafeMode(),
                                        width_mode::Symbol = :twocolumn,
                                        out_dir::AbstractString = _default_out_dir(method),
                                        export_plots_dir::Union{Nothing, AbstractString} = _DEFAULT_PAPER_PLOTS_DIR,
                                        axis_labelsize::Real = 13,
                                        tick_labelsize::Real = 12,
                                        title_size::Real = 16,
                                        legend_labelsize::Real = 10)
    mkpath(out_dir)
    method_tag = _method_tag(method)
    boundary_model = _normalize_spectrum_boundary_model(boundary_model)
    boundary_tag = _boundary_model_tag(boundary_model)
    rtoken = _R_token(R)

    outputs = Dict{Int, Any}()
    for order in orders
        data = _collect_method_spectra(order;
                                       method = method,
                                       points = points,
                                       R = R,
                                       p = p,
                                       mode = mode,
                                       tiny_zero_tol = tiny_zero_tol,
                                       boundary_model = boundary_model)

        out_ref = joinpath(out_dir,
                           "hyperbolic_block_reflective_$(boundary_tag)_$(method_tag)_order$(order)_N$(points)_R$(rtoken).pdf")
        out_rad = joinpath(out_dir,
                           "hyperbolic_block_radiative_$(boundary_tag)_$(method_tag)_order$(order)_N$(points)_R$(rtoken).pdf")
        out_lap = joinpath(out_dir,
                           "laplacian_divg_$(boundary_tag)_$(method_tag)_order$(order)_N$(points)_R$(rtoken).pdf")

        ref_path = _maybe_save_overlay_plot(data.hyper_reflective,
                                            out_ref;
                                            width_mode = width_mode,
                                            axis_labelsize = axis_labelsize,
                                            tick_labelsize = tick_labelsize,
                                            title_size = title_size,
                                            legend_labelsize = legend_labelsize,
                                            title = _hyperbolic_plot_title(:reflective,
                                                                           boundary_model))
        rad_path = _maybe_save_overlay_plot(data.hyper_radiative,
                                            out_rad;
                                            width_mode = width_mode,
                                            axis_labelsize = axis_labelsize,
                                            tick_labelsize = tick_labelsize,
                                            title_size = title_size,
                                            legend_labelsize = legend_labelsize,
                                            title = _hyperbolic_plot_title(:radiative,
                                                                           boundary_model))
        lap_path = _maybe_save_overlay_plot(data.laplacian,
                                            out_lap;
                                            width_mode = width_mode,
                                            axis_labelsize = axis_labelsize,
                                            tick_labelsize = tick_labelsize,
                                            title_size = title_size,
                                            legend_labelsize = legend_labelsize,
                                            title = "Laplacian spectrum")
        paths_to_export = String[]
        !isnothing(ref_path) && push!(paths_to_export, ref_path)
        !isnothing(rad_path) && push!(paths_to_export, rad_path)
        !isnothing(lap_path) && push!(paths_to_export, lap_path)
        _export_plots_to_dir(paths_to_export, method, export_plots_dir)

        outputs[order] = (reflective_plot = ref_path,
                          radiative_plot = rad_path,
                          laplacian_plot = lap_path,
                          success_count = length(data.laplacian),
                          total_sources = data.total_sources,
                          failures = data.failures)

        @printf("order=%d boundary_model=%s plot summary: %d/%d sources succeeded, %d failed\n",
                order, boundary_tag, length(data.laplacian), data.total_sources,
                length(data.failures))
        println("  reflective: ", out_ref)
        println("  radiative : ", out_rad)
        println("  laplacian : ", out_lap)
    end

    return outputs
end

function generate_banded_spectrum_plots(;
                                        points::Int = 31,
                                        p::Int = 2,
                                        R::Rational{BigInt} = big(30) // big(1),
                                        tiny_zero_tol::Float64 = 1e-16,
                                        boundary_model::Symbol = :sat,
                                        mode = SafeMode(),
                                        width_mode::Symbol = :twocolumn,
                                        out_dir::AbstractString = _default_out_dir(:banded),
                                        export_plots_dir::Union{Nothing, AbstractString} = _DEFAULT_PAPER_PLOTS_DIR)
    return generate_method_spectrum_plots(;
                                          method = :banded,
                                          orders = (4, 6),
                                          points = points,
                                          p = p,
                                          R = R,
                                          tiny_zero_tol = tiny_zero_tol,
                                          boundary_model = boundary_model,
                                          mode = mode,
                                          width_mode = width_mode,
                                          out_dir = out_dir,
                                          export_plots_dir = export_plots_dir)
end

function generate_diagonal_spectrum_plots(;
                                          points::Int = 31,
                                          p::Int = 2,
                                          R::Rational{BigInt} = big(30) // big(1),
                                          tiny_zero_tol::Float64 = 1e-16,
                                          boundary_model::Symbol = :sat,
                                          mode = SafeMode(),
                                          width_mode::Symbol = :twocolumn,
                                          out_dir::AbstractString = _default_out_dir(:diagonal),
                                          export_plots_dir::Union{Nothing, AbstractString} = _DEFAULT_PAPER_PLOTS_DIR)
    return generate_method_spectrum_plots(;
                                          method = :diagonal,
                                          orders = (4, 6, 8),
                                          points = points,
                                          p = p,
                                          R = R,
                                          tiny_zero_tol = tiny_zero_tol,
                                          boundary_model = boundary_model,
                                          mode = mode,
                                          width_mode = width_mode,
                                          out_dir = out_dir,
                                          export_plots_dir = export_plots_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_banded_spectrum_plots()
end
