using CairoMakie
using Printf: @printf, @sprintf

# Reuse the exact same banded-operator construction and high-precision
# spectrum helpers used for the table generation pipeline.
include(joinpath(@__DIR__, "generate_banded_spectrum_tables.jl"))

const _APS_REVTeX_COLUMNWIDTH_PT = 246.0
const _APS_REVTeX_TEXTWIDTH_PT = 510.0
const _REF_BG = RGBf(1.0, 1.0, 1.0)
const _REF_GRID = RGBAf(0.0, 0.0, 0.0, 0.10)
const _REF_ZERO = RGBAf(0.0, 0.0, 0.0, 0.35)
const _DEFAULT_PAPER_PLOTS_DIR = "/home/svretina/PhD/mypapers/Spherical-SBP-Operators-Paper/plots"

@inline function _with_custom_theme(f::Function)
    return with_theme(SphericalSBPOperators.mytheme_aps()) do
        f()
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

@inline function _to_complex64(vals::AbstractVector{<:Complex}; tiny_zero_tol::Float64=1e-16)
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

function _axis_limits(series::AbstractVector; pad_frac::Float64=0.10)
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

@inline _plot_cite_key(label::String) = get(_SOURCE_TO_CITE, label, label)

function _sort_series!(series::Vector)
    sort!(series; by = row -> (get(_CITE_SORT_INDEX, row.cite_key, 10_000), row.source_label))
    return series
end

function _build_diagonal_ops(source, order::Int;
                             points::Int,
                             R::Rational{BigInt},
                             p::Int,
                             mode)
    return spherical_operators(
        source;
        accuracy_order = order,
        N = points,
        R = R,
        p = p,
        mode = mode
    )
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
                            title::AbstractString,
                            width_mode::Symbol=:twocolumn,
                            aspect_ratio::Float64=0.70,
                            markersize::Real=8,
                            axis_labelsize::Real=13,
                            tick_labelsize::Real=12,
                            title_size::Real=16,
                            legend_labelsize::Real=10)
    isempty(series) && throw(ArgumentError("Cannot plot empty spectrum series."))

    mkpath(dirname(path))

    _with_custom_theme() do
        fig_size = _revtex_aps_figure_size(width_mode, aspect_ratio)
        fig_padding = (8, 8, 6, 6)
        fig = Figure(
            size = fig_size,
            figure_padding = fig_padding,
            backgroundcolor = _REF_BG
        )
        ax = Axis(fig[1, 1],
                  xlabel = "Re(λ)",
                  ylabel = "Im(λ)",
                  title = title,
                  backgroundcolor = _REF_BG,
                  xlabelsize = axis_labelsize,
                  ylabelsize = axis_labelsize,
                  xticklabelsize = tick_labelsize,
                  yticklabelsize = tick_labelsize,
                  titlesize = title_size,
                  spinewidth = 1.8,
                  xgridvisible = true,
                  ygridvisible = true,
                  xgridcolor = _REF_GRID,
                  ygridcolor = _REF_GRID,
                  xgridwidth = 1.0,
                  ygridwidth = 1.0,
                  xminorticksvisible = true,
                  yminorticksvisible = true,
                  xminorticks = IntervalsBetween(5),
                  yminorticks = IntervalsBetween(5),
                  xminortickalign = 1.0,
                  yminortickalign = 1.0,
                  xminorticksize = 4,
                  yminorticksize = 4,
                  xminortickwidth = 1.1,
                  yminortickwidth = 1.1,
                  xtickalign = 1.0,
                  ytickalign = 1.0,
                  xticksmirrored = true,
                  yticksmirrored = true,
                  xticksize = 8,
                  yticksize = 8,
                  xtickwidth = 1.3,
                  ytickwidth = 1.3)

        seen_labels = Set{String}()
        for row in series
            legend_label = if row.legend_label in seen_labels
                nothing
            else
                push!(seen_labels, row.legend_label)
                row.legend_label
            end

            if isnothing(legend_label)
                scatter!(ax, real.(row.eigvals), imag.(row.eigvals);
                         markersize = markersize)
            else
                scatter!(ax, real.(row.eigvals), imag.(row.eigvals);
                         markersize = markersize,
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
        legend_nbanks = _legend_nbanks(
            labels_sorted;
            figure_size = fig_size,
            figure_padding = fig_padding,
            labelsize = legend_labelsize,
            patchsize = patch_size,
            colgap = col_gap
        )

        Legend(
            fig[2, 1],
            ax;
            orientation = :horizontal,
            framevisible = false,
            labelsize = legend_labelsize,
            patchsize = patch_size,
            rowgap = 2,
            colgap = col_gap,
            nbanks = legend_nbanks,
            margin = (0, 0, 0, 0),
            padding = (2, 2, 2, 2),
            tellheight = true,
            tellwidth = false
        )
        rowsize!(fig.layout, 2, Auto(0.12))

        # Export in physical points for direct compatibility with REVTeX widths.
        save(path, fig; pt_per_unit = 1.0)
    end
    return path
end

function _collect_method_spectra(order::Int;
                                 method::Symbol,
                                 points::Int,
                                 R::Rational{BigInt},
                                 p::Int,
                                 mode,
                                 tiny_zero_tol::Float64)
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
        cite_key = _plot_cite_key(source_label)
        try
            ops = builder(source, order; points = points, R = R, p = p, mode = mode)

            L_ref = _assemble_hyperbolic_block(ops; bc = :reflecting)
            L_rad = _assemble_hyperbolic_block(ops; bc = :absorbing)
            L_lap = Matrix(ops.D * ops.Geven)

            λ_ref = _to_complex64(_high_precision_schur_values(L_ref); tiny_zero_tol = tiny_zero_tol)
            λ_rad = _to_complex64(_high_precision_schur_values(L_rad); tiny_zero_tol = tiny_zero_tol)
            λ_lap = _to_complex64(_high_precision_schur_values(L_lap); tiny_zero_tol = tiny_zero_tol)

            push!(hyper_reflective, (;
                                     source_label = source_label,
                                     cite_key = cite_key,
                                     legend_label = cite_key,
                                     eigvals = λ_ref))
            push!(hyper_radiative, (;
                                    source_label = source_label,
                                    cite_key = cite_key,
                                    legend_label = cite_key,
                                    eigvals = λ_rad))
            push!(laplacian, (;
                              source_label = source_label,
                              cite_key = cite_key,
                              legend_label = cite_key,
                              eigvals = λ_lap))
        catch err
            push!(failures, (; label = source_label, error = sprint(showerror, err)))
        end
    end

    _sort_series!(hyper_reflective)
    _sort_series!(hyper_radiative)
    _sort_series!(laplacian)

    return (
        hyper_reflective = hyper_reflective,
        hyper_radiative = hyper_radiative,
        laplacian = laplacian,
        failures = failures,
        total_sources = length(gathered.sources),
    )
end

function generate_method_spectrum_plots(;
                                        method::Symbol=:banded,
                                        orders::Tuple{Vararg{Int}}=(4, 6),
                                        points::Int=31,
                                        p::Int=2,
                                        R::Rational{BigInt}=big(30) // big(1),
                                        tiny_zero_tol::Float64=1e-16,
                                        mode=SafeMode(),
                                        width_mode::Symbol=:twocolumn,
                                        out_dir::AbstractString=_default_out_dir(method),
                                        export_plots_dir::Union{Nothing, AbstractString}=_DEFAULT_PAPER_PLOTS_DIR)
    mkpath(out_dir)
    method_tag = _method_tag(method)
    rtoken = _R_token(R)

    outputs = Dict{Int,Any}()
    for order in orders
        data = _collect_method_spectra(order;
                                       method = method,
                                       points = points,
                                       R = R,
                                       p = p,
                                       mode = mode,
                                       tiny_zero_tol = tiny_zero_tol)

        out_ref = joinpath(out_dir, "hyperbolic_block_reflective_$(method_tag)_order$(order)_N$(points)_R$(rtoken).pdf")
        out_rad = joinpath(out_dir, "hyperbolic_block_radiative_$(method_tag)_order$(order)_N$(points)_R$(rtoken).pdf")
        out_lap = joinpath(out_dir, "laplacian_divg_$(method_tag)_order$(order)_N$(points)_R$(rtoken).pdf")

        _save_overlay_plot(
            data.hyper_reflective,
            out_ref;
            width_mode = width_mode,
            title = "Hyperbolic block spectrum (reflective)"
        )
        _save_overlay_plot(
            data.hyper_radiative,
            out_rad;
            width_mode = width_mode,
            title = "Hyperbolic block spectrum (radiative)"
        )
        _save_overlay_plot(
            data.laplacian,
            out_lap;
            width_mode = width_mode,
            title = "Laplacian spectrum"
        )
        _export_plots_to_dir([out_ref, out_rad, out_lap], method, export_plots_dir)

        outputs[order] = (
            reflective_plot = out_ref,
            radiative_plot = out_rad,
            laplacian_plot = out_lap,
            success_count = length(data.laplacian),
            total_sources = data.total_sources,
            failures = data.failures,
        )

        @printf("order=%d plot summary: %d/%d sources succeeded, %d failed\n",
                order, length(data.laplacian), data.total_sources, length(data.failures))
        println("  reflective: ", out_ref)
        println("  radiative : ", out_rad)
        println("  laplacian : ", out_lap)
    end

    return outputs
end

function generate_banded_spectrum_plots(;
                                        points::Int=31,
                                        p::Int=2,
                                        R::Rational{BigInt}=big(30) // big(1),
                                        tiny_zero_tol::Float64=1e-16,
                                        mode=SafeMode(),
                                        width_mode::Symbol=:twocolumn,
                                        out_dir::AbstractString=_default_out_dir(:banded),
                                        export_plots_dir::Union{Nothing, AbstractString}=_DEFAULT_PAPER_PLOTS_DIR)
    return generate_method_spectrum_plots(;
                                          method = :banded,
                                          orders = (4, 6),
                                          points = points,
                                          p = p,
                                          R = R,
                                          tiny_zero_tol = tiny_zero_tol,
                                          mode = mode,
                                          width_mode = width_mode,
                                          out_dir = out_dir,
                                          export_plots_dir = export_plots_dir)
end

function generate_diagonal_spectrum_plots(;
                                          points::Int=31,
                                          p::Int=2,
                                          R::Rational{BigInt}=big(30) // big(1),
                                          tiny_zero_tol::Float64=1e-16,
                                          mode=SafeMode(),
                                          width_mode::Symbol=:twocolumn,
                                          out_dir::AbstractString=_default_out_dir(:diagonal),
                                          export_plots_dir::Union{Nothing, AbstractString}=_DEFAULT_PAPER_PLOTS_DIR)
    return generate_method_spectrum_plots(;
                                          method = :diagonal,
                                          orders = (4, 6, 8),
                                          points = points,
                                          p = p,
                                          R = R,
                                          tiny_zero_tol = tiny_zero_tol,
                                          mode = mode,
                                          width_mode = width_mode,
                                          out_dir = out_dir,
                                          export_plots_dir = export_plots_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_banded_spectrum_plots()
end
