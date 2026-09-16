"""
    laplacian_matrix(ops_or_mats; as_dense=false)
    laplacian_matrix(D, G; as_dense=false)

Build the discrete scalar Laplacian matrix `L = D * G`, where `G` is either
`Geven` (diagonal-mass operators) or `G` (non-diagonal-mass operators).
"""
function laplacian_matrix(D::AbstractMatrix, G::AbstractMatrix; as_dense::Bool = false)
    nD1, nD2 = size(D)
    nG1, nG2 = size(G)
    nD1 == nD2 || throw(DimensionMismatch("`D` must be square."))
    nG1 == nG2 || throw(DimensionMismatch("`G` must be square."))
    nD2 == nG1 ||
        throw(DimensionMismatch("`D` and `G` dimensions are incompatible for `D*G`."))

    L = D * G
    return as_dense ? Matrix(L) : sparse(L)
end

@inline _hasprop(x, s::Symbol) = hasproperty(x, s)
@inline _default_spectrum_xscale() = identity

function _spectrum_limits_with_padding(eigvals::AbstractVector{<:Complex};
                                       pad_frac::Float64 = 0.08)
    x = real.(eigvals)
    y = imag.(eigvals)
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)

    dx = max(xmax - xmin, max(abs(xmin), abs(xmax), 1.0))
    dy = max(ymax - ymin, max(abs(ymin), abs(ymax), 1.0))
    xpad = pad_frac * dx
    ypad = pad_frac * dy

    return (xmin - xpad,
            xmax + xpad,
            ymin - ypad,
            ymax + ypad)
end

function _negative_x_limits(eigvals::AbstractVector{<:Complex};
                            left_pad_frac::Float64 = 0.1,
                            right_pad_frac::Float64 = 0.03)
    x = real.(eigvals)
    xmin, xmax = extrema(x)

    span = max(xmax - xmin, max(abs(xmin), abs(xmax), 1.0))
    xlo = xmin - left_pad_frac * span
    xhi = min(0.0, xmax + right_pad_frac * span)

    if xlo >= xhi
        xhi = min(0.0, xmax)
        xlo = xhi - (1 + left_pad_frac) * span
    end

    return xlo, xhi
end

function _extract_laplacian_factors(ops)
    _hasprop(ops, :D) ||
        throw(ArgumentError("Input must provide a divergence matrix field `D`."))

    D = getproperty(ops, :D)
    if _hasprop(ops, :Geven)
        return D, getproperty(ops, :Geven)
    elseif _hasprop(ops, :G)
        return D, getproperty(ops, :G)
    end

    throw(ArgumentError("Input must provide either `Geven` (diagonal-mass) or `G` (non-diagonal-mass)."))
end

function laplacian_matrix(ops; as_dense::Bool = false)
    D, G = _extract_laplacian_factors(ops)
    return laplacian_matrix(D, G; as_dense = as_dense)
end

"""
    laplacian_spectrum(ops_or_mats)
    laplacian_spectrum(D, G)

Compute eigenvalues of the discrete Laplacian `L = D*G` with the package-wide
high-precision spectrum path:

1. convert matrix entries to `Rational{BigInt}`,
2. convert to `Float64x4`,
3. compute Schur values via `GenericSchur`.
"""
function laplacian_spectrum(D::AbstractMatrix, G::AbstractMatrix)
    L = laplacian_matrix(D, G; as_dense = true)
    return _high_precision_schur_values(L)
end

function laplacian_spectrum(ops)
    D, G = _extract_laplacian_factors(ops)
    return laplacian_spectrum(D, G)
end

"""
    save_laplacian_spectrum_plot(ops_or_mats, path; kwargs...)
    save_laplacian_spectrum_plot(D, G, path; kwargs...)

Plot `eig(D*G)` in the complex plane and save to `path`.
"""
function save_laplacian_spectrum_plot(D::AbstractMatrix,
                                      G::AbstractMatrix,
                                      path::AbstractString;
                                      title = L"\mathrm{Laplacian\ Spectrum}\ (D\,G)",
                                      markersize::Real = 8,
                                      color = (:dodgerblue, 0.7),
                                      xscale = _default_spectrum_xscale(),
                                      axis_labelsize::Real = 28,
                                      tick_labelsize::Real = 18,
                                      title_size::Real = 30,
                                      draw_axes::Bool = true)
    return _with_publication_theme() do
        eigvals = laplacian_spectrum(D, G)
        xlo, xhi, ylo, yhi = _spectrum_limits_with_padding(eigvals)

        fig = Figure(size = (760, 560))
        ax = Axis(fig[1, 1],
                  xlabel = L"\mathrm{Re}(\lambda)",
                  ylabel = L"\mathrm{Im}(\lambda)",
                  xscale = xscale,
                  xlabelsize = axis_labelsize,
                  ylabelsize = axis_labelsize,
                  xticklabelsize = tick_labelsize,
                  yticklabelsize = tick_labelsize,
                  titlesize = title_size,
                  title = title)

        scatter!(ax, real.(eigvals), imag.(eigvals); markersize = markersize, color = color)

        if draw_axes
            vlines!(ax, [0.0]; linestyle = :dash, color = :black, linewidth = 2)
            hlines!(ax, [0.0]; linestyle = :dot, color = (:black, 0.55), linewidth = 1.5)
        end

        xlims!(ax, xlo, xhi)
        ylims!(ax, ylo, yhi)
        save(path, fig)
        return path
    end
end

function save_laplacian_spectrum_plot(ops,
                                      path::AbstractString;
                                      title = L"\mathrm{Laplacian\ Spectrum}\ (D\,G)",
                                      markersize::Real = 8,
                                      color = (:dodgerblue, 0.7),
                                      xscale = _default_spectrum_xscale(),
                                      axis_labelsize::Real = 28,
                                      tick_labelsize::Real = 18,
                                      title_size::Real = 30,
                                      draw_axes::Bool = true)
    D, G = _extract_laplacian_factors(ops)
    return save_laplacian_spectrum_plot(D, G, path;
                                        title = title,
                                        markersize = markersize,
                                        color = color,
                                        xscale = xscale,
                                        axis_labelsize = axis_labelsize,
                                        tick_labelsize = tick_labelsize,
                                        title_size = title_size,
                                        draw_axes = draw_axes)
end

@inline _source_label(source) = string(nameof(typeof(source)))

"""
    save_laplacian_spectrum_sources_plot(sources, path; kwargs...)

Build diagonal-mass spherical operators for each SBP `source` via
`diagonal_spherical_operators(...)`, compute `eig(D*Geven)`, and save an overlaid spectrum
plot.

This routine intentionally targets the diagonal-mass construction path.
"""
function save_laplacian_spectrum_sources_plot(sources::AbstractVector,
                                              path::AbstractString;
                                              labels::Union{Nothing,
                                                            AbstractVector{<:AbstractString}} = nothing,
                                              accuracy_order::Int = 6,
                                              N::Int = 64,
                                              R::Real = 1.0,
                                              p::Int = 2,
                                              mode = SafeMode(),
                                              markersize::Real = 8,
                                              xscale = _default_spectrum_xscale(),
                                              axis_labelsize::Real = 30,
                                              tick_labelsize::Real = 18,
                                              title_size::Real = 34,
                                              legend_labelsize::Real = 16,
                                              legend_position::Symbol = :lt,
                                              negative_x_only::Bool = true,
                                              x_left_pad_frac::Float64 = 0.1,
                                              x_right_pad_frac::Float64 = 0.03,
                                              draw_axes::Bool = true,
                                              title = L"\mathrm{Laplacian\ Spectrum\ Across\ SBP\ Sources}",
                                              skip_failed::Bool = false)
    ns = length(sources)
    ns > 0 || throw(ArgumentError("`sources` must be non-empty."))
    if labels !== nothing && length(labels) != ns
        throw(DimensionMismatch("`labels` length ($(length(labels))) must match `sources` length ($ns)."))
    end

    palette = (:dodgerblue,
               :darkorange,
               :seagreen,
               :crimson,
               :purple,
               :teal,
               :goldenrod,
               :firebrick,
               :slateblue,
               :olivedrab)

    return _with_publication_theme() do
        fig = Figure(size = (1200, 820), figure_padding = (24, 48, 28, 20))
        ax = Axis(fig[1, 1],
                  xlabel = L"\mathrm{Re}(\lambda)",
                  ylabel = L"\mathrm{Im}(\lambda)",
                  xscale = xscale,
                  xlabelsize = axis_labelsize,
                  ylabelsize = axis_labelsize,
                  xticklabelsize = tick_labelsize,
                  yticklabelsize = tick_labelsize,
                  titlesize = title_size,
                  title = title)

        spectra = NamedTuple[]
        failures = NamedTuple[]
        eigvals_all = ComplexF64[]

        for (k, source) in enumerate(sources)
            label = labels === nothing ? _source_label(source) : labels[k]
            color = (palette[mod1(k, length(palette))], 0.75)

            try
                ops = diagonal_spherical_operators(source;
                                          accuracy_order = accuracy_order,
                                          N = N,
                                          R = R,
                                          p = p,
                                          mode = mode)
                eigvals = laplacian_spectrum(ops)
                scatter!(ax, real.(eigvals), imag.(eigvals);
                         markersize = markersize,
                         color = color,
                         label = label)
                push!(spectra, (; source, label, eigvals))
                append!(eigvals_all, ComplexF64.(eigvals))
            catch err
                if !skip_failed
                    rethrow(err)
                end
                push!(failures, (; source, label, error = sprint(showerror, err)))
                @warn("Skipping source while plotting Laplacian spectrum.", source=label,
                      error=sprint(showerror, err))
            end
        end

        isempty(spectra) &&
            throw(ArgumentError("No Laplacian spectra were plotted; all sources failed."))

        if draw_axes
            vlines!(ax, [0.0]; linestyle = :dash, color = :black, linewidth = 2)
            hlines!(ax, [0.0]; linestyle = :dot, color = (:black, 0.55), linewidth = 1.5)
        end
        _, _, ylo, yhi = _spectrum_limits_with_padding(eigvals_all)
        if negative_x_only
            xlo, xhi = _negative_x_limits(eigvals_all;
                                          left_pad_frac = x_left_pad_frac,
                                          right_pad_frac = x_right_pad_frac)
        else
            xlo, xhi, _, _ = _spectrum_limits_with_padding(eigvals_all)
        end
        xlims!(ax, xlo, xhi)
        ylims!(ax, ylo, yhi)
        axislegend(ax;
                   position = legend_position,
                   labelsize = legend_labelsize,
                   framevisible = true,
                   rowgap = 2,
                   margin = (6, 6, 6, 6),
                   padding = (6, 6, 6, 6))

        save(path, fig)
        return (path = path, spectra = spectra, failures = failures)
    end
end
