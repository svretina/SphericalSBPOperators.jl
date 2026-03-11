function mytheme_aps()
    return Theme(
                 ;
                 Axis = Attributes(;
                                   spinewidth = 1.1,
                                   xgridvisible = true,
                                   xlabelpadding = -0,
                                   xlabelsize = 12,
                                   xminortickalign = 1,
                                   xminorticks = IntervalsBetween(5, true),
                                   xminorticksize = 3,
                                   xminorticksvisible = true,
                                   xminortickwidth = 0.75,
                                   xtickalign = 1,
                                   xticklabelsize = 8,
                                   xticksize = 5,
                                   xticksmirrored = true,
                                   xtickwidth = 0.8,
                                   ygridvisible = true,
                                   ylabelpadding = 2,
                                   ylabelsize = 12,
                                   yminortickalign = 1,
                                   yminorticks = IntervalsBetween(5, true),
                                   yminorticksize = 3,
                                   yminorticksvisible = true,
                                   yminortickwidth = 0.75,
                                   ytickalign = 1,
                                   yticklabelsize = 10,
                                   yticksize = 5,
                                   yticksmirrored = true,
                                   ytickwidth = 0.8,
                                   xticklabelfont = "cmr10",
                                   yticklabelfont = "cmr10",
                                   xticklabelstyle = Attributes(; italic = false),
                                   yticklabelstyle = Attributes(; italic = false)
                                  ),
                 colgap = 8,
                 figure_padding = 0,
                 rowgap = 8,
                 size = (243, 165),
                 Colorbar = Attributes(;
                                       labelpadding = 2,
                                       labelsize = 10,
                                       minortickalign = 1,
                                       minorticksize = 3,
                                       minorticksvisible = true,
                                       minortickwidth = 0.75,
                                       size = 8,
                                       spinewidth = 1.1,
                                       tickalign = 1,
                                       ticklabelpad = 2,
                                       ticklabelsize = 8,
                                       ticksize = 5,
                                       tickwidth = 0.8
                                      ),
                 fonts = Attributes(;
                                    bold = "NewComputerModern10 Bold",
                                    bold_italic = "NewComputerModern10 Bold Italic",
                                    italic = "NewComputerModern10 Italic",
                                    regular = "NewComputerModern Math Regular"
                                   ),
                 Legend = Attributes(;
                                    colgap = 4,
                                    framecolor = (:grey, 0.5),
                                    framevisible = false,
                                    labelsize = 7.5,
                                    margin = (0, 0, 0, 0),
                                    nbanks = 1,
                                    padding = (2, 2, 2, 2),
                                    rowgap = -10
                                   ),
                 Lines = Attributes(;
                                   cycle = Cycle([[:color] => :color], true)
                                  ),
                 Scatter = Attributes(;
                                     cycle = Cycle([[:color] => :color, [:marker] => :marker], true),
                                     markersize = 7,
                                     strokewidth = 0
                                    ),
                 markersize = 7,
                 palette = Attributes(;
                                      color = [RGBAf(0.298039, 0.447059, 0.690196, 1.0),
                                               RGBAf(0.866667, 0.517647, 0.321569, 1.0),
                                               RGBAf(0.333333, 0.658824, 0.407843, 1.0),
                                               RGBAf(0.768627, 0.305882, 0.321569, 1.0),
                                               RGBAf(0.505882, 0.447059, 0.701961, 1.0),
                                               RGBAf(0.576471, 0.470588, 0.376471, 1.0),
                                               RGBAf(0.854902, 0.545098, 0.764706, 1.0),
                                               RGBAf(0.54902, 0.54902, 0.54902, 1.0),
                                               RGBAf(0.8, 0.72549, 0.454902, 1.0),
                                               RGBAf(0.392157, 0.709804, 0.803922, 1.0)],
                                      linestyle = [nothing, :dash, :dot, :dashdot, :dashdotdot],
                                      marker = [:circle, :rect, :dtriangle, :utriangle, :cross,
                                                :diamond, :ltriangle, :rtriangle, :pentagon,
                                                :xcross, :hexagon],
                                      markercolor = [RGBAf(0.298039, 0.447059, 0.690196, 1.0),
                                                     RGBAf(0.866667, 0.517647, 0.321569, 1.0),
                                                     RGBAf(0.333333, 0.658824, 0.407843, 1.0),
                                                     RGBAf(0.768627, 0.305882, 0.321569, 1.0),
                                                     RGBAf(0.505882, 0.447059, 0.701961, 1.0),
                                                     RGBAf(0.576471, 0.470588, 0.376471, 1.0),
                                                     RGBAf(0.854902, 0.545098, 0.764706, 1.0),
                                                     RGBAf(0.54902, 0.54902, 0.54902, 1.0),
                                                     RGBAf(0.8, 0.72549, 0.454902, 1.0),
                                                     RGBAf(0.392157, 0.709804, 0.803922, 1.0)],
                                      patchcolor = [RGBAf(0.298039, 0.447059, 0.690196, 1.0),
                                                    RGBAf(0.866667, 0.517647, 0.321569, 1.0),
                                                    RGBAf(0.333333, 0.658824, 0.407843, 1.0),
                                                    RGBAf(0.768627, 0.305882, 0.321569, 1.0),
                                                    RGBAf(0.505882, 0.447059, 0.701961, 1.0),
                                                    RGBAf(0.576471, 0.470588, 0.376471, 1.0),
                                                    RGBAf(0.854902, 0.545098, 0.764706, 1.0),
                                                    RGBAf(0.54902, 0.54902, 0.54902, 1.0),
                                                    RGBAf(0.8, 0.72549, 0.454902, 1.0),
                                                    RGBAf(0.392157, 0.709804, 0.803922, 1.0)]
                                     ),
                 PolarAxis = Attributes(; spinewidth = 1.1)
                )
end

@inline function _with_aps_theme(f::Function)
    return with_theme(mytheme_aps()) do
        f()
    end
end

function save_spectrum_plot(eigvals::AbstractVector{<:Complex}, path::AbstractString)
    return _with_aps_theme() do
        fig = Figure(size = (760, 560))
        ax = Axis(fig[1, 1],
                  xlabel = L"\mathrm{Re}(\lambda)",
                  ylabel = L"\mathrm{Im}(\lambda)",
                  title = L"\mathrm{Eigenvalue\ Spectrum}")
        scatter!(ax, real.(eigvals), imag.(eigvals); markersize = 6, color = (:dodgerblue, 0.7))
        vlines!(ax, [0.0]; linestyle = :dash, color = :black, linewidth = 2)
        save(path, fig)
        return path
    end
end

function save_nullspace_plot(null_report, path::AbstractString; max_modes::Int = 6)
    return _with_aps_theme() do
        if isempty(null_report.entries)
            fig = Figure(size = (760, 240))
            ax = Axis(fig[1, 1],
                      title = L"\mathrm{Null\ Space\ Modes}",
                      xlabel = L"i",
                      ylabel = L"\mathrm{value}")
            text!(ax, 0.5, 0.5, text = L"\mathrm{No\ near\text{-}zero\ modes\ detected}", space = :relative, align = (:center, :center))
            save(path, fig)
            return path
        end

        nm = min(max_modes, length(null_report.entries))
        fig = Figure(size = (980, 220 * nm))

        for k in 1:nm
            entry = null_report.entries[k]
            mode_title = latexstring("\\mathrm{mode}\\ ", entry.eigen_index,
                                     "\\;|\\;\\mathrm{spurious}=",
                                     entry.is_spurious ? "1" : "0")
            ax = Axis(fig[k, 1],
                      xlabel = L"i_r",
                      ylabel = L"\mathrm{amplitude}",
                      title = mode_title)
            lines!(ax, entry.pi_mode; color = :royalblue, linewidth = 2, label = L"\Pi")
            lines!(ax, entry.psi_mode; color = :darkorange, linewidth = 2, linestyle = :dash, label = L"\Psi")
            axislegend(ax; position = :rt)
        end

        save(path, fig)
        return path
    end
end

function save_reflection_heatmap(reflection_report, path::AbstractString)
    return _with_aps_theme() do
        fig = Figure(size = (900, 560))
        ax = Axis(fig[1, 1], xlabel = L"t", ylabel = L"i_r", title = L"\mathrm{Wave\ Packet\ Reflection}")
        hm = heatmap!(ax, reflection_report.t, 1:size(reflection_report.pi, 1), reflection_report.pi')
        Colorbar(fig[1, 2], hm, label = L"\Pi(r,t)")
        save(path, fig)
        return path
    end
end

function save_energy_trace(reflection_report, path::AbstractString)
    return _with_aps_theme() do
        fig = Figure(size = (760, 420))
        ax = Axis(fig[1, 1], xlabel = L"t", ylabel = L"E", title = L"\mathrm{Discrete\ Energy}")
        lines!(ax, reflection_report.t, reflection_report.energy; linewidth = 2, color = :black)
        save(path, fig)
        return path
    end
end

function save_dashboard(report, path::AbstractString)
    return _with_aps_theme() do
        fig = Figure(size = (1600, 1000))

        # Spectrum
        ax1 = Axis(fig[1, 1],
                   xlabel = L"\mathrm{Re}(\lambda)",
                   ylabel = L"\mathrm{Im}(\lambda)",
                   title = L"\mathrm{Spectrum}")
        scatter!(ax1, real.(report.spectral.eigvals), imag.(report.spectral.eigvals); markersize = 5, color = (:dodgerblue, 0.7))
        vlines!(ax1, [0.0]; linestyle = :dash, color = :black, linewidth = 2)

        # Null-space representative mode (first, if present)
        ax2 = Axis(fig[1, 2], xlabel = L"i_r", ylabel = L"\mathrm{mode\ amplitude}", title = L"\mathrm{Null\ Space\ Modes}")
        if isempty(report.nullspace.entries)
            text!(ax2, 0.5, 0.5, text = L"\mathrm{No\ near\text{-}zero\ modes}", space = :relative, align = (:center, :center))
        else
            mode = report.nullspace.entries[1]
            lines!(ax2, mode.pi_mode; color = :royalblue, linewidth = 2, label = L"\Pi")
            lines!(ax2, mode.psi_mode; color = :darkorange, linewidth = 2, linestyle = :dash, label = L"\Psi")
            axislegend(ax2; position = :rt)
        end

        # Reflection heatmap
        ax3 = Axis(fig[2, 1], xlabel = L"t", ylabel = L"i_r", title = L"\mathrm{Reflection\ Waterfall}")
        hm = heatmap!(ax3, report.reflection.t, 1:size(report.reflection.pi, 1), report.reflection.pi')
        Colorbar(fig[2, 2], hm, label = L"\Pi(r,t)")

        # Energy trace in inset axis
        ax4 = Axis(fig[2, 2], xlabel = L"t", ylabel = L"E", title = L"\mathrm{Energy}")
        lines!(ax4, report.reflection.t, report.reflection.energy; color = :black, linewidth = 2)

        save(path, fig)
        return path
    end
end

function save_reflection_animation(reflection_report, r, path::AbstractString)
    return _with_aps_theme() do
        rf = Float64.(r)
        fig = Figure(size = (900, 500))
        ax = Axis(fig[1, 1], xlabel = L"r", ylabel = L"\Pi(r,t)", title = L"\mathrm{Reflection\ Animation}")

        y = Observable(copy(reflection_report.pi[:, 1]))
        lines!(ax, rf, y; color = :royalblue, linewidth = 2)

        record(fig, path, 1:length(reflection_report.t); framerate = 20) do k
            y[] = reflection_report.pi[:, k]
            ax.title = latexstring("\\Pi(r,t),\\ t=", string(round(Float64(reflection_report.t[k]); digits = 4)))
        end

        return path
    end
end
