"""
    plot_energy_history(t, energies; title="Energy Conservation")

Plot energy history in publication-friendly form.

- For very small drift, plot `E/E0 - 1` so y-axis reflects true order (e.g. `1e-17`).
- Otherwise plot `E/E0`.
"""
_format_ticks_sci(values) = [@sprintf("%.1e", Float64(v)) for v in values]
_format_ticks_fixed(values) = [@sprintf("%.6f", Float64(v)) for v in values]

function plot_energy_history(
        t::AbstractVector,
        energies::AbstractVector;
        title::AbstractString = L"Energy\ Conservation",
        representation::Symbol = :auto,
        switch_tol::Real = 1.0e-4
    )
    E0 = energies[begin]
    normalized_energy = energies ./ E0
    rel_drift = normalized_energy .- 1.0
    max_rel_drift = maximum(abs.(rel_drift))

    rep = representation
    if rep === :auto
        rep = max_rel_drift <= Float64(switch_tol) ? :drift : :normalized
    end
    rep in (:drift, :normalized) ||
        throw(ArgumentError("`representation` must be :auto, :drift, or :normalized."))

    yvals = rep === :drift ? rel_drift : normalized_energy
    ylabel_txt = rep === :drift ? L"E/E_0 - 1" : L"E/E_0"
    yref = rep === :drift ? 0.0 : 1.0
    yfmt = rep === :drift ?
        _format_ticks_sci :
        _format_ticks_fixed
    subtitle = latexstring("\\max\\,|E/E_0-1| = ", @sprintf("%.2e", max_rel_drift))

    return with_theme(theme_prd()) do
        fig = Figure(size = (980, 460))
        Label(fig[0, 1], subtitle; fontsize = 13, halign = :right)
        ax = Axis(
            fig[1, 1];
            title = title,
            xlabel = L"t",
            ylabel = ylabel_txt,
            ytickformat = yfmt,
            titlesize = 18,
            xlabelsize = 16,
            ylabelsize = 16,
            xticklabelsize = 12,
            yticklabelsize = 12
        )
        rowsize!(fig.layout, 1, Relative(1))
        colsize!(fig.layout, 1, Relative(1))
        lines!(ax, t, yvals; color = :dodgerblue4, linewidth = 2)
        hlines!(ax, [yref]; color = :black, linestyle = :dash, linewidth = 1)
        return fig
    end
end

function _rescaled_fields(Π::AbstractMatrix, Ξ::AbstractMatrix, r::AbstractVector)
    n, nt = size(Π)
    size(Ξ) == (n, nt) || throw(DimensionMismatch("`Π` and `Ξ` must have same shape."))
    length(r) == n || throw(DimensionMismatch("`r` length must match field rows."))

    rΠ = Matrix{Float64}(undef, n, nt)
    rΞ = Matrix{Float64}(undef, n, nt)
    for k in 1:nt
        @views rΠ[:, k] .= r .* Π[:, k]
        @views rΞ[:, k] .= r .* Ξ[:, k]
    end
    return rΠ, rΞ
end

function _cell_edges(v::AbstractVector)
    n = length(v)
    n >= 2 || throw(ArgumentError("Need at least two points to build cell edges."))
    e = Vector{Float64}(undef, n + 1)
    for i in 2:n
        e[i] = 0.5 * (Float64(v[i - 1]) + Float64(v[i]))
    end
    e[1] = Float64(v[1]) - (e[2] - Float64(v[1]))
    e[end] = Float64(v[end]) + (Float64(v[end]) - e[end - 1])
    return e
end

"""
    plot_initial_data(Π, Ξ, r; title="Initial Data (rescaled by r)")

Plot the initial rescaled fields `r\\Pi(r,0)` and `r\\Xi(r,0)`.
"""
function plot_initial_data(
        Π::AbstractMatrix,
        Ξ::AbstractMatrix,
        r::AbstractVector;
        title::AbstractString = L"Initial\ Data"
    )
    size(Π, 2) >= 1 || throw(ArgumentError("`Π` must contain at least one timestep."))
    size(Ξ, 2) >= 1 || throw(ArgumentError("`Ξ` must contain at least one timestep."))
    length(r) == size(Π, 1) || throw(DimensionMismatch("`r` length must match field rows."))
    size(Ξ) == size(Π) || throw(DimensionMismatch("`Π` and `Ξ` must have same shape."))

    rΠ0 = r .* view(Π, :, 1)
    rΞ0 = r .* view(Ξ, :, 1)

    return with_theme(theme_prd()) do
        fig = Figure(size = (920, 380))
        Label(fig[0, 1:2], title; fontsize = 18)

        ax1 = Axis(fig[1, 1]; xlabel = L"r", ylabel = L"r\Pi(r,0)", title = L"r\Pi(r,0)")
        ax2 = Axis(fig[1, 2]; xlabel = L"r", ylabel = L"r\Xi(r,0)", title = L"r\Xi(r,0)")
        lines!(ax1, r, rΠ0; color = :steelblue4, linewidth = 2.6)
        lines!(ax2, r, rΞ0; color = :firebrick3, linewidth = 2.6)
        hlines!(ax1, [0.0]; color = RGBAf(0.0, 0.0, 0.0, 0.75), linewidth = 1.0)
        hlines!(ax2, [0.0]; color = RGBAf(0.0, 0.0, 0.0, 0.75), linewidth = 1.0)
        vlines!(
            ax1, [0.0]; color = RGBAf(0.0, 0.0, 0.0, 0.4),
            linewidth = 1.0, linestyle = :dash
        )
        vlines!(
            ax2, [0.0]; color = RGBAf(0.0, 0.0, 0.0, 0.4),
            linewidth = 1.0, linestyle = :dash
        )
        return fig
    end
end

"""
    plot_field_evolution(t, Π, Ξ, r; title="Field Evolution")

Plot all timesteps as two heatmaps using the rescaled fields `r\\Pi` and `r\\Xi`.
Rows correspond to time, columns to radius.
"""
function plot_field_evolution(
        t::AbstractVector,
        Π::AbstractMatrix,
        Ξ::AbstractMatrix,
        r::AbstractVector;
        title::AbstractString = L"Field\ Evolution"
    )
    length(t) == size(Π, 2) ||
        throw(DimensionMismatch("`t` length must match number of time snapshots."))

    rΠ, rΞ = _rescaled_fields(Π, Ξ, r)
    redges = _cell_edges(r)
    tedges = _cell_edges(t)

    return with_theme(theme_prd()) do
        fig = Figure(size = (960, 720))
        Label(fig[0, 1:2], title; fontsize = 18)

        ax1 = Axis(fig[1, 1]; xlabel = L"r", ylabel = L"t", title = L"r\Pi(r,t)")
        ax2 = Axis(fig[1, 2]; xlabel = L"r", ylabel = L"t", title = L"r\Xi(r,t)")

        hm1 = heatmap!(ax1, redges, tedges, rΠ; colormap = :balance)
        hm2 = heatmap!(ax2, redges, tedges, rΞ; colormap = :balance)
        Colorbar(fig[2, 1], hm1; vertical = false, label = L"r\Pi")
        Colorbar(fig[2, 2], hm2; vertical = false, label = L"r\Xi")

        return fig
    end
end

"""
    animate_field_evolution(t, Π, Ξ, r; filename="fields.gif", framerate=24, max_frames=200)

Create a GIF of line plots for rescaled fields `r\\Pi` and `r\\Xi`.
"""
function animate_field_evolution(
        t::AbstractVector,
        Π::AbstractMatrix,
        Ξ::AbstractMatrix,
        r::AbstractVector;
        filename::AbstractString = "fields.gif",
        framerate::Int = 24,
        max_frames::Int = 200
    )
    nt = length(t)
    nt == size(Π, 2) ||
        throw(DimensionMismatch("`t` length must match number of time snapshots."))

    rΠ, rΞ = _rescaled_fields(Π, Ξ, r)

    stride = max(1, cld(nt, max(1, max_frames)))
    frame_idx = collect(1:stride:nt)

    y1min, y1max = extrema(rΠ)
    y2min, y2max = extrema(rΞ)
    if y1min == y1max
        y1min -= 1.0e-12
        y1max += 1.0e-12
    end
    if y2min == y2max
        y2min -= 1.0e-12
        y2max += 1.0e-12
    end

    with_theme(theme_prd()) do
        fig = Figure(size = (1280, 460))
        colgap!(fig.layout, 28)

        ax1 = Axis(fig[1, 1]; xlabel = L"r", ylabel = L"r\Pi(r,t)", title = L"r\Pi(r,t)")
        ax2 = Axis(fig[1, 2]; xlabel = L"r", ylabel = L"r\Xi(r,t)", title = L"r\Xi(r,t)")

        ylims!(ax1, y1min, y1max)
        ylims!(ax2, y2min, y2max)

        iobs = Observable(first(frame_idx))
        rΠ_obs = @lift(rΠ[:, $iobs])
        rΞ_obs = @lift(rΞ[:, $iobs])
        time_obs = @lift(latexstring("t = ", @sprintf("%.6g", t[$iobs])))

        lines!(ax1, r, rΠ_obs; color = :steelblue4, linewidth = 2.6)
        lines!(ax2, r, rΞ_obs; color = :firebrick3, linewidth = 2.6)
        hlines!(ax1, [0.0]; color = RGBAf(0.0, 0.0, 0.0, 0.75), linewidth = 1.0)
        hlines!(ax2, [0.0]; color = RGBAf(0.0, 0.0, 0.0, 0.75), linewidth = 1.0)
        vlines!(
            ax1, [0.0]; color = RGBAf(0.0, 0.0, 0.0, 0.4),
            linewidth = 1.0, linestyle = :dash
        )
        vlines!(
            ax2, [0.0]; color = RGBAf(0.0, 0.0, 0.0, 0.4),
            linewidth = 1.0, linestyle = :dash
        )
        Label(fig[0, 1:2], time_obs; fontsize = 18)

        record(fig, filename, frame_idx; framerate = framerate) do k
            iobs[] = k
        end
    end

    return filename
end

function _to_nameable(v)
    return v === nothing ? "auto" : v
end

"""
    generate_simulation_plots(result, run_params; make_animation=true, framerate=24, max_frames=200)

Save standard plots to DrWatson plots directory:
`plots/wave/<savename(run_params)>/`.
"""
function generate_simulation_plots(
        result,
        run_params::NamedTuple;
        make_animation::Bool = true,
        framerate::Int = 24,
        max_frames::Int = 200
    )
    name_params = merge(run_params, (dt = _to_nameable(get(run_params, :dt, nothing)),))
    run_name = savename(name_params)
    outdir = plotsdir("wave", run_name)
    mkpath(outdir)

    t = result.sol.t
    r = result.sol.r
    Π = result.sol.Π
    Ξ = result.sol.Ξ
    E = result.sol.energy

    figE = plot_energy_history(t, E)
    save(joinpath(outdir, "energy_history.png"), figE)
    save(joinpath(outdir, "energy_history.pdf"), figE)

    fig0 = plot_initial_data(Π, Ξ, r)
    save(joinpath(outdir, "initial_data.png"), fig0)
    save(joinpath(outdir, "initial_data.pdf"), fig0)

    figF = plot_field_evolution(t, Π, Ξ, r)
    save(joinpath(outdir, "field_evolution.png"), figF)
    save(joinpath(outdir, "field_evolution.pdf"), figF)

    anim_path = ""
    if make_animation
        anim_path = joinpath(outdir, "field_evolution.gif")
        animate_field_evolution(
            t, Π, Ξ, r;
            filename = anim_path,
            framerate = framerate,
            max_frames = max_frames
        )
    end

    return (outdir = outdir, run_name = run_name, animation = anim_path)
end

function plot_bc_energy_comparison(
        sol_reflect, sol_absorb;
        title::AbstractString = L"Boundary\ Condition\ Comparison",
        representation::Symbol = :auto,
        switch_tol::Real = 1.0e-4
    )
    Er = sol_reflect.energy ./ sol_reflect.energy[1]
    Ea = sol_absorb.energy ./ sol_absorb.energy[1]
    dr = Er .- 1.0
    da = Ea .- 1.0
    max_rel_drift = max(maximum(abs.(dr)), maximum(abs.(da)))

    rep = representation
    if rep === :auto
        rep = max_rel_drift <= Float64(switch_tol) ? :drift : :normalized
    end
    rep in (:drift, :normalized) ||
        throw(ArgumentError("`representation` must be :auto, :drift, or :normalized."))

    yr = rep === :drift ? dr : Er
    ya = rep === :drift ? da : Ea
    ylabel_txt = rep === :drift ? L"E/E_0 - 1" : L"E/E_0"
    yref = rep === :drift ? 0.0 : 1.0
    yfmt = rep === :drift ?
        _format_ticks_sci :
        _format_ticks_fixed
    subtitle = latexstring("\\max\\,|E/E_0-1| = ", @sprintf("%.2e", max_rel_drift))

    return with_theme(theme_prd()) do
        fig = Figure(size = (980, 460))
        Label(fig[0, 1], subtitle; fontsize = 13, halign = :right)
        ax = Axis(
            fig[1, 1];
            title = title,
            xlabel = L"t",
            ylabel = ylabel_txt,
            ytickformat = yfmt,
            titlesize = 18,
            xlabelsize = 16,
            ylabelsize = 16,
            xticklabelsize = 12,
            yticklabelsize = 12
        )
        rowsize!(fig.layout, 1, Relative(1))
        colsize!(fig.layout, 1, Relative(1))
        lines!(
            ax, sol_reflect.t, yr;
            label = L"\mathrm{reflecting}", linewidth = 2
        )
        lines!(
            ax, sol_absorb.t, ya;
            label = L"\mathrm{absorbing}", linewidth = 2, linestyle = :dash
        )
        hlines!(
            ax, [yref]; color = RGBAf(0.0, 0.0, 0.0, 0.75),
            linestyle = :dash, linewidth = 1.0
        )
        axislegend(ax; position = :cb, orientation = :horizontal, framevisible = false)
        return fig
    end
end

"""
    theme_prd(; kwargs...)

APS/PRD publication theme wrapper (MakiePublication.jl).
"""
function theme_prd(; kwargs...)
    return theme_aps(; kwargs...)
end
