using CairoMakie
using DrWatson
using LaTeXStrings
using MakiePublication: theme_aps
using Printf
using SphericalSBPOperators

include(joinpath(@__DIR__, "run_wave_evolution.jl"))

"""
    plot_energy_history(t, energies; title="Energy Conservation")

Plot energy history in publication-friendly form.

- For very small drift, plot `E/E0 - 1` so y-axis reflects true order (e.g. `1e-17`).
- Otherwise plot `E/E0`.
"""
function plot_energy_history(t::AbstractVector,
    energies::AbstractVector;
    title::AbstractString=L"Energy\ Conservation",
    representation::Symbol=:auto,
    switch_tol::Real=1e-4)
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
           (values -> [@sprintf("%.1e", Float64(v)) for v in values]) :
           (values -> [@sprintf("%.6f", Float64(v)) for v in values])
    subtitle = latexstring("\\max\\,|E/E_0-1| = ", @sprintf("%.2e", max_rel_drift))

    return with_theme(theme_prd()) do
        fig = Figure(size=(980, 460))
        Label(fig[0, 1], subtitle; fontsize=13, halign=:right)
        ax = Axis(fig[1, 1];
            title=title,
            xlabel=L"t",
            ylabel=ylabel_txt,
            ytickformat=yfmt,
            titlesize=18,
            xlabelsize=16,
            ylabelsize=16,
            xticklabelsize=12,
            yticklabelsize=12)
        rowsize!(fig.layout, 1, Relative(1))
        colsize!(fig.layout, 1, Relative(1))
        lines!(ax, t, yvals; color=:dodgerblue4, linewidth=2)
        hlines!(ax, [yref]; color=:black, linestyle=:dash, linewidth=1)
        return fig
    end
end

function _rescaled_fields(Π::AbstractMatrix, Ξ::AbstractMatrix, r::AbstractVector)
    n, nt = size(Π)
    size(Ξ) == (n, nt) || throw(DimensionMismatch("`Π` and `Ξ` must have same shape."))
    length(r) == n || throw(DimensionMismatch("`r` length must match field rows."))

    rΠ = Matrix{Float64}(undef, n, nt)
    rΞ = Matrix{Float64}(undef, n, nt)
    @inbounds for k in 1:nt
        @views rΠ[:, k] .= r .* Π[:, k]
        @views rΞ[:, k] .= r .* Ξ[:, k]
    end
    return rΠ, rΞ
end

function _cell_edges(v::AbstractVector)
    n = length(v)
    n >= 2 || throw(ArgumentError("Need at least two points to build cell edges."))
    e = Vector{Float64}(undef, n + 1)
    @inbounds for i in 2:n
        e[i] = 0.5 * (Float64(v[i-1]) + Float64(v[i]))
    end
    e[1] = Float64(v[1]) - (e[2] - Float64(v[1]))
    e[end] = Float64(v[end]) + (Float64(v[end]) - e[end-1])
    return e
end

"""
    plot_initial_data(Π, Ξ, r; title="Initial Data (rescaled by r)")

Plot the initial rescaled fields `r\\Pi(r,0)` and `r\\Xi(r,0)`.
"""
function plot_initial_data(Π::AbstractMatrix,
    Ξ::AbstractMatrix,
    r::AbstractVector;
    title::AbstractString=L"Initial\ Data")
    size(Π, 2) >= 1 || throw(ArgumentError("`Π` must contain at least one timestep."))
    size(Ξ, 2) >= 1 || throw(ArgumentError("`Ξ` must contain at least one timestep."))
    length(r) == size(Π, 1) || throw(DimensionMismatch("`r` length must match field rows."))
    size(Ξ) == size(Π) || throw(DimensionMismatch("`Π` and `Ξ` must have same shape."))

    rΠ0 = r .* view(Π, :, 1)
    rΞ0 = r .* view(Ξ, :, 1)

    return with_theme(theme_prd()) do
        fig = Figure(size=(920, 380))
        Label(fig[0, 1:2], title; fontsize=18)

        ax1 = Axis(fig[1, 1]; xlabel=L"r", ylabel=L"r\Pi(r,0)", title=L"r\Pi(r,0)")
        ax2 = Axis(fig[1, 2]; xlabel=L"r", ylabel=L"r\Xi(r,0)", title=L"r\Xi(r,0)")
        lines!(ax1, r, rΠ0; color=:steelblue4, linewidth=2.6)
        lines!(ax2, r, rΞ0; color=:firebrick3, linewidth=2.6)
        hlines!(ax1, [0.0]; color=RGBAf(0.0, 0.0, 0.0, 0.75), linewidth=1.0)
        hlines!(ax2, [0.0]; color=RGBAf(0.0, 0.0, 0.0, 0.75), linewidth=1.0)
        vlines!(ax1, [0.0]; color=RGBAf(0.0, 0.0, 0.0, 0.4), linewidth=1.0, linestyle=:dash)
        vlines!(ax2, [0.0]; color=RGBAf(0.0, 0.0, 0.0, 0.4), linewidth=1.0, linestyle=:dash)
        return fig
    end
end

"""
    plot_field_evolution(t, Π, Ξ, r; title="Field Evolution")

Plot all timesteps as two heatmaps using the rescaled fields `r\\Pi` and `r\\Xi`.
Rows correspond to time, columns to radius.
"""
function plot_field_evolution(t::AbstractVector,
    Π::AbstractMatrix,
    Ξ::AbstractMatrix,
    r::AbstractVector;
    title::AbstractString=L"Field\ Evolution")
    length(t) == size(Π, 2) || throw(DimensionMismatch("`t` length must match number of time snapshots."))

    rΠ, rΞ = _rescaled_fields(Π, Ξ, r)
    redges = _cell_edges(r)
    tedges = _cell_edges(t)

    return with_theme(theme_prd()) do
        fig = Figure(size=(960, 720))
        Label(fig[0, 1:2], title; fontsize=18)

        ax1 = Axis(fig[1, 1]; xlabel=L"r", ylabel=L"t", title=L"r\Pi(r,t)")
        ax2 = Axis(fig[1, 2]; xlabel=L"r", ylabel=L"t", title=L"r\Xi(r,t)")

        hm1 = heatmap!(ax1, redges, tedges, rΠ; colormap=:balance)
        hm2 = heatmap!(ax2, redges, tedges, rΞ; colormap=:balance)
        Colorbar(fig[2, 1], hm1; vertical=false, label=L"r\Pi")
        Colorbar(fig[2, 2], hm2; vertical=false, label=L"r\Xi")

        return fig
    end
end

"""
    animate_field_evolution(t, Π, Ξ, r; filename="fields.gif", framerate=24, max_frames=200)

Create a GIF of line plots for rescaled fields `r\\Pi` and `r\\Xi`.
"""
function animate_field_evolution(t::AbstractVector,
    Π::AbstractMatrix,
    Ξ::AbstractMatrix,
    r::AbstractVector;
    filename::AbstractString="fields.gif",
    framerate::Int=24,
    max_frames::Int=200)
    nt = length(t)
    nt == size(Π, 2) || throw(DimensionMismatch("`t` length must match number of time snapshots."))

    rΠ, rΞ = _rescaled_fields(Π, Ξ, r)

    stride = max(1, cld(nt, max(1, max_frames)))
    frame_idx = collect(1:stride:nt)

    y1min, y1max = extrema(rΠ)
    y2min, y2max = extrema(rΞ)
    if y1min == y1max
        y1min -= 1e-12
        y1max += 1e-12
    end
    if y2min == y2max
        y2min -= 1e-12
        y2max += 1e-12
    end

    with_theme(theme_prd()) do
        fig = Figure(size=(1280, 460))
        colgap!(fig.layout, 28)

        ax1 = Axis(fig[1, 1]; xlabel=L"r", ylabel=L"r\Pi(r,t)", title=L"r\Pi(r,t)")
        ax2 = Axis(fig[1, 2]; xlabel=L"r", ylabel=L"r\Xi(r,t)", title=L"r\Xi(r,t)")

        ylims!(ax1, y1min, y1max)
        ylims!(ax2, y2min, y2max)

        iobs = Observable(first(frame_idx))
        rΠ_obs = @lift(rΠ[:, $iobs])
        rΞ_obs = @lift(rΞ[:, $iobs])
        time_obs = @lift(latexstring("t = ", @sprintf("%.6g", t[$iobs])))

        lines!(ax1, r, rΠ_obs; color=:steelblue4, linewidth=2.6)
        lines!(ax2, r, rΞ_obs; color=:firebrick3, linewidth=2.6)
        hlines!(ax1, [0.0]; color=RGBAf(0.0, 0.0, 0.0, 0.75), linewidth=1.0)
        hlines!(ax2, [0.0]; color=RGBAf(0.0, 0.0, 0.0, 0.75), linewidth=1.0)
        vlines!(ax1, [0.0]; color=RGBAf(0.0, 0.0, 0.0, 0.4), linewidth=1.0, linestyle=:dash)
        vlines!(ax2, [0.0]; color=RGBAf(0.0, 0.0, 0.0, 0.4), linewidth=1.0, linestyle=:dash)
        Label(fig[0, 1:2], time_obs; fontsize=18)

        record(fig, filename, frame_idx; framerate=framerate) do k
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
function generate_simulation_plots(result,
    run_params::NamedTuple;
    make_animation::Bool=true,
    framerate::Int=24,
    max_frames::Int=200)
    name_params = merge(run_params, (dt=_to_nameable(get(run_params, :dt, nothing)),))
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
        animate_field_evolution(t, Π, Ξ, r;
            filename=anim_path,
            framerate=framerate,
            max_frames=max_frames)
    end

    return (outdir=outdir, run_name=run_name, animation=anim_path)
end

"""
    run_and_plot_wave(; kwargs...)

Run a wave evolution and generate:
1. energy-vs-time plot,
2. full field-evolution heatmaps (all timesteps),
3. animation (GIF).

All saved under DrWatson `plotsdir("wave", savename(...))`.
"""
function run_and_plot_wave(;
    # Discretization / physics
    accuracy_order::Int=6,
    N::Int=64,
    R::Real=1.0,
    p::Int=2,
    # Time evolution
    T_final::Real=0.2,
    dt=nothing,
    alg=nothing,
    safety_factor::Real=0.9,
    boundary_condition::Symbol=:absorbing,
    ϕ0=nothing,
    Π0=nothing,
    Ξ0=nothing,
    phi0=nothing,
    pi0=nothing,
    xi0=nothing,
    initial_data_tag::AbstractString="default_profile",
    save_every::Int=1,
    build_matrix::Symbol=:matrix_if_square,
    noise_amplitude::Real=0.0,
    noise_seed=nothing,
    # Plot controls
    make_animation::Bool=true,
    framerate::Int=24,
    max_frames::Int=200,
    verbose::Bool=true)
    run_params = (
        accuracy_order=accuracy_order,
        N=N,
        R=R,
        p=p,
        T_final=T_final,
        dt=dt,
        alg=alg === nothing ? "auto" : string(typeof(alg)),
        safety_factor=safety_factor,
        boundary_condition=boundary_condition,
        initial_data_tag=initial_data_tag,
        save_every=save_every,
        build_matrix=build_matrix,
        noise_amplitude=noise_amplitude,
        noise_seed=_to_nameable(noise_seed)
    )

    result = run_wave_evolution(;
        accuracy_order=accuracy_order,
        N=N,
        R=R,
        p=p,
        T_final=T_final,
        dt=dt,
        alg=alg,
        safety_factor=safety_factor,
        boundary_condition=boundary_condition,
        ϕ0=ϕ0,
        Π0=Π0,
        Ξ0=Ξ0,
        phi0=phi0,
        pi0=pi0,
        xi0=xi0,
        save_every=save_every,
        build_matrix=build_matrix,
        noise_amplitude=noise_amplitude,
        noise_seed=noise_seed,
        verbose=verbose)
    plot_info = generate_simulation_plots(result, run_params;
        make_animation=make_animation,
        framerate=framerate,
        max_frames=max_frames)

    if verbose
        println("\nSaved plots to: ", plot_info.outdir)
        if make_animation
            println("Saved animation: ", plot_info.animation)
        end
    end

    return merge(result, (plots=plot_info,))
end

function _energy_diagnostics(sol)
    E = sol.energy
    dE = diff(E)
    E0 = E[1]
    Ef = E[end]
    return (
        E0=E0,
        Ef=Ef,
        normalized_final=Ef / E0,
        max_step_increase=maximum(dE),
        max_step_decrease=minimum(dE),
        nonincreasing_with_tol_1e6=maximum(dE) <= 1e-6,
    )
end

function plot_bc_energy_comparison(sol_reflect, sol_absorb;
    title::AbstractString=L"Boundary\ Condition\ Comparison",
    representation::Symbol=:auto,
    switch_tol::Real=1e-4)
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
           (values -> [@sprintf("%.1e", Float64(v)) for v in values]) :
           (values -> [@sprintf("%.6f", Float64(v)) for v in values])
    subtitle = latexstring("\\max\\,|E/E_0-1| = ", @sprintf("%.2e", max_rel_drift))

    return with_theme(theme_prd()) do
        fig = Figure(size=(980, 460))
        Label(fig[0, 1], subtitle; fontsize=13, halign=:right)
        ax = Axis(fig[1, 1];
            title=title,
            xlabel=L"t",
            ylabel=ylabel_txt,
            ytickformat=yfmt,
            titlesize=18,
            xlabelsize=16,
            ylabelsize=16,
            xticklabelsize=12,
            yticklabelsize=12)
        rowsize!(fig.layout, 1, Relative(1))
        colsize!(fig.layout, 1, Relative(1))
        lines!(ax, sol_reflect.t, yr;
            label=L"\mathrm{reflecting}", linewidth=2)
        lines!(ax, sol_absorb.t, ya;
            label=L"\mathrm{absorbing}", linewidth=2, linestyle=:dash)
        hlines!(ax, [yref]; color=RGBAf(0.0, 0.0, 0.0, 0.75), linestyle=:dash, linewidth=1.0)
        axislegend(ax; position=:cb, orientation=:horizontal, framevisible=false)
        return fig
    end
end

"""
    run_reflecting_absorbing_campaign(; kwargs...)

Run two 6th-order wave simulations with:
- grid spacing `dr = 0.1`,
- CFL `cfl = 0.25`,
- `ImplicitMidpoint()` for the reflecting run,
- one reflecting run and one absorbing run,
and save plots/animations for both using DrWatson paths and savename.
"""
function run_reflecting_absorbing_campaign(;
    accuracy_order::Int=6,
    p::Int=2,
    R::Real=16.0,
    dr::Real=0.1,
    cfl::Real=0.25,
    T_reflecting::Real=32.0,
    T_absorbing::Real=60.0,
    initial_data_tag::AbstractString="pi_bumpb_radius1_center0_xi_zero",
    save_every::Int=1,
    make_animation::Bool=true,
    framerate::Int=24,
    max_frames::Int=400,
    build_matrix::Symbol=:matrix_if_square,
    verbose::Bool=true)
    dr > 0 || throw(ArgumentError("`dr` must be positive."))
    cfl > 0 || throw(ArgumentError("`cfl` must be positive."))

    N = round(Int, R / dr)
    isapprox(N * dr, R; atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("`R/dr` must be an integer to realize the requested spacing exactly. Got R=$R, dr=$dr."))

    dt = cfl * dr
    Π0_fn = r -> bumpb_profile(r; amplitude = 1.0, center = 0.0, radius = 1.0)
    Ξ0_fn = r -> zeros(eltype(r), length(r))

    common_kwargs = (
        accuracy_order=accuracy_order,
        N=N,
        R=R,
        p=p,
        dt=dt,
        Π0=Π0_fn,
        Ξ0=Ξ0_fn,
        initial_data_tag=initial_data_tag,
        save_every=save_every,
        build_matrix=build_matrix,
        make_animation=make_animation,
        framerate=framerate,
        max_frames=max_frames,
        verbose=verbose,
    )

    result_reflect = run_and_plot_wave(;
        common_kwargs...,
        T_final=T_reflecting,
        boundary_condition=:reflecting,
        alg=ImplicitMidpoint())
    result_absorb = run_and_plot_wave(;
        common_kwargs...,
        T_final=T_absorbing,
        boundary_condition=:absorbing)

    diag_reflect = _energy_diagnostics(result_reflect.sol)
    diag_absorb = _energy_diagnostics(result_absorb.sol)

    comp_params = (
        case="reflecting_absorbing_campaign",
        accuracy_order=accuracy_order,
        p=p,
        R=R,
        dr=dr,
        cfl=cfl,
        dt=dt,
        initial_data_tag=initial_data_tag,
        T_reflecting=T_reflecting,
        T_absorbing=T_absorbing,
    )
    comp_dir = plotsdir("wave", "bc_comparison", savename(comp_params))
    mkpath(comp_dir)
    fig_comp = plot_bc_energy_comparison(result_reflect.sol, result_absorb.sol)
    save(joinpath(comp_dir, "energy_comparison.png"), fig_comp)
    save(joinpath(comp_dir, "energy_comparison.pdf"), fig_comp)

    if verbose
        println("\nTwo-BC campaign summary")
        println("  accuracy_order = ", accuracy_order, ", dr = ", dr, ", CFL = ", cfl, ", dt = ", dt)
        println("  reflecting: steps = ", result_reflect.sol.nsteps, ", E_final/E0 = ", diag_reflect.normalized_final,
            ", max step increase = ", diag_reflect.max_step_increase)
        println("  absorbing : steps = ", result_absorb.sol.nsteps, ", E_final/E0 = ", diag_absorb.normalized_final,
            ", max step increase = ", diag_absorb.max_step_increase)
        println("  comparison plot dir = ", comp_dir)
        println("  reflecting plot dir = ", result_reflect.plots.outdir)
        println("  absorbing plot dir = ", result_absorb.plots.outdir)
    end

    return (
        reflecting=result_reflect,
        absorbing=result_absorb,
        diagnostics=(reflecting=diag_reflect, absorbing=diag_absorb),
        comparison_dir=comp_dir,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_reflecting_absorbing_campaign()
end


"""
    theme_prd(; kwargs...)

APS/PRD publication theme wrapper (MakiePublication.jl).
"""
function theme_prd(; kwargs...)
    return theme_aps(; kwargs...)
end
