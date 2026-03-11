import CairoMakie: save
import DrWatson: plotsdir, savename
import OrdinaryDiffEqSDIRK: ImplicitMidpoint
import SphericalSBPOperators: bumpb_profile
import SphericalSBPOperators.Plots: animate_field_evolution, generate_simulation_plots,
    plot_bc_energy_comparison, plot_energy_history,
    plot_field_evolution, plot_initial_data, theme_prd

include(joinpath(@__DIR__, "run_wave_evolution.jl"))

function _to_nameable(v)
    return v === nothing ? "auto" : v
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
        accuracy_order::Int = 6,
        N::Int = 64,
        R::Real = 1.0,
        p::Int = 2,
        # Time evolution
        T_final::Real = 0.2,
        dt = nothing,
        alg = nothing,
        safety_factor::Real = 0.9,
        boundary_condition::Symbol = :absorbing,
        phi0 = nothing,
        pi0 = nothing,
        xi0 = nothing,
        initial_data_tag::AbstractString = "default_profile",
        save_every::Int = 1,
        build_matrix::Symbol = :matrix_if_square,
        noise_amplitude::Real = 0.0,
        noise_seed = nothing,
        # Plot controls
        make_animation::Bool = true,
        framerate::Int = 24,
        max_frames::Int = 200,
        verbose::Bool = true,
        kwargs...
    )
    phi0_unicode = get(kwargs, :ϕ0, nothing)
    pi0_unicode = get(kwargs, :Π0, nothing)
    xi0_unicode = get(kwargs, :Ξ0, nothing)

    if !isnothing(phi0) && !isnothing(phi0_unicode)
        throw(ArgumentError("Use either `phi0` or `ϕ0`, not both."))
    end
    if !isnothing(pi0) && !isnothing(pi0_unicode)
        throw(ArgumentError("Use either `pi0` or `Π0`, not both."))
    end
    if !isnothing(xi0) && !isnothing(xi0_unicode)
        throw(ArgumentError("Use either `xi0` or `Ξ0`, not both."))
    end

    phi0_value = isnothing(phi0) ? phi0_unicode : phi0
    pi0_value = isnothing(pi0) ? pi0_unicode : pi0
    xi0_value = isnothing(xi0) ? xi0_unicode : xi0

    run_params = (
        accuracy_order = accuracy_order,
        N = N,
        R = R,
        p = p,
        T_final = T_final,
        dt = dt,
        alg = alg === nothing ? "auto" : string(typeof(alg)),
        safety_factor = safety_factor,
        boundary_condition = boundary_condition,
        initial_data_tag = initial_data_tag,
        save_every = save_every,
        build_matrix = build_matrix,
        noise_amplitude = noise_amplitude,
        noise_seed = _to_nameable(noise_seed),
    )

    result = run_wave_evolution(;
        accuracy_order = accuracy_order,
        N = N,
        R = R,
        p = p,
        T_final = T_final,
        dt = dt,
        alg = alg,
        safety_factor = safety_factor,
        boundary_condition = boundary_condition,
        phi0 = phi0_value,
        pi0 = pi0_value,
        xi0 = xi0_value,
        save_every = save_every,
        build_matrix = build_matrix,
        noise_amplitude = noise_amplitude,
        noise_seed = noise_seed,
        verbose = verbose
    )
    plot_info = generate_simulation_plots(
        result, run_params;
        make_animation = make_animation,
        framerate = framerate,
        max_frames = max_frames
    )

    if verbose
        println("\nSaved plots to: ", plot_info.outdir)
        if make_animation
            println("Saved animation: ", plot_info.animation)
        end
    end

    return merge(result, (plots = plot_info,))
end

function _energy_diagnostics(sol)
    E = sol.energy
    dE = diff(E)
    E0 = E[1]
    Ef = E[end]
    return (
        E0 = E0,
        Ef = Ef,
        normalized_final = Ef / E0,
        max_step_increase = maximum(dE),
        max_step_decrease = minimum(dE),
        nonincreasing_with_tol_1e6 = maximum(dE) <= 1.0e-6,
    )
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
        accuracy_order::Int = 6,
        p::Int = 2,
        R::Real = 16.0,
        dr::Real = 0.1,
        cfl::Real = 0.25,
        T_reflecting::Real = 32.0,
        T_absorbing::Real = 60.0,
        initial_data_tag::AbstractString = "pi_bumpb_radius1_center0_xi_zero",
        save_every::Int = 1,
        make_animation::Bool = true,
        framerate::Int = 24,
        max_frames::Int = 400,
        build_matrix::Symbol = :matrix_if_square,
        verbose::Bool = true
    )
    dr > 0 || throw(ArgumentError("`dr` must be positive."))
    cfl > 0 || throw(ArgumentError("`cfl` must be positive."))

    N = round(Int, R / dr)
    isapprox(N * dr, R; atol = 1.0e-12, rtol = 1.0e-12) ||
        throw(ArgumentError("`R/dr` must be an integer to realize the requested spacing exactly. Got R=$R, dr=$dr."))

    dt = cfl * dr
    pi0_fn(r) = bumpb_profile(r; amplitude = 1.0, center = 0.0, radius = 1.0)
    xi0_fn(r) = zeros(eltype(r), length(r))

    common_kwargs = (
        accuracy_order = accuracy_order,
        N = N,
        R = R,
        p = p,
        dt = dt,
        pi0 = pi0_fn,
        xi0 = xi0_fn,
        initial_data_tag = initial_data_tag,
        save_every = save_every,
        build_matrix = build_matrix,
        make_animation = make_animation,
        framerate = framerate,
        max_frames = max_frames,
        verbose = verbose,
    )

    result_reflect = run_and_plot_wave(;
        common_kwargs...,
        T_final = T_reflecting,
        boundary_condition = :reflecting,
        alg = ImplicitMidpoint()
    )
    result_absorb = run_and_plot_wave(;
        common_kwargs...,
        T_final = T_absorbing,
        boundary_condition = :absorbing
    )

    diag_reflect = _energy_diagnostics(result_reflect.sol)
    diag_absorb = _energy_diagnostics(result_absorb.sol)

    comp_params = (
        case = "reflecting_absorbing_campaign",
        accuracy_order = accuracy_order,
        p = p,
        R = R,
        dr = dr,
        cfl = cfl,
        dt = dt,
        initial_data_tag = initial_data_tag,
        T_reflecting = T_reflecting,
        T_absorbing = T_absorbing,
    )
    comp_dir = plotsdir("wave", "bc_comparison", savename(comp_params))
    mkpath(comp_dir)
    fig_comp = plot_bc_energy_comparison(result_reflect.sol, result_absorb.sol)
    save(joinpath(comp_dir, "energy_comparison.png"), fig_comp)
    save(joinpath(comp_dir, "energy_comparison.pdf"), fig_comp)

    if verbose
        println("\nTwo-BC campaign summary")
        println(
            "  accuracy_order = ", accuracy_order,
            ", dr = ", dr, ", CFL = ", cfl, ", dt = ", dt
        )
        println(
            "  reflecting: steps = ", result_reflect.sol.nsteps,
            ", E_final/E0 = ", diag_reflect.normalized_final,
            ", max step increase = ", diag_reflect.max_step_increase
        )
        println(
            "  absorbing : steps = ", result_absorb.sol.nsteps,
            ", E_final/E0 = ", diag_absorb.normalized_final,
            ", max step increase = ", diag_absorb.max_step_increase
        )
        println("  comparison plot dir = ", comp_dir)
        println("  reflecting plot dir = ", result_reflect.plots.outdir)
        println("  absorbing plot dir = ", result_absorb.plots.outdir)
    end

    return (
        reflecting = result_reflect,
        absorbing = result_absorb,
        diagnostics = (reflecting = diag_reflect, absorbing = diag_absorb),
        comparison_dir = comp_dir,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_reflecting_absorbing_campaign()
end
