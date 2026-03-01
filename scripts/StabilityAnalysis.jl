import SphericalSBPOperators: spherical_operators
import SphericalSBPOperators.Plots: build_folded_sbp_dissipation, discrete_energy,
    inward_packet_initial_data, nullspace_and_checkerboard,
    oscillation_metrics, print_suite_summary,
    reflection_ibvp_test, rk4_amplification_max,
    rk4_dt_max_from_spectrum, save_dashboard,
    save_energy_trace, save_nullspace_plot,
    save_reflection_animation, save_reflection_heatmap,
    save_spectrum_overlay_plot, save_spectrum_plot,
    spectral_analysis, test_pole_stability,
    test_pole_stability_dissipative_overlay,
    wave_rhs_vec!, zero_crossings
import SummationByPartsOperators: MattssonNordström2004, SafeMode

using Printf: @printf

"""
    run_stability_suite(; kwargs...)

Full four-step analysis pipeline.

Inputs:
- either pass `ops` with fields `Geven`, `D`, `H`, `B`, `r`, `p`,
- or pass `G`, `D`, `H`, `B`, `r` directly.
"""
function run_stability_suite(;
        ops = nothing,
        G = nothing,
        D = nothing,
        H = nothing,
        B = nothing,
        r = nothing,
        p::Int = 2,
        boundary_condition::Symbol = :absorbing,
        enforce_origin::Bool = true,
        growth_tol::Float64 = 1.0e-12,
        null_tol::Float64 = 1.0e-11,
        checkerboard_zc_ratio::Float64 = 0.6,
        checkerboard_tv_ratio::Float64 = 1.0,
        throw_on_instability::Bool = true,
        reflection_save_every::Int = 1,
        reflection_max_steps::Int = 200_000,
        reflection_max_saved_steps::Int = 2_000,
        make_plots::Bool = true,
        make_animation::Bool = false,
        output_dir::AbstractString = "plots/stability"
    )
    if ops !== nothing
        G = getproperty(ops, :Geven)
        D = getproperty(ops, :D)
        H = getproperty(ops, :H)
        B = getproperty(ops, :B)
        r = getproperty(ops, :r)
        if hasproperty(ops, :p)
            p = Int(getproperty(ops, :p))
        end
    end

    G === nothing && throw(ArgumentError("Pass `G` or `ops`."))
    D === nothing && throw(ArgumentError("Pass `D` or `ops`."))
    H === nothing && throw(ArgumentError("Pass `H` or `ops`."))
    r === nothing && throw(ArgumentError("Pass `r` or `ops`."))

    n = size(D, 1)
    size(G, 1) == n == size(H, 1) ||
        throw(DimensionMismatch("`D`, `G`, `H` must have matching sizes."))

    spectral = spectral_analysis(
        D,
        G;
        H = H,
        B = B,
        boundary_condition = boundary_condition,
        enforce_origin = enforce_origin,
        growth_tol = growth_tol,
        throw_on_instability = throw_on_instability
    )

    nullspace = nullspace_and_checkerboard(
        spectral.eigvals,
        spectral.eigvecs,
        n;
        null_tol = null_tol,
        checkerboard_zc_ratio = checkerboard_zc_ratio,
        checkerboard_tv_ratio = checkerboard_tv_ratio
    )

    reflection = reflection_ibvp_test(
        D,
        G,
        H,
        r;
        B = B,
        boundary_condition = boundary_condition,
        enforce_origin = enforce_origin,
        p = p,
        rk4_dt_limit = spectral.rk4_dt_max_full_spectrum,
        save_every = reflection_save_every,
        max_steps = reflection_max_steps,
        max_saved_steps = reflection_max_saved_steps
    )

    plot_files = Dict{Symbol, String}()

    if make_plots
        mkpath(output_dir)

        spectrum_path = joinpath(output_dir, "eigen_spectrum.png")
        null_path = joinpath(output_dir, "nullspace_modes.png")
        heat_path = joinpath(output_dir, "reflection_waterfall.png")
        energy_path = joinpath(output_dir, "energy_trace.png")
        dashboard_path = joinpath(output_dir, "stability_dashboard.png")

        save_spectrum_plot(spectral.eigvals, spectrum_path)
        save_nullspace_plot(nullspace, null_path)
        save_reflection_heatmap(reflection, heat_path)
        save_energy_trace(reflection, energy_path)
        save_dashboard((; spectral, nullspace, reflection), dashboard_path)

        plot_files[:spectrum] = spectrum_path
        plot_files[:nullspace] = null_path
        plot_files[:waterfall] = heat_path
        plot_files[:energy] = energy_path
        plot_files[:dashboard] = dashboard_path

        if make_animation
            mp4_path = joinpath(output_dir, "reflection_animation.mp4")
            save_reflection_animation(reflection, r, mp4_path)
            plot_files[:animation] = mp4_path
        end
    end

    return (
        spectral = spectral,
        nullspace = nullspace,
        reflection = reflection,
        plots = plot_files,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    source = MattssonNordström2004()
    ops = spherical_operators(
        source;
        accuracy_order = 8,
        N = 24,
        R = 1.0,
        p = 2,
        mode = SafeMode()
    )

    pole_stability = test_pole_stability(
        Matrix(ops.D),
        Matrix(ops.Geven),
        Matrix(ops.H);
        B = Matrix(ops.B),
        boundary_condition = :reflecting,
        enforce_origin = true,
        tol = 1.0e-10,
        spectrum_plot_path = "plots/stability/pole_closed_spectrum_reflecting.png",
        make_plot = true
    )
    @printf(
        "Reflective closed-operator stability tuple: (is_stable=%s, max_real=%.12e)\n",
        string(pole_stability[1]), pole_stability[2]
    )

    diss_overlay = test_pole_stability_dissipative_overlay(
        ops;
        epsilon = 0.05,
        boundary_condition = :absorbing,
        enforce_origin = true,
        tol = 1.0e-10,
        overlay_plot_path = "plots/stability/pole_closed_spectrum_dissipative_overlay.png",
        make_plot = true,
        throw_on_violation = true
    )
    @printf(
        "Absorbing dissipative stability: (is_stable=%s, baseline_max_real=%.12e, dissipative_max_real=%.12e)\n",
        string(diss_overlay.is_stable),
        diss_overlay.max_real_baseline,
        diss_overlay.max_real_dissipative
    )

    report = run_stability_suite(
        ;
        ops = ops,
        boundary_condition = :reflecting,
        growth_tol = 1.0e-12,
        null_tol = 1.0e-11,
        throw_on_instability = false,
        reflection_max_saved_steps = 800,
        output_dir = "plots/stability"
    )
    print_suite_summary(report)
end
