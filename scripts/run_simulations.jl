# Run this script from the repository environment, for example:
#   julia --project=. scripts/run_simulations.jl
#
# Load the package source directly from this checkout.
# This keeps the script convenient while we are developing inside the repository.
include(joinpath(@__DIR__, "..", "src", "SphericalSBPOperators.jl"))

using Dates: format, now
using JLD2: jldsave
using MultiFloats: Float64x2
using OrdinaryDiffEqHighOrderRK: DP8, TsitPap8
using OrdinaryDiffEqLowOrderRK: RK4
using OrdinaryDiffEqSDIRK: ImplicitMidpoint
using Printf: @sprintf
using SummationByPartsOperators: MattssonNordström2004, SafeMode

"""
    run_simulation(; kwargs...)

Construct one operator family through the unified API and run the wave solver.

The script separates the setup into two groups:

- operator construction parameters:
  - `family`: `:diagonal`, `:mixed_order_diagonal`, `:staggered`, `:non_diagonal`,
    or `:non_diagonal_exp`
  - `accuracy_order`, `N`, `R`, `p`, `source`, `mode`, `target_eltype`
  - `outer_boundary_closure_help`: only used for `family=:non_diagonal_exp`; when
    `false`, use the left/origin-only experimental `V` construction without the
    extra outer-boundary closure help
- wave evolution parameters:
  - `T_final`: final simulation time
  - `dt`: fixed time step; use `nothing` to estimate a stable step automatically
  - `boundary_condition`: `:absorbing`, `:reflecting`, `:dirichlet`, or `:none`
  - `alg_name`: `:auto`, `:dp8`, `:tsitpap8`, `:rk4`, or `:implicit_midpoint`
  - `save_every`: save every this many fixed steps
  - `check_provided_dt_stability`: run the dense spectral fixed-`dt` check

The script supports four initial-data presets:

- `:characteristic_incoming_bump`
  - `w_in0` is a compact-support bump profile
  - `w_out0` is set to zero
  - `solve_wave_ode` builds `pi0` and `psi0` from the characteristic variables
- `:even_gaussian_pi`
  - uses the paper-style even Gaussian envelope
  - excites only the left-moving/incoming characteristic mode
  - equivalently, `pi0 = psi0 = A * (exp(-((r-r0)^2)/d^2) + exp(-((r+r0)^2)/d^2))`
- `:even_gaussian_pi_zero_psi`
  - uses the same paper-style even Gaussian envelope for `pi0`
  - sets `psi0 = 0` everywhere initially
  - this is not potential-consistent unless `pi0` is constant, so `Psi` will appear
    immediately through the wave coupling
- `:regular_left_moving_spherical_gaussian`
  - uses the regular analytic image construction for a left-moving spherical Gaussian
  - sets `phi0`, `pi0`, and `psi0` from the exact solution at `t = 0`
  - uses `regular_gaussian_amplitude` and `regular_gaussian_width`
  - uses `regular_gaussian_center` for the pulse center

The script now supports all four unified operator families.
For the non-diagonal path, the standard unified constructor supports accuracy orders 4
and 6. The experimental non-diagonal path is available through `family=:non_diagonal_exp`
and currently targets the 6th-order split-mass construction.
"""

# Default operator setup for the wave run.
const DEFAULT_OPERATOR_CONFIG = (
                                 # Choose the operator family used in the simulation.
                                 family = :non_diagonal,

                                 # Full-grid SBP source family.
                                 source = MattssonNordström2004(),

                                 # Requested design accuracy order.
                                 accuracy_order = 4,

                                 # Number of subintervals on [0, R]; the folded grid has N + 1 nodes.
                                 N = 40,

                                 # Outer radius of the computational domain [0, R].
                                 R = 40.0,

                                 # Metric power in r^p.
                                 p = 2,

                                 # Arithmetic/extraction mode from SummationByPartsOperators.
                                 mode = SafeMode(),

                                 # Experimental SBP6 split-mass option: keep the extra
                                 # right-boundary V closure helper by default.
                                 outer_boundary_closure_help = false,

                                 # Use exact arithmetic during construction when needed, but store the
                                 # operators used by the wave evolution in floating point.
                                 target_eltype = Float64)

# Default wave setup.
const DEFAULT_WAVE_CONFIG = (
                             # Final simulation time.
                             T_final = 25.0,

                             # Fixed time step. Use nothing to estimate a stable step automatically.
                             dt = nothing,

                             # Safety factor applied when dt is estimated from the operator spectrum.
                             safety_factor = 0.9,

                             # Boundary condition imposed at r = R.
                             boundary_condition = :dirichlet,

                             # Time integrator choice.
                             # :auto uses ImplicitMidpoint for reflecting boundaries and DP8 otherwise.
                             alg_name = :dp8,

                             # Save every this many fixed steps.
                             save_every = 1,

                             # Dense spectral checks are useful when choosing a new dt, but too costly
                             # for every resolution in a fixed-CFL convergence sweep.
                             check_provided_dt_stability = true,

                             # Enforce odd symmetry at the origin when the operator includes r = 0.
                             enforce_origin = true,

                             # Initial-data preset.
                             # :characteristic_incoming_bump gives a pure incoming characteristic pulse.
                             # :even_gaussian_pi uses the paper-style even Gaussian envelope,
                             # but excites only the left-moving/incoming mode.
                             # :even_gaussian_pi_zero_psi uses the same Gaussian pi profile
                             # with psi initialized to zero.
                             # :regular_left_moving_spherical_gaussian uses the analytic,
                             # regular left-moving spherical Gaussian pulse.
                             initial_data_kind = :regular_left_moving_spherical_gaussian,

                             # Amplitude of the incoming characteristic bump w_in0.
                             incoming_amplitude = 1.0,

                             # Center of the incoming bump in the same physical coordinates as `r`.
                             incoming_center = 10.0,

                             # Radius of the compact-support incoming bump in physical units.
                             incoming_radius = 2.0,

                             # Outgoing characteristic amplitude multiplier.
                             # Keep this at zero for a pure incoming bump.
                             outgoing_amplitude = 0.0,

                             # Amplitude A of the even Gaussian pi(r,0) profile.
                             gaussian_pi_amplitude = 1.0,

                             # Gaussian center r0 in the paper-style even profile.
                             gaussian_pi_center = 5.0,

                             # Gaussian width d in the paper-style even profile.
                             gaussian_pi_width = 2.0,

                             # Amplitude A of the regular left-moving spherical Gaussian pulse.
                             regular_gaussian_amplitude = 1.0,

                             # Width d of the regular left-moving spherical Gaussian pulse.
                             regular_gaussian_width = 2.0,

                             # Center of the regular left-moving spherical Gaussian pulse
                             # in the same physical coordinates as `r`.
                             regular_gaussian_center = 10.0,

                             # Print a short simulation summary.
                             verbose = true)

function _load_glmakie()
    if !isdefined(@__MODULE__, :GLMakie)
        @eval import GLMakie
    end
    return Base.invokelatest(getfield, @__MODULE__, :GLMakie)
end

function _padded_extrema(values::AbstractArray; relative_pad::Real = 0.08,
                         absolute_pad::Real = 1.0e-12)
    vmin, vmax = extrema(Float64.(values))
    span = vmax - vmin
    pad = max(Float64(absolute_pad),
              Float64(relative_pad) * max(span, max(abs(vmin), abs(vmax), 1.0)))
    return vmin - pad, vmax + pad
end

function _reconstruct_phi_history(ops,
                                  psi_history::AbstractMatrix;
                                  gauge::Symbol = :origin)
    n, nt = size(psi_history)
    length(ops.r) == n ||
        throw(DimensionMismatch("`psi_history` row count must match the operator grid."))
    gauge in (:origin, :mean) ||
        throw(ArgumentError("`gauge` must be :origin or :mean."))

    G = Matrix{Float64}(ops.Geven)
    phi_history = zeros(Float64, n, nt)

    if gauge === :origin
        if n >= 2
            A = @view G[:, 2:n]
            for k in 1:nt
                @views phi_history[2:n, k] .= A \ Float64.(psi_history[:, k])
            end
        end
    else
        A = [G; fill(1.0 / n, 1, n)]
        for k in 1:nt
            rhs = vcat(Float64.(psi_history[:, k]), 0.0)
            @views phi_history[:, k] .= A \ rhs
        end
    end

    return phi_history
end

function _rescale_history(history::AbstractMatrix, r::AbstractVector)
    n, nt = size(history)
    length(r) == n ||
        throw(DimensionMismatch("`r` length must match the history row count."))

    scaled = Matrix{Float64}(undef, n, nt)
    for k in 1:nt
        @views scaled[:, k] .= Float64.(r) .* Float64.(history[:, k])
    end
    return scaled
end

@inline function _extract_solution(simulation)
    return hasproperty(simulation, :sol) ? getproperty(simulation, :sol) : simulation
end

function _common_nested_wave_sampling(sol_h, sol_h2, sol_h4)
    time_sampling, common_t = SphericalSBPOperators._common_time_sampling(sol_h.t,
                                                                          sol_h2.t,
                                                                          sol_h4.t)
    spatial_sampling = SphericalSBPOperators._nested_common_sampling(length(sol_h.r),
                                                                     length(sol_h2.r),
                                                                     length(sol_h4.r))

    common_r = SphericalSBPOperators._validate_nested_vector_alignment(sol_h.r,
                                                                       sol_h2.r,
                                                                       sol_h4.r,
                                                                       spatial_sampling;
                                                                       name = "r")

    return (t = common_t,
            r = common_r,
            time = time_sampling,
            space = spatial_sampling)
end

function _sample_history(history::AbstractMatrix, sampling)
    return Float64.(history[sampling.space, sampling.time])
end

"""
    interactive_wave_viewer(result; gauge=:origin, start_index=1, display_figure=true)

Open an interactive GLMakie viewer for a wave simulation.

The viewer uses the package APS theme and shows three synchronized line plots:

- `r*phi(r,t)`, reconstructed from `Psi(r,t)` by least squares
- `r*Pi(r,t)`
- `r*Psi(r,t)`

A time slider at the bottom selects the snapshot shown in all three panels.
"""
function interactive_wave_viewer(result;
                                 gauge::Symbol = :origin,
                                 start_index::Int = 1,
                                 display_figure::Bool = true)
    GLM = Base.invokelatest(_load_glmakie)
    return Base.invokelatest(_interactive_wave_viewer_impl,
                             GLM,
                             result,
                             gauge,
                             start_index,
                             display_figure)
end

function _interactive_wave_viewer_impl(GLM,
                                       result,
                                       gauge::Symbol,
                                       start_index::Int,
                                       display_figure::Bool)
    ops = result.ops
    sol = result.sol
    nt = length(sol.t)
    1 <= start_index <= nt ||
        throw(ArgumentError("`start_index` must satisfy 1 <= start_index <= $(nt)."))

    GLM.activate!()

    r = Float64.(sol.r)
    pi_history = Float64.(sol.Π)
    psi_history = Float64.(sol.Ψ)
    phi_history = _reconstruct_phi_history(ops, psi_history; gauge = gauge)
    rphi_history = _rescale_history(phi_history, r)
    rpi_history = _rescale_history(pi_history, r)
    rpsi_history = _rescale_history(psi_history, r)

    phi_limits = _padded_extrema(rphi_history)
    pi_limits = _padded_extrema(rpi_history)
    psi_limits = _padded_extrema(rpsi_history)

    fig = GLM.with_theme(SphericalSBPOperators.mytheme_aps()) do
        fig = GLM.Figure(size = (1500, 840), figure_padding = (28, 24, 22, 18))

        GLM.Label(fig[0, 1:3],
                  "Wave Evolution Snapshots";
                  fontsize = 20,
                  font = :bold)

        ax_phi = GLM.Axis(fig[1, 1];
                          xlabel = "r",
                          ylabel = "r*phi(r, t)",
                          title = "r*phi")
        ax_pi = GLM.Axis(fig[1, 2];
                         xlabel = "r",
                         ylabel = "r*Pi(r, t)",
                         title = "r*Pi")
        ax_psi = GLM.Axis(fig[1, 3];
                          xlabel = "r",
                          ylabel = "r*Psi(r, t)",
                          title = "r*Psi")

        for (ax, limits) in ((ax_phi, phi_limits), (ax_pi, pi_limits), (ax_psi, psi_limits))
            GLM.hlines!(ax, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
            GLM.vlines!(ax, [0.0]; color = (:black, 0.35), linewidth = 1.0,
                        linestyle = :dot)
            GLM.xlims!(ax, minimum(r), maximum(r))
            GLM.ylims!(ax, limits[1], limits[2])
        end

        slider = GLM.Slider(fig[3, 1:3];
                            range = 1:nt,
                            startvalue = start_index,
                            snap = true)
        snapshot_index = slider.value

        time_label = GLM.lift(snapshot_index) do idx
            return "t = $(round(Float64(sol.t[idx]); digits = 6))    snapshot = $(idx) / $(nt)"
        end
        GLM.Label(fig[2, 1:3], time_label; fontsize = 15)

        phi_obs = GLM.lift(idx -> rphi_history[:, idx], snapshot_index)
        pi_obs = GLM.lift(idx -> rpi_history[:, idx], snapshot_index)
        psi_obs = GLM.lift(idx -> rpsi_history[:, idx], snapshot_index)

        GLM.lines!(ax_phi, r, phi_obs; color = :royalblue4, linewidth = 2.5)
        GLM.lines!(ax_pi, r, pi_obs; color = :darkorange3, linewidth = 2.5)
        GLM.lines!(ax_psi, r, psi_obs; color = :seagreen4, linewidth = 2.5)

        fig
    end

    screen = display_figure ? GLM.display(fig) : nothing
    return (fig = fig,
            screen = screen,
            phi_history = phi_history,
            rphi_history = rphi_history,
            rpi_history = rpi_history,
            rpsi_history = rpsi_history)
end

"""
    interactive_wave_viewer_resolutions(result_h, result_h2, result_h4;
                                        gauge=:origin, start_index=1,
                                        labels=("dr", "dr/2", "dr/4"),
                                        display_figure=true)

Open an interactive GLMakie viewer that overlays three nested resolutions on the same
common sampled grid and common sampled times, following the same coarse/medium/fine
sampling used by the convergence utilities.

The viewer shows synchronized snapshots of:

- `r*phi(r,t)`
- `r*Pi(r,t)`
- `r*Psi(r,t)`

for the three supplied simulations, with a shared time slider.
"""
function interactive_wave_viewer_resolutions(result_h,
                                             result_h2,
                                             result_h4;
                                             gauge::Symbol = :origin,
                                             start_index::Int = 1,
                                             labels::NTuple{3, AbstractString} = ("dr",
                                                                                  "dr/2",
                                                                                  "dr/4"),
                                             display_figure::Bool = true)
    GLM = Base.invokelatest(_load_glmakie)
    return Base.invokelatest(_interactive_wave_viewer_resolutions_impl,
                             GLM,
                             result_h,
                             result_h2,
                             result_h4,
                             gauge,
                             start_index,
                             labels,
                             display_figure)
end

function _interactive_wave_viewer_resolutions_impl(GLM,
                                                   result_h,
                                                   result_h2,
                                                   result_h4,
                                                   gauge::Symbol,
                                                   start_index::Int,
                                                   labels::NTuple{3, AbstractString},
                                                   display_figure::Bool)
    sol_h = _extract_solution(result_h)
    sol_h2 = _extract_solution(result_h2)
    sol_h4 = _extract_solution(result_h4)

    ops_h = hasproperty(result_h, :ops) ? getproperty(result_h, :ops) : nothing
    ops_h2 = hasproperty(result_h2, :ops) ? getproperty(result_h2, :ops) : nothing
    ops_h4 = hasproperty(result_h4, :ops) ? getproperty(result_h4, :ops) : nothing
    isnothing(ops_h) &&
        throw(ArgumentError("The coarse simulation must provide an `ops` field."))
    isnothing(ops_h2) &&
        throw(ArgumentError("The medium simulation must provide an `ops` field."))
    isnothing(ops_h4) &&
        throw(ArgumentError("The fine simulation must provide an `ops` field."))

    sampling = _common_nested_wave_sampling(sol_h, sol_h2, sol_h4)
    nt = length(sampling.t)
    1 <= start_index <= nt ||
        throw(ArgumentError("`start_index` must satisfy 1 <= start_index <= $(nt)."))

    GLM.activate!()

    phi_h = _reconstruct_phi_history(ops_h, Float64.(sol_h.Ψ); gauge = gauge)
    phi_h2 = _reconstruct_phi_history(ops_h2, Float64.(sol_h2.Ψ); gauge = gauge)
    phi_h4 = _reconstruct_phi_history(ops_h4, Float64.(sol_h4.Ψ); gauge = gauge)

    common_r = Float64.(sampling.r)
    rphi_h = common_r .* _sample_history(phi_h,
                             (; space = sampling.space.coarse,
                              time = sampling.time.coarse))
    rphi_h2 = common_r .* _sample_history(phi_h2,
                              (; space = sampling.space.medium,
                               time = sampling.time.medium))
    rphi_h4 = common_r .* _sample_history(phi_h4,
                              (; space = sampling.space.fine,
                               time = sampling.time.fine))

    rpi_h = common_r .* _sample_history(Float64.(sol_h.Π),
                            (; space = sampling.space.coarse,
                             time = sampling.time.coarse))
    rpi_h2 = common_r .* _sample_history(Float64.(sol_h2.Π),
                             (; space = sampling.space.medium,
                              time = sampling.time.medium))
    rpi_h4 = common_r .* _sample_history(Float64.(sol_h4.Π),
                             (; space = sampling.space.fine,
                              time = sampling.time.fine))

    rpsi_h = common_r .* _sample_history(Float64.(sol_h.Ψ),
                             (; space = sampling.space.coarse,
                              time = sampling.time.coarse))
    rpsi_h2 = common_r .* _sample_history(Float64.(sol_h2.Ψ),
                              (; space = sampling.space.medium,
                               time = sampling.time.medium))
    rpsi_h4 = common_r .* _sample_history(Float64.(sol_h4.Ψ),
                              (; space = sampling.space.fine,
                               time = sampling.time.fine))

    phi_limits = _padded_extrema(vcat(vec(rphi_h), vec(rphi_h2), vec(rphi_h4)))
    pi_limits = _padded_extrema(vcat(vec(rpi_h), vec(rpi_h2), vec(rpi_h4)))
    psi_limits = _padded_extrema(vcat(vec(rpsi_h), vec(rpsi_h2), vec(rpsi_h4)))

    fig = GLM.with_theme(SphericalSBPOperators.mytheme_aps()) do
        fig = GLM.Figure(size = (1500, 840), figure_padding = (28, 24, 22, 18))

        GLM.Label(fig[0, 1:3],
                  "Wave Evolution Snapshots Across Resolutions";
                  fontsize = 20,
                  font = :bold)

        ax_phi = GLM.Axis(fig[1, 1];
                          xlabel = "r",
                          ylabel = "r*phi(r, t)",
                          title = "r*phi")
        ax_pi = GLM.Axis(fig[1, 2];
                         xlabel = "r",
                         ylabel = "r*Pi(r, t)",
                         title = "r*Pi")
        ax_psi = GLM.Axis(fig[1, 3];
                          xlabel = "r",
                          ylabel = "r*Psi(r, t)",
                          title = "r*Psi")

        for (ax, limits) in ((ax_phi, phi_limits), (ax_pi, pi_limits), (ax_psi, psi_limits))
            GLM.hlines!(ax, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
            GLM.vlines!(ax, [0.0]; color = (:black, 0.35), linewidth = 1.0,
                        linestyle = :dot)
            GLM.xlims!(ax, minimum(common_r), maximum(common_r))
            GLM.ylims!(ax, limits[1], limits[2])
        end

        slider = GLM.Slider(fig[3, 1:3];
                            range = 1:nt,
                            startvalue = start_index,
                            snap = true)
        snapshot_index = slider.value

        time_label = GLM.lift(snapshot_index) do idx
            return "t = $(round(Float64(sampling.t[idx]); digits = 6))    snapshot = $(idx) / $(nt)"
        end
        GLM.Label(fig[2, 1:3], time_label; fontsize = 15)

        rphi_h_obs = GLM.lift(idx -> rphi_h[:, idx], snapshot_index)
        rphi_h2_obs = GLM.lift(idx -> rphi_h2[:, idx], snapshot_index)
        rphi_h4_obs = GLM.lift(idx -> rphi_h4[:, idx], snapshot_index)

        rpi_h_obs = GLM.lift(idx -> rpi_h[:, idx], snapshot_index)
        rpi_h2_obs = GLM.lift(idx -> rpi_h2[:, idx], snapshot_index)
        rpi_h4_obs = GLM.lift(idx -> rpi_h4[:, idx], snapshot_index)

        rpsi_h_obs = GLM.lift(idx -> rpsi_h[:, idx], snapshot_index)
        rpsi_h2_obs = GLM.lift(idx -> rpsi_h2[:, idx], snapshot_index)
        rpsi_h4_obs = GLM.lift(idx -> rpsi_h4[:, idx], snapshot_index)

        GLM.lines!(ax_phi, common_r, rphi_h_obs; color = :royalblue4, linewidth = 2.5,
                   linestyle = :solid, label = labels[1])
        GLM.lines!(ax_phi, common_r, rphi_h2_obs; color = :royalblue4, linewidth = 2.5,
                   linestyle = :dash, label = labels[2])
        GLM.lines!(ax_phi, common_r, rphi_h4_obs; color = :royalblue4, linewidth = 2.5,
                   linestyle = :dot, label = labels[3])
        GLM.axislegend(ax_phi; position = :rb)

        GLM.lines!(ax_pi, common_r, rpi_h_obs; color = :darkorange3, linewidth = 2.5,
                   linestyle = :solid, label = labels[1])
        GLM.lines!(ax_pi, common_r, rpi_h2_obs; color = :darkorange3, linewidth = 2.5,
                   linestyle = :dash, label = labels[2])
        GLM.lines!(ax_pi, common_r, rpi_h4_obs; color = :darkorange3, linewidth = 2.5,
                   linestyle = :dot, label = labels[3])
        GLM.axislegend(ax_pi; position = :rb)

        GLM.lines!(ax_psi, common_r, rpsi_h_obs; color = :seagreen4, linewidth = 2.5,
                   linestyle = :solid, label = labels[1])
        GLM.lines!(ax_psi, common_r, rpsi_h2_obs; color = :seagreen4, linewidth = 2.5,
                   linestyle = :dash, label = labels[2])
        GLM.lines!(ax_psi, common_r, rpsi_h4_obs; color = :seagreen4, linewidth = 2.5,
                   linestyle = :dot, label = labels[3])
        GLM.axislegend(ax_psi; position = :rb)

        # Re-apply the physical domain after attaching all plotted observables.
        for ax in (ax_phi, ax_pi, ax_psi)
            GLM.xlims!(ax, minimum(common_r), maximum(common_r))
        end

        fig
    end

    screen = display_figure ? GLM.display(fig) : nothing
    return (fig = fig,
            screen = screen,
            t = sampling.t,
            r = common_r,
            rphi = (coarse = rphi_h, medium = rphi_h2, fine = rphi_h4),
            rpi = (coarse = rpi_h, medium = rpi_h2, fine = rpi_h4),
            rpsi = (coarse = rpsi_h, medium = rpsi_h2, fine = rpsi_h4))
end

function _construct_wave_operator(config)
    if config.family === :diagonal
        ops = SphericalSBPOperators.diagonal_spherical_operators(config.source;
                                                                 accuracy_order = config.accuracy_order,
                                                                 N = config.N,
                                                                 R = config.R,
                                                                 p = config.p,
                                                                 mode = config.mode,
                                                                 target_eltype = config.target_eltype)
        report = SphericalSBPOperators.validate(ops;
                                                max_monomial_degree = config.accuracy_order,
                                                verbose = false)
        return ops, report
    elseif config.family === :mixed_order_diagonal
        ops = SphericalSBPOperators.mixed_order_diagonal_spherical_operators(config.source;
                                                                             accuracy_order = config.accuracy_order,
                                                                             N = config.N,
                                                                             R = config.R,
                                                                             p = config.p,
                                                                             mode = config.mode,
                                                                             target_eltype = config.target_eltype)
        report = SphericalSBPOperators.validate(ops;
                                                max_monomial_degree = config.accuracy_order,
                                                verbose = false)
        return ops, report
    elseif config.family === :staggered
        ops = SphericalSBPOperators.staggered_spherical_operators(config.source;
                                                                  accuracy_order = config.accuracy_order,
                                                                  N = config.N,
                                                                  R = config.R,
                                                                  p = config.p,
                                                                  mode = config.mode,
                                                                  target_eltype = config.target_eltype)
        report = SphericalSBPOperators.validate_staggered(ops;
                                                          max_monomial_degree = config.accuracy_order,
                                                          verbose = false)
        return ops, report
    elseif config.family === :non_diagonal
        ops = SphericalSBPOperators.non_diagonal_spherical_operators(config.source;
                                                                     accuracy_order = config.accuracy_order,
                                                                     N = config.N,
                                                                     R = config.R,
                                                                     p = config.p,
                                                                     mode = config.mode,
                                                                     target_eltype = config.target_eltype)
        return ops, nothing
    elseif config.family === :non_diagonal_exp
        config.accuracy_order == 6 ||
            throw(ArgumentError("`family=:non_diagonal_exp` currently requires `accuracy_order = 6`."))
        ops = SphericalSBPOperators.non_diagonal_exp_spherical_operators(config.source;
                                                                         N = config.N,
                                                                         R = config.R,
                                                                         p = config.p,
                                                                         mode = config.mode,
                                                                         outer_boundary_closure_help = config.outer_boundary_closure_help,
                                                                         target_eltype = config.target_eltype)
        return ops, nothing
    end

    throw(ArgumentError("`family` must be :diagonal, :mixed_order_diagonal, :staggered, :non_diagonal, or :non_diagonal_exp."))
end

function _select_algorithm(alg_name::Symbol, boundary_condition::Symbol)
    if alg_name === :auto
        return boundary_condition in (:reflecting, :reflective) ? ImplicitMidpoint() :
               DP8()
    elseif alg_name === :dp8
        return DP8()
    elseif alg_name === :tsitpap8
        return TsitPap8()
    elseif alg_name === :rk4
        return RK4()
    elseif alg_name === :implicit_midpoint
        return ImplicitMidpoint()
    end

    throw(ArgumentError("`alg_name` must be :auto, :dp8, :tsitpap8, :rk4, or :implicit_midpoint."))
end

function _algorithm_tag(alg)
    if alg isa DP8
        return "dp8"
    elseif alg isa TsitPap8
        return "tsitpap8"
    elseif alg isa RK4
        return "rk4"
    elseif alg isa ImplicitMidpoint
        return "implicit_midpoint"
    end

    return lowercase(replace(string(nameof(typeof(alg))), r"[^A-Za-z0-9]+" => "_"))
end

function _eltype_tag(T::Type)
    if T === Float64
        return "float64"
    elseif T === Float64x2
        return "float64x2"
    end

    return lowercase(replace(string(T), r"[^A-Za-z0-9]+" => "_"))
end

function _initial_data_save_stem(initial_data_kind::Symbol)
    if initial_data_kind === :even_gaussian_pi_zero_psi
        return "gundlach_wave"
    elseif initial_data_kind === :even_gaussian_pi
        return "incoming_gaussian_wave"
    elseif initial_data_kind === :characteristic_incoming_bump
        return "characteristic_wave"
    elseif initial_data_kind === :regular_left_moving_spherical_gaussian
        return "regular_left_moving_spherical_gaussian_wave"
    end

    return "$(lowercase(replace(string(initial_data_kind), r"[^A-Za-z0-9]+" => "_")))_wave"
end

function _boundary_tag(boundary_condition::Symbol)
    boundary_condition in (:reflecting, :reflective, :dirichlet) && return "reflecting"
    boundary_condition in (:absorbing, :radiative, :none) && return "radiative"
    return lowercase(replace(string(boundary_condition), r"[^A-Za-z0-9]+" => "_"))
end

function _boundary_group(boundary_condition::Symbol)
    boundary_condition in (:reflecting, :reflective, :dirichlet) && return "reflective"
    return "radiative"
end

function _nice_cfl_candidates(raw_cfl::Real)
    raw = Float64(raw_cfl)
    raw > 0 || throw(ArgumentError("`raw_cfl` must be positive."))

    candidates = Float64[]
    log10_center = floor(Int, log10(raw))
    for k in (log10_center - 4):(log10_center + 2)
        scale = 10.0^k
        for mantissa in (1.0, 2.0, 2.5, 5.0)
            push!(candidates, mantissa * scale)
        end
    end

    log2_center = floor(Int, log2(raw))
    for k in (log2_center - 8):(log2_center + 2)
        push!(candidates, 2.0^k)
    end

    filtered = filter(x -> isfinite(x) && x > 0, candidates)
    unique_candidates = unique!(filtered)
    sort!(unique_candidates; rev = true)
    return unique_candidates
end

function _snap_down_to_nice_cfl(raw_cfl::Real)
    raw = Float64(raw_cfl)
    for candidate in _nice_cfl_candidates(raw)
        candidate <= raw && return candidate
    end
    return raw
end

_imaginary_axis_stability_limit(::RK4) = 2.8284271247461903
_imaginary_axis_stability_limit(::DP8) = 5.960359811702602
_imaginary_axis_stability_limit(::TsitPap8) = 0.118
_imaginary_axis_stability_limit(alg) = nothing

function _reflective_explicit_dt_limit(ops,
                                       alg;
                                       boundary_condition::Symbol,
                                       enforce_origin::Bool)
    imag_limit = _imaginary_axis_stability_limit(alg)
    imag_limit === nothing && return nothing
    bc_norm = SphericalSBPOperators._normalize_boundary_condition(boundary_condition)
    bc_norm in (:reflecting, :dirichlet) || return nothing

    A = SphericalSBPOperators._wave_linear_operator(ops;
                                                    boundary_condition = bc_norm,
                                                    enforce_origin = enforce_origin)
    eigvals = SphericalSBPOperators._high_precision_schur_values(A)
    max_abs_imag = maximum(abs.(Float64.(imag.(eigvals))))
    max_abs_imag > 0 || return nothing

    return (dt_limit = imag_limit / max_abs_imag,
            imag_limit = imag_limit,
            max_abs_imag = max_abs_imag)
end

function _resolve_auto_wave_dt(ops,
                               alg;
                               dt,
                               safety_factor::Real,
                               boundary_condition::Symbol,
                               enforce_origin::Bool,
                               snap_to_nice::Bool)
    if !isnothing(dt)
        dt_float = Float64(dt)
        return (dt = dt_float,
                raw_dt = dt_float,
                snapped = nothing,
                reflective_limit = nothing)
    end

    reflective_limit = _reflective_explicit_dt_limit(ops,
                                                     alg;
                                                     boundary_condition = boundary_condition,
                                                     enforce_origin = enforce_origin)
    raw_dt = if isnothing(reflective_limit)
        SphericalSBPOperators.estimate_wave_timestep(ops;
                                                     alg = alg,
                                                     safety_factor = safety_factor,
                                                     boundary_condition = boundary_condition,
                                                     enforce_origin = enforce_origin)
    else
        Float64(safety_factor) * reflective_limit.dt_limit
    end
    raw_dt = raw_dt > 0.5 ? 0.5 : raw_dt

    snapped = snap_to_nice ?
              _stable_snapped_dt(ops,
                                 alg,
                                 raw_dt;
                                 boundary_condition = boundary_condition,
                                 enforce_origin = enforce_origin) :
              nothing
    resolved_dt = isnothing(snapped) ? Float64(raw_dt) : snapped.dt
    return (dt = resolved_dt,
            raw_dt = Float64(raw_dt),
            snapped = snapped,
            reflective_limit = reflective_limit)
end

function _stable_snapped_dt(ops,
                            alg,
                            raw_dt::Real;
                            boundary_condition::Symbol,
                            enforce_origin::Bool)
    raw_dt_float = Float64(raw_dt)
    dr_min = _spacing_stats(ops.r).min
    raw_cfl = raw_dt_float / dr_min

    for candidate_cfl in _nice_cfl_candidates(raw_cfl)
        candidate_cfl <= raw_cfl || continue
        candidate_dt = candidate_cfl * dr_min
        stability = SphericalSBPOperators._check_provided_dt_stability(ops,
                                                                       candidate_dt,
                                                                       alg;
                                                                       boundary_condition = boundary_condition,
                                                                       enforce_origin = enforce_origin,
                                                                       stability_tol = 1.0e-6,
                                                                       throw_on_failure = false)
        stability.stable && return (dt = candidate_dt,
                cfl = candidate_cfl,
                stability = stability)
    end

    stability = SphericalSBPOperators._check_provided_dt_stability(ops,
                                                                   raw_dt_float,
                                                                   alg;
                                                                   boundary_condition = boundary_condition,
                                                                   enforce_origin = enforce_origin,
                                                                   stability_tol = 1.0e-6,
                                                                   throw_on_failure = false)
    stability.stable ||
        throw(ErrorException("Internal error: raw estimated dt failed the spectral stability check."))
    return (dt = raw_dt_float,
            cfl = raw_cfl,
            stability = stability)
end

function _resolve_convergence_dt(; family::Symbol,
                                 source,
                                 accuracy_order::Int,
                                 N::Int,
                                 R,
                                 p::Int,
                                 mode,
                                 outer_boundary_closure_help::Bool,
                                 target_eltype::Union{Nothing, Type},
                                 dt,
                                 safety_factor::Real,
                                 boundary_condition::Symbol,
                                 alg_name::Symbol,
                                 enforce_origin::Bool)
    if !isnothing(dt)
        return Float64(dt)
    end

    operator_config = (;
                       family = family,
                       source = source,
                       accuracy_order = accuracy_order,
                       N = N,
                       R = R,
                       p = p,
                       mode = mode,
                       outer_boundary_closure_help = outer_boundary_closure_help,
                       target_eltype = target_eltype)
    ops, _ = _construct_wave_operator(operator_config)
    alg = _select_algorithm(alg_name, boundary_condition)
    resolved = _resolve_auto_wave_dt(ops,
                                     alg;
                                     dt = dt,
                                     safety_factor = safety_factor,
                                     boundary_condition = boundary_condition,
                                     enforce_origin = enforce_origin,
                                     snap_to_nice = true)
    dr_min = _spacing_stats(ops.r).min
    raw_cfl = resolved.raw_dt / dr_min
    snapped_cfl = isnothing(resolved.snapped) ? raw_cfl : resolved.snapped.cfl
    snapped_dt = isnothing(resolved.snapped) ? resolved.dt : resolved.snapped.dt
    max_amp = isnothing(resolved.snapped) ? NaN :
              resolved.snapped.stability.max_amplification
    println("Auto coarse dt estimate = ", resolved.raw_dt,
            ", dr_min = ", dr_min,
            ", raw CFL = ", raw_cfl,
            ", snapped CFL = ", snapped_cfl,
            ", using dt = ", snapped_dt,
            ", max |R(dt*lambda)| = ", max_amp)
    if !isnothing(resolved.reflective_limit)
        println("  reflective explicit dt limit used: imag-axis radius = ",
                resolved.reflective_limit.imag_limit,
                ", max |Im(lambda)| = ",
                resolved.reflective_limit.max_abs_imag)
    end
    return resolved.dt
end

function _gundlach_save_path(accuracy_order::Int,
                             family::Symbol,
                             alg,
                             target_eltype::Type;
                             initial_data_kind::Symbol,
                             boundary_condition::Symbol,
                             outer_boundary_closure_help::Bool = true)
    family_tag = family === :non_diagonal ? "non_diagonal" :
                 family === :non_diagonal_exp ?
                 (outer_boundary_closure_help ? "non_diagonal_exp" :
                  "non_diagonal_exp_no_outer_boundary_help") :
                 string(family)
    boundary_group = _boundary_group(boundary_condition)
    alg_tag = _algorithm_tag(alg)
    eltype_tag = _eltype_tag(target_eltype)
    stem = _initial_data_save_stem(initial_data_kind)
    filename = "$(stem)_$(accuracy_order)th_order_$(family_tag)_$(alg_tag)_$(eltype_tag).jld2"
    return joinpath("data", "sims", boundary_group, filename)
end

function run_mixed_order_diagonal(; accuracy_order::Int = 4, p::Int = 2, dt = 0.01,
                                  save_every::Int = 1, T_final::Real = 20.0, N::Int = 25,
                                  R::Real = 25.0,
                                  target_eltype::Union{Nothing, Type} = DEFAULT_OPERATOR_CONFIG.target_eltype,
                                  boundary_condition::Symbol = DEFAULT_WAVE_CONFIG.boundary_condition,
                                  initial_data_kind::Symbol = DEFAULT_WAVE_CONFIG.initial_data_kind,
                                  enforce_origin::Bool = DEFAULT_WAVE_CONFIG.enforce_origin,
                                  check_provided_dt_stability::Bool = DEFAULT_WAVE_CONFIG.check_provided_dt_stability,
                                  incoming_amplitude::Real = DEFAULT_WAVE_CONFIG.incoming_amplitude,
                                  incoming_center::Real = DEFAULT_WAVE_CONFIG.incoming_center,
                                  incoming_radius::Real = DEFAULT_WAVE_CONFIG.incoming_radius,
                                  outgoing_amplitude::Real = DEFAULT_WAVE_CONFIG.outgoing_amplitude,
                                  gaussian_pi_amplitude::Real = DEFAULT_WAVE_CONFIG.gaussian_pi_amplitude,
                                  gaussian_pi_center::Real = DEFAULT_WAVE_CONFIG.gaussian_pi_center,
                                  gaussian_pi_width::Real = DEFAULT_WAVE_CONFIG.gaussian_pi_width,
                                  alg_name::Symbol = DEFAULT_WAVE_CONFIG.alg_name,
                                  include_h32::Bool = false)
    family = :mixed_order_diagonal
    dt_base = _resolve_convergence_dt(family = family,
                                      source = DEFAULT_OPERATOR_CONFIG.source,
                                      accuracy_order = accuracy_order,
                                      N = N,
                                      R = R,
                                      p = p,
                                      mode = DEFAULT_OPERATOR_CONFIG.mode,
                                      outer_boundary_closure_help = DEFAULT_OPERATOR_CONFIG.outer_boundary_closure_help,
                                      target_eltype = target_eltype,
                                      dt = dt,
                                      safety_factor = DEFAULT_WAVE_CONFIG.safety_factor,
                                      boundary_condition = boundary_condition,
                                      alg_name = alg_name,
                                      enforce_origin = enforce_origin)
    println("Running mixed-order diagonal SBP operators for accuracy order $accuracy_order...")
    println("================================================================================")
    println("Running baseline (h)")
    run_h = run_simulation(family = family,
                           accuracy_order = accuracy_order,
                           N = N,
                           R = R,
                           p = p,
                           target_eltype = target_eltype,
                           boundary_condition = boundary_condition,
                           initial_data_kind = initial_data_kind,
                           enforce_origin = enforce_origin,
                           incoming_amplitude = incoming_amplitude,
                           incoming_center = incoming_center,
                           incoming_radius = incoming_radius,
                           outgoing_amplitude = outgoing_amplitude,
                           gaussian_pi_amplitude = gaussian_pi_amplitude,
                           gaussian_pi_center = gaussian_pi_center,
                           gaussian_pi_width = gaussian_pi_width,
                           alg_name = alg_name,
                           check_provided_dt_stability = check_provided_dt_stability,
                           T_final = T_final, dt = dt_base,
                           save_every = save_every)
    println("Running h/2")
    run_h2 = run_simulation(family = family,
                            accuracy_order = accuracy_order,
                            N = N * 2,
                            R = R,
                            p = p,
                            target_eltype = target_eltype,
                            boundary_condition = boundary_condition,
                            initial_data_kind = initial_data_kind,
                            enforce_origin = enforce_origin,
                            incoming_amplitude = incoming_amplitude,
                            incoming_center = incoming_center,
                            incoming_radius = incoming_radius,
                            outgoing_amplitude = outgoing_amplitude,
                            gaussian_pi_amplitude = gaussian_pi_amplitude,
                            gaussian_pi_center = gaussian_pi_center,
                            gaussian_pi_width = gaussian_pi_width,
                            alg_name = alg_name,
                            check_provided_dt_stability = check_provided_dt_stability,
                            T_final = T_final, dt = dt_base / 2,
                            save_every = save_every * 2)
    println("Running h/4")
    run_h4 = run_simulation(family = family,
                            accuracy_order = accuracy_order,
                            N = N * 4,
                            R = R,
                            p = p,
                            target_eltype = target_eltype,
                            boundary_condition = boundary_condition,
                            initial_data_kind = initial_data_kind,
                            enforce_origin = enforce_origin,
                            incoming_amplitude = incoming_amplitude,
                            incoming_center = incoming_center,
                            incoming_radius = incoming_radius,
                            outgoing_amplitude = outgoing_amplitude,
                            gaussian_pi_amplitude = gaussian_pi_amplitude,
                            gaussian_pi_center = gaussian_pi_center,
                            gaussian_pi_width = gaussian_pi_width,
                            alg_name = alg_name,
                            check_provided_dt_stability = check_provided_dt_stability,
                            T_final = T_final, dt = dt_base / 4,
                            save_every = save_every * 4)
    println("Running h/8")
    run_h8 = run_simulation(family = family,
                            accuracy_order = accuracy_order,
                            N = N * 8,
                            R = R,
                            p = p,
                            target_eltype = target_eltype,
                            boundary_condition = boundary_condition,
                            initial_data_kind = initial_data_kind,
                            enforce_origin = enforce_origin,
                            incoming_amplitude = incoming_amplitude,
                            incoming_center = incoming_center,
                            incoming_radius = incoming_radius,
                            outgoing_amplitude = outgoing_amplitude,
                            gaussian_pi_amplitude = gaussian_pi_amplitude,
                            gaussian_pi_center = gaussian_pi_center,
                            gaussian_pi_width = gaussian_pi_width,
                            alg_name = alg_name,
                            check_provided_dt_stability = check_provided_dt_stability,
                            T_final = T_final, dt = dt_base / 8,
                            save_every = save_every * 8)
    println("Running h/16")
    run_h16 = run_simulation(family = family,
                             accuracy_order = accuracy_order,
                             N = N * 16,
                             R = R,
                             p = p,
                             target_eltype = target_eltype,
                             boundary_condition = boundary_condition,
                             initial_data_kind = initial_data_kind,
                             enforce_origin = enforce_origin,
                             incoming_amplitude = incoming_amplitude,
                             incoming_center = incoming_center,
                             incoming_radius = incoming_radius,
                             outgoing_amplitude = outgoing_amplitude,
                             gaussian_pi_amplitude = gaussian_pi_amplitude,
                             gaussian_pi_center = gaussian_pi_center,
                             gaussian_pi_width = gaussian_pi_width,
                             alg_name = alg_name,
                             check_provided_dt_stability = check_provided_dt_stability,
                             T_final = T_final, dt = dt_base / 16,
                             save_every = save_every * 16)
    run_h32 = nothing
    if include_h32
        println("Running h/32")
        run_h32 = try
            run_simulation(family = family,
                           accuracy_order = accuracy_order,
                           N = N * 32,
                           R = R,
                           p = p,
                           target_eltype = target_eltype,
                           boundary_condition = boundary_condition,
                           initial_data_kind = initial_data_kind,
                           enforce_origin = enforce_origin,
                           incoming_amplitude = incoming_amplitude,
                           incoming_center = incoming_center,
                           incoming_radius = incoming_radius,
                           outgoing_amplitude = outgoing_amplitude,
                           gaussian_pi_amplitude = gaussian_pi_amplitude,
                           gaussian_pi_center = gaussian_pi_center,
                           gaussian_pi_width = gaussian_pi_width,
                           alg_name = alg_name,
                           check_provided_dt_stability = check_provided_dt_stability,
                           T_final = T_final, dt = dt_base / 32,
                           save_every = save_every * 32)
        catch e
            println("Failed to run h/32 simulation: ", e)
            nothing
        end
    end
    println("Saving to file...")
    save_path = _gundlach_save_path(accuracy_order,
                                    family,
                                    run_h.alg,
                                    eltype(run_h.ops.r);
                                    initial_data_kind = run_h.initial_data.kind,
                                    boundary_condition = run_h.wave_config.boundary_condition,
                                    outer_boundary_closure_help = run_h.operator_config.outer_boundary_closure_help)
    mkpath(dirname(save_path))
    println("  path = ", save_path)
    runs = if isnothing(run_h32)
        (; run_h = run_h,
         run_h2 = run_h2,
         run_h4 = run_h4,
         run_h8 = run_h8,
         run_h16 = run_h16)
    else
        (; run_h = run_h,
         run_h2 = run_h2,
         run_h4 = run_h4,
         run_h8 = run_h8,
         run_h16 = run_h16,
         run_h32 = run_h32)
    end
    metadata = _bundle_metadata(save_path, runs)
    metadata_markdown = _bundle_metadata_markdown(save_path, runs)
    if isnothing(run_h32)
        jldsave(save_path;
                metadata = metadata,
                metadata_markdown = metadata_markdown,
                run_h = runs.run_h,
                run_h2 = runs.run_h2,
                run_h4 = runs.run_h4,
                run_h8 = runs.run_h8,
                run_h16 = runs.run_h16)
    else
        jldsave(save_path;
                metadata = metadata,
                metadata_markdown = metadata_markdown,
                run_h = runs.run_h,
                run_h2 = runs.run_h2,
                run_h4 = runs.run_h4,
                run_h8 = runs.run_h8,
                run_h16 = runs.run_h16,
                run_h32 = runs.run_h32)
    end
    println("  metadata = JLD2:/metadata")
    println("Done!")
end

function _build_characteristic_data(ops, wave_config)
    center = Float64(wave_config.incoming_center)
    radius = Float64(wave_config.incoming_radius)

    w_in0 = SphericalSBPOperators.bumpb_profile(ops.r;
                                                amplitude = wave_config.incoming_amplitude,
                                                center = center,
                                                radius = radius)
    w_out0 = zeros(Float64, length(ops.r))

    return w_in0, w_out0
end

function _build_even_gaussian_pi_data(ops, wave_config)
    amplitude = Float64(wave_config.gaussian_pi_amplitude)
    r0 = Float64(wave_config.gaussian_pi_center)
    d = Float64(wave_config.gaussian_pi_width)
    d > 0 || throw(ArgumentError("`gaussian_pi_width` must be positive."))

    r = Float64.(ops.r)
    gaussian_envelope = amplitude .* (exp.(-((r .- r0) .^ 2) ./ d^2) .+
                         exp.(-((r .+ r0) .^ 2) ./ d^2))
    w_in0 = 2.0 .* gaussian_envelope
    w_out0 = zeros(Float64, length(r))
    char_data = SphericalSBPOperators.characteristic_initial_data(r;
                                                                  w_in0 = w_in0,
                                                                  w_out0 = w_out0,
                                                                  enforce_origin = wave_config.enforce_origin,
                                                                  has_origin_node = iszero(first(r)))
    pi0 = copy(char_data.pi0)
    psi0 = copy(char_data.psi0)
    phi0 = zeros(Float64, length(r))

    return (phi0 = phi0,
            pi0 = pi0,
            psi0 = psi0,
            w_in0 = w_in0,
            w_out0 = w_out0,
            r0 = r0,
            d = d,
            amplitude = amplitude)
end

function _build_even_gaussian_pi_zero_psi_data(ops, wave_config)
    amplitude = Float64(wave_config.gaussian_pi_amplitude)
    r0 = Float64(wave_config.gaussian_pi_center)
    d = Float64(wave_config.gaussian_pi_width)
    d > 0 || throw(ArgumentError("`gaussian_pi_width` must be positive."))

    r = Float64.(ops.r)
    pi0 = amplitude .* (exp.(-((r .- r0) .^ 2) ./ d^2) .+
                        exp.(-((r .+ r0) .^ 2) ./ d^2))
    psi0 = zeros(Float64, length(r))
    phi0 = zeros(Float64, length(r))

    return (phi0 = phi0,
            pi0 = pi0,
            psi0 = psi0,
            r0 = r0,
            d = d,
            amplitude = amplitude)
end

function _build_regular_left_moving_spherical_gaussian_data(ops, wave_config)
    amplitude = Float64(wave_config.regular_gaussian_amplitude)
    d = Float64(wave_config.regular_gaussian_width)
    R = Float64(ops.R)
    rc = Float64(wave_config.regular_gaussian_center)
    d > 0 || throw(ArgumentError("`regular_gaussian_width` must be positive."))
    0.0 <= rc <= R ||
        throw(ArgumentError("`regular_gaussian_center` must lie in [0,R]."))

    exact = SphericalSBPOperators.regular_left_moving_spherical_gaussian_solution(0.0,
                                                                                  ops.r;
                                                                                  amplitude = amplitude,
                                                                                  R = R,
                                                                                  d = d,
                                                                                  rc = rc)

    return (phi0 = Float64.(exact.Φ),
            pi0 = Float64.(exact.Π),
            psi0 = Float64.(exact.Ψ),
            amplitude = amplitude,
            d = d,
            R = R,
            rc = rc)
end

function _build_initial_data(ops, wave_config)
    kind = wave_config.initial_data_kind

    if kind === :characteristic_incoming_bump
        w_in0, w_out0 = _build_characteristic_data(ops, wave_config)
        return (kind = kind,
                solver_kwargs = (initial_data_mode = :characteristic,
                                 w_in0 = w_in0,
                                 w_out0 = w_out0),
                summary = (w_in0 = w_in0,
                           w_out0 = w_out0))
    elseif kind === :even_gaussian_pi
        gaussian_data = _build_even_gaussian_pi_data(ops, wave_config)
        return (kind = kind,
                solver_kwargs = (initial_data_mode = :characteristic,
                                 w_in0 = gaussian_data.w_in0,
                                 w_out0 = gaussian_data.w_out0),
                summary = gaussian_data)
    elseif kind === :even_gaussian_pi_zero_psi
        gaussian_data = _build_even_gaussian_pi_zero_psi_data(ops, wave_config)
        return (kind = kind,
                solver_kwargs = (phi0 = gaussian_data.phi0,
                                 pi0 = gaussian_data.pi0,
                                 psi0 = gaussian_data.psi0),
                summary = gaussian_data)
    elseif kind === :regular_left_moving_spherical_gaussian
        gaussian_data = _build_regular_left_moving_spherical_gaussian_data(ops, wave_config)
        return (kind = kind,
                solver_kwargs = (phi0 = gaussian_data.phi0,
                                 pi0 = gaussian_data.pi0,
                                 psi0 = gaussian_data.psi0),
                summary = gaussian_data)
    end

    throw(ArgumentError("`initial_data_kind` must be :characteristic_incoming_bump, :even_gaussian_pi, :even_gaussian_pi_zero_psi, or :regular_left_moving_spherical_gaussian."))
end

function _spacing_stats(r::AbstractVector)
    length(r) >= 2 ||
        throw(ArgumentError("Need at least two grid points to estimate spacing."))
    dr = diff(Float64.(r))
    return (min = minimum(dr),
            max = maximum(dr),
            mean = sum(dr) / length(dr))
end

_format_diag_value(x::Integer) = string(x)
_format_diag_value(x::Symbol) = string(x)
_format_diag_value(x::Bool) = x ? "yes" : "no"
_format_diag_value(x::AbstractString) = x
_format_diag_value(x::Type) = string(x)
_format_diag_value(x) = @sprintf("%.6e", Float64(x))

function _format_diagnostics_table(title::AbstractString,
                                   rows::Vector{Tuple{String, String}})
    key_width = maximum(length(first(row)) for row in rows)
    value_width = maximum(length(last(row)) for row in rows)
    inner_width = key_width + value_width + 7
    border = "+" * repeat("-", inner_width) * "+"
    title_pad = max(0, inner_width - length(title) - 2)
    left_pad = fld(title_pad, 2)
    right_pad = title_pad - left_pad
    title_row = "|" * repeat(" ", left_pad + 1) * title * repeat(" ", right_pad + 1) * "|"

    io = IOBuffer()
    println(io, border)
    println(io, title_row)
    println(io, border)
    for (key, value) in rows
        println(io,
                "| ",
                rpad(key, key_width),
                " | ",
                rpad(value, value_width),
                " |")
    end
    print(io, border)
    return String(take!(io))
end

function _collect_run_diagnostics(family::Symbol,
                                  ops,
                                  validation_report,
                                  sol,
                                  dt_requested,
                                  alg,
                                  initial_data_kind::Symbol)
    spacing = _spacing_stats(ops.r)
    cfl = sol.dt / spacing.min
    domain = (left = 0.0, right = Float64(ops.R))
    grid_support = (left = Float64(first(ops.r)), right = Float64(last(ops.r)))
    energy_ratio = sol.energy[end] / sol.energy[1]

    return (family = family,
            operator_type = string(typeof(ops)),
            accuracy_order = ops.accuracy_order,
            p = ops.p,
            domain_left = domain.left,
            domain_right = domain.right,
            grid_left = grid_support.left,
            grid_right = grid_support.right,
            has_origin_node = iszero(first(ops.r)),
            Nh = ops.Nh,
            M_full = ops.M_full,
            closure_width = ops.closure_width,
            dt_requested = Float64(dt_requested),
            dt_effective = Float64(sol.dt),
            T_final = Float64(last(sol.t)),
            nsteps = sol.nsteps,
            save_count = length(sol.t),
            dr_min = spacing.min,
            dr_max = spacing.max,
            dr_mean = spacing.mean,
            cfl = cfl,
            integrator = string(typeof(alg)),
            initial_data_kind = initial_data_kind,
            boundary_condition = sol.boundary_condition,
            initial_energy = Float64(sol.energy[1]),
            final_energy = Float64(sol.energy[end]),
            energy_ratio = energy_ratio,
            sbp_no_origin = validation_report === nothing ? nothing :
                            validation_report.sbp.sbp_no_origin)
end

_metadata_format_value(x::Nothing) = "nothing"
_metadata_format_value(x::Bool) = x ? "true" : "false"
_metadata_format_value(x::Symbol) = string(x)
_metadata_format_value(x::Integer) = string(x)
_metadata_format_value(x::AbstractString) = x
_metadata_format_value(x::Type) = string(x)
_metadata_format_value(x::AbstractFloat) = @sprintf("%.12g", Float64(x))
_metadata_format_value(x::Rational) = string(x)
_metadata_format_value(x) = string(x)

function _metadata_compact_value(value)
    if value isa AbstractVector
        return _metadata_vector_summary(value)
    elseif value isa Union{Nothing, Bool, Symbol, Integer, AbstractString, Type,
                 AbstractFloat, Rational}
        return value
    end
    return string(nameof(typeof(value)))
end

function _metadata_vector_summary(values::AbstractVector)
    isempty(values) && return "length=0"
    vals = Float64.(values)
    return "length=$(length(values)), min=$(@sprintf("%.12g", minimum(vals))), max=$(@sprintf("%.12g", maximum(vals)))"
end

function _metadata_compact_namedtuple(nt; skip_keys = Symbol[])
    pairs = Pair{Symbol, Any}[]
    for key in propertynames(nt)
        key in skip_keys && continue
        push!(pairs, key => _metadata_compact_value(getproperty(nt, key)))
    end
    return (; pairs...)
end

function _metadata_lines_from_namedtuple(nt; skip_keys = Symbol[])
    lines = String[]
    for key in propertynames(nt)
        key in skip_keys && continue
        value = getproperty(nt, key)
        if value isa AbstractVector
            push!(lines, "- `$(key)`: $(_metadata_vector_summary(value))")
        elseif value isa Union{Nothing, Bool, Symbol, Integer, AbstractString, Type,
                     AbstractFloat, Rational}
            push!(lines, "- `$(key)`: $(_metadata_format_value(value))")
        else
            push!(lines,
                  "- `$(key)`: $(_metadata_format_value(nameof(typeof(value))))")
        end
    end
    return lines
end

function _metadata_shared_lines(run)
    op = run.operator_config
    wave = run.wave_config
    init_summary = run.initial_data.summary

    lines = String["- `family`: $(_metadata_format_value(op.family))",
                   "- `source`: $(_metadata_format_value(typeof(op.source)))",
                   "- `mode`: $(_metadata_format_value(typeof(op.mode)))",
                   "- `accuracy_order`: $(_metadata_format_value(op.accuracy_order))",
                   "- `R`: $(_metadata_format_value(op.R))",
                   "- `p`: $(_metadata_format_value(op.p))",
                   "- `target_eltype`: $(_metadata_format_value(op.target_eltype))",
                   "- `T_final`: $(_metadata_format_value(wave.T_final))",
                   "- `boundary_condition`: $(_metadata_format_value(wave.boundary_condition))",
                   "- `alg_name`: $(_metadata_format_value(wave.alg_name))",
                   "- `safety_factor`: $(_metadata_format_value(wave.safety_factor))",
                   "- `check_provided_dt_stability`: $(_metadata_format_value(wave.check_provided_dt_stability))",
                   "- `enforce_origin`: $(_metadata_format_value(wave.enforce_origin))",
                   "- `initial_data_kind`: $(_metadata_format_value(wave.initial_data_kind))",
                   "- `incoming_amplitude`: $(_metadata_format_value(wave.incoming_amplitude))",
                   "- `incoming_center`: $(_metadata_format_value(wave.incoming_center))",
                   "- `incoming_radius`: $(_metadata_format_value(wave.incoming_radius))",
                   "- `outgoing_amplitude`: $(_metadata_format_value(wave.outgoing_amplitude))",
                   "- `gaussian_pi_amplitude`: $(_metadata_format_value(wave.gaussian_pi_amplitude))",
                   "- `gaussian_pi_center`: $(_metadata_format_value(wave.gaussian_pi_center))",
                   "- `gaussian_pi_width`: $(_metadata_format_value(wave.gaussian_pi_width))",
                   "- `regular_gaussian_amplitude`: $(_metadata_format_value(wave.regular_gaussian_amplitude))",
                   "- `regular_gaussian_width`: $(_metadata_format_value(wave.regular_gaussian_width))",
                   "- `regular_gaussian_center`: $(_metadata_format_value(wave.regular_gaussian_center))",
                   "- `verbose`: $(_metadata_format_value(wave.verbose))",
                   "- `initial_data_summary.kind`: $(_metadata_format_value(run.initial_data.kind))"]

    for key in propertynames(init_summary)
        value = getproperty(init_summary, key)
        if value isa AbstractVector
            push!(lines,
                  "- `initial_data_summary.$(key)`: $(_metadata_vector_summary(value))")
        else
            push!(lines,
                  "- `initial_data_summary.$(key)`: $(_metadata_format_value(value))")
        end
    end

    return lines
end

function _bundle_metadata(save_path::AbstractString, runs::NamedTuple)
    created_at = format(now(), "yyyy-mm-dd HH:MM:SS")
    run_labels = Tuple(keys(runs))
    base_run = runs[first(run_labels)]

    run_entries = (;
                   (label => begin
                        run = runs[label]
                        validation_summary = if run.validation_report === nothing
                            nothing
                        elseif hasproperty(run.validation_report, :sbp)
                            _metadata_compact_namedtuple(getproperty(run.validation_report,
                                                                     :sbp))
                        else
                            "saved_in_bundle"
                        end
                        (resolution_label = label,
                         operator_config = _metadata_compact_namedtuple(run.operator_config),
                         wave_config = _metadata_compact_namedtuple(run.wave_config),
                         initial_data = (kind = run.initial_data.kind,
                                         summary = _metadata_compact_namedtuple(run.initial_data.summary)),
                         realized = (alg_type = string(typeof(run.alg)),
                                     ops_type = string(typeof(run.ops)),
                                     diagnostics = _metadata_compact_namedtuple(run.diagnostics),
                                     saved_payload = (t = _metadata_vector_summary(run.sol.t),
                                                      r = _metadata_vector_summary(run.sol.r),
                                                      energy = _metadata_vector_summary(run.sol.energy)),
                                     validation_report_sbp = validation_summary))
                    end for label in run_labels)...)

    return (schema_version = 2,
            data_file = basename(save_path),
            created_at_local = created_at,
            boundary_group = Symbol(_boundary_group(base_run.wave_config.boundary_condition)),
            family = base_run.operator_config.family,
            accuracy_order = base_run.operator_config.accuracy_order,
            initial_data_kind = base_run.initial_data.kind,
            initial_data_stem = _initial_data_save_stem(base_run.initial_data.kind),
            resolution_labels = run_labels,
            shared = (operator_config = _metadata_compact_namedtuple(base_run.operator_config),
                      wave_config = _metadata_compact_namedtuple(base_run.wave_config),
                      initial_data = (kind = base_run.initial_data.kind,
                                      summary = _metadata_compact_namedtuple(base_run.initial_data.summary))),
            runs = run_entries)
end

function _bundle_metadata_markdown(save_path::AbstractString, runs::NamedTuple)
    created_at = format(now(), "yyyy-mm-dd HH:MM:SS")
    run_labels = collect(keys(runs))
    base_run = runs[run_labels[1]]
    basename_path = basename(save_path)

    lines = String["# Simulation Metadata",
                   "",
                   "- `data_file`: `$(basename_path)`",
                   "- `created_at_local`: `$(created_at)`",
                   "- `resolution_labels`: `$(join(string.(run_labels), ", "))`",
                   "",
                   "## Shared Choices"]
    append!(lines, _metadata_shared_lines(base_run))

    for label in run_labels
        run = runs[label]
        push!(lines, "")
        push!(lines, "## $(label)")
        push!(lines, "")
        push!(lines, "### Requested Configuration")
        append!(lines, _metadata_lines_from_namedtuple(run.operator_config))
        append!(lines, _metadata_lines_from_namedtuple(run.wave_config))

        push!(lines, "")
        push!(lines, "### Realized Run")
        push!(lines, "- `alg_type`: $(_metadata_format_value(typeof(run.alg)))")
        push!(lines, "- `ops_type`: $(_metadata_format_value(typeof(run.ops)))")
        append!(lines, _metadata_lines_from_namedtuple(run.diagnostics))

        push!(lines, "")
        push!(lines, "### Saved Payload")
        push!(lines, "- `sol.t`: $(_metadata_vector_summary(run.sol.t))")
        push!(lines, "- `sol.r`: $(_metadata_vector_summary(run.sol.r))")
        push!(lines, "- `sol.energy`: $(_metadata_vector_summary(run.sol.energy))")

        if run.validation_report !== nothing
            push!(lines, "")
            push!(lines, "### Validation Report")
            if hasproperty(run.validation_report, :sbp)
                sbp = getproperty(run.validation_report, :sbp)
                for key in propertynames(sbp)
                    value = getproperty(sbp, key)
                    push!(lines,
                          "- `validation_report.sbp.$(key)`: $(_metadata_format_value(value))")
                end
            else
                push!(lines, "- validation report saved in JLD2 bundle")
            end
        end
    end

    push!(lines, "")
    return join(lines, "\n")
end

function _metadata_sidecar_path(save_path::AbstractString)
    return replace(save_path, r"\.jld2$" => ".metadata.md")
end

function _write_metadata_sidecar(save_path::AbstractString, runs::NamedTuple)
    metadata_path = _metadata_sidecar_path(save_path)
    open(metadata_path, "w") do io
        write(io, _bundle_metadata_markdown(save_path, runs))
    end
    println("  metadata = ", metadata_path)
    return metadata_path
end

function _print_run_summary(diagnostics)
    rows = Tuple{String, String}[("family", _format_diag_value(diagnostics.family)),
                                 ("operator", diagnostics.operator_type),
                                 ("accuracy_order",
                                  _format_diag_value(diagnostics.accuracy_order)),
                                 ("p", _format_diag_value(diagnostics.p)),
                                 ("domain",
                                  "[$(_format_diag_value(diagnostics.domain_left)), $(_format_diag_value(diagnostics.domain_right))]"),
                                 ("grid_support",
                                  "[$(_format_diag_value(diagnostics.grid_left)), $(_format_diag_value(diagnostics.grid_right))]"),
                                 ("has_origin_node",
                                  _format_diag_value(diagnostics.has_origin_node)),
                                 ("Nh", _format_diag_value(diagnostics.Nh)),
                                 ("M_full", _format_diag_value(diagnostics.M_full)),
                                 ("closure_width",
                                  _format_diag_value(diagnostics.closure_width)),
                                 ("integrator", diagnostics.integrator),
                                 ("initial_data_kind",
                                  _format_diag_value(diagnostics.initial_data_kind)),
                                 ("boundary_condition",
                                  _format_diag_value(diagnostics.boundary_condition)),
                                 ("T_final", _format_diag_value(diagnostics.T_final)),
                                 ("dt_requested",
                                  _format_diag_value(diagnostics.dt_requested)),
                                 ("dt_effective",
                                  _format_diag_value(diagnostics.dt_effective)),
                                 ("nsteps", _format_diag_value(diagnostics.nsteps)),
                                 ("saved_snapshots",
                                  _format_diag_value(diagnostics.save_count)),
                                 ("dr_min", _format_diag_value(diagnostics.dr_min)),
                                 ("dr_max", _format_diag_value(diagnostics.dr_max)),
                                 ("dr_mean", _format_diag_value(diagnostics.dr_mean)),
                                 ("CFL = dt/dr_min", _format_diag_value(diagnostics.cfl)),
                                 ("initial_energy",
                                  _format_diag_value(diagnostics.initial_energy)),
                                 ("final_energy",
                                  _format_diag_value(diagnostics.final_energy)),
                                 ("final/initial_energy",
                                  _format_diag_value(diagnostics.energy_ratio))]

    if diagnostics.sbp_no_origin !== nothing
        push!(rows, ("sbp_no_origin", _format_diag_value(diagnostics.sbp_no_origin)))
    end

    println(_format_diagnostics_table("Wave Solve Diagnostics", rows))
end

function run_simulation(;
                        family::Symbol = DEFAULT_OPERATOR_CONFIG.family,
                        source = DEFAULT_OPERATOR_CONFIG.source,
                        accuracy_order::Int = DEFAULT_OPERATOR_CONFIG.accuracy_order,
                        N::Int = DEFAULT_OPERATOR_CONFIG.N,
                        R = DEFAULT_OPERATOR_CONFIG.R,
                        p::Int = DEFAULT_OPERATOR_CONFIG.p,
                        mode = DEFAULT_OPERATOR_CONFIG.mode,
                        outer_boundary_closure_help::Bool = DEFAULT_OPERATOR_CONFIG.outer_boundary_closure_help,
                        target_eltype::Union{Nothing, Type} = DEFAULT_OPERATOR_CONFIG.target_eltype,
                        T_final::Real = DEFAULT_WAVE_CONFIG.T_final,
                        dt = DEFAULT_WAVE_CONFIG.dt,
                        safety_factor::Real = DEFAULT_WAVE_CONFIG.safety_factor,
                        boundary_condition::Symbol = DEFAULT_WAVE_CONFIG.boundary_condition,
                        alg_name::Symbol = DEFAULT_WAVE_CONFIG.alg_name,
                        save_every::Int = DEFAULT_WAVE_CONFIG.save_every,
                        check_provided_dt_stability::Bool = DEFAULT_WAVE_CONFIG.check_provided_dt_stability,
                        enforce_origin::Bool = DEFAULT_WAVE_CONFIG.enforce_origin,
                        initial_data_kind::Symbol = DEFAULT_WAVE_CONFIG.initial_data_kind,
                        incoming_amplitude::Real = DEFAULT_WAVE_CONFIG.incoming_amplitude,
                        incoming_center::Real = DEFAULT_WAVE_CONFIG.incoming_center,
                        incoming_radius::Real = DEFAULT_WAVE_CONFIG.incoming_radius,
                        outgoing_amplitude::Real = DEFAULT_WAVE_CONFIG.outgoing_amplitude,
                        gaussian_pi_amplitude::Real = DEFAULT_WAVE_CONFIG.gaussian_pi_amplitude,
                        gaussian_pi_center::Real = DEFAULT_WAVE_CONFIG.gaussian_pi_center,
                        gaussian_pi_width::Real = DEFAULT_WAVE_CONFIG.gaussian_pi_width,
                        regular_gaussian_amplitude::Real = DEFAULT_WAVE_CONFIG.regular_gaussian_amplitude,
                        regular_gaussian_width::Real = DEFAULT_WAVE_CONFIG.regular_gaussian_width,
                        regular_gaussian_center::Real = DEFAULT_WAVE_CONFIG.regular_gaussian_center,
                        verbose::Bool = DEFAULT_WAVE_CONFIG.verbose)
    operator_config = (;
                       family = family,
                       source = source,
                       accuracy_order = accuracy_order,
                       N = N,
                       R = R,
                       p = p,
                       mode = mode,
                       outer_boundary_closure_help = outer_boundary_closure_help,
                       target_eltype = target_eltype)
    wave_config = (;
                   T_final = T_final,
                   dt = dt,
                   safety_factor = safety_factor,
                   boundary_condition = boundary_condition,
                   alg_name = alg_name,
                   save_every = save_every,
                   check_provided_dt_stability = check_provided_dt_stability,
                   enforce_origin = enforce_origin,
                   initial_data_kind = initial_data_kind,
                   incoming_amplitude = incoming_amplitude,
                   incoming_center = incoming_center,
                   incoming_radius = incoming_radius,
                   outgoing_amplitude = outgoing_amplitude,
                   gaussian_pi_amplitude = gaussian_pi_amplitude,
                   gaussian_pi_center = gaussian_pi_center,
                   gaussian_pi_width = gaussian_pi_width,
                   regular_gaussian_amplitude = regular_gaussian_amplitude,
                   regular_gaussian_width = regular_gaussian_width,
                   regular_gaussian_center = regular_gaussian_center,
                   verbose = verbose)

    ops, validation_report = _construct_wave_operator(operator_config)
    alg = _select_algorithm(wave_config.alg_name, wave_config.boundary_condition)
    initial_data = _build_initial_data(ops, wave_config)

    resolved_dt = _resolve_auto_wave_dt(ops,
                                        alg;
                                        dt = wave_config.dt,
                                        safety_factor = wave_config.safety_factor,
                                        boundary_condition = wave_config.boundary_condition,
                                        enforce_origin = wave_config.enforce_origin,
                                        snap_to_nice = true)
    dt_requested = resolved_dt.dt

    operator_backend = operator_config.family === :non_diagonal_exp ? :sparse :
                       :kernel
    sol = SphericalSBPOperators.solve_wave_ode(ops;
                                               operator_backend = operator_backend,
                                               T_final = wave_config.T_final,
                                               dt = resolved_dt.dt,
                                               alg = alg,
                                               safety_factor = wave_config.safety_factor,
                                               boundary_condition = wave_config.boundary_condition,
                                               initial_data.solver_kwargs...,
                                               save_every = wave_config.save_every,
                                               check_provided_dt_stability = wave_config.check_provided_dt_stability,
                                               enforce_origin = wave_config.enforce_origin,
                                               verbose = wave_config.verbose)

    diagnostics = _collect_run_diagnostics(operator_config.family,
                                           ops,
                                           validation_report,
                                           sol,
                                           dt_requested,
                                           alg,
                                           initial_data.kind)

    return (ops = ops,
            validation_report = validation_report,
            operator_config = operator_config,
            wave_config = wave_config,
            initial_data = initial_data,
            w_in0 = get(initial_data.summary, :w_in0, nothing),
            w_out0 = get(initial_data.summary, :w_out0, nothing),
            sol = sol,
            alg = alg,
            diagnostics = diagnostics)
end

function run_diagonal(; accuracy_order::Int = 4, p::Int = 2, dt = 0.01,
                      save_every::Int = 1, T_final::Real = 20.0, N::Int = 25,
                      R::Real = 25.0,
                      target_eltype::Union{Nothing, Type} = DEFAULT_OPERATOR_CONFIG.target_eltype,
                      boundary_condition::Symbol = DEFAULT_WAVE_CONFIG.boundary_condition,
                      initial_data_kind::Symbol = DEFAULT_WAVE_CONFIG.initial_data_kind,
                      enforce_origin::Bool = DEFAULT_WAVE_CONFIG.enforce_origin,
                      check_provided_dt_stability::Bool = DEFAULT_WAVE_CONFIG.check_provided_dt_stability,
                      incoming_amplitude::Real = DEFAULT_WAVE_CONFIG.incoming_amplitude,
                      incoming_center::Real = DEFAULT_WAVE_CONFIG.incoming_center,
                      incoming_radius::Real = DEFAULT_WAVE_CONFIG.incoming_radius,
                      outgoing_amplitude::Real = DEFAULT_WAVE_CONFIG.outgoing_amplitude,
                      gaussian_pi_amplitude::Real = DEFAULT_WAVE_CONFIG.gaussian_pi_amplitude,
                      gaussian_pi_center::Real = DEFAULT_WAVE_CONFIG.gaussian_pi_center,
                      gaussian_pi_width::Real = DEFAULT_WAVE_CONFIG.gaussian_pi_width,
                      regular_gaussian_amplitude::Real = DEFAULT_WAVE_CONFIG.regular_gaussian_amplitude,
                      regular_gaussian_width::Real = DEFAULT_WAVE_CONFIG.regular_gaussian_width,
                      regular_gaussian_center::Real = DEFAULT_WAVE_CONFIG.regular_gaussian_center,
                      alg_name::Symbol = DEFAULT_WAVE_CONFIG.alg_name,
                      include_h32::Bool = false)
    dt_base = _resolve_convergence_dt(family = :diagonal,
                                      source = DEFAULT_OPERATOR_CONFIG.source,
                                      accuracy_order = accuracy_order,
                                      N = N,
                                      R = R,
                                      p = p,
                                      mode = DEFAULT_OPERATOR_CONFIG.mode,
                                      outer_boundary_closure_help = DEFAULT_OPERATOR_CONFIG.outer_boundary_closure_help,
                                      target_eltype = target_eltype,
                                      dt = dt,
                                      safety_factor = DEFAULT_WAVE_CONFIG.safety_factor,
                                      boundary_condition = boundary_condition,
                                      alg_name = alg_name,
                                      enforce_origin = enforce_origin)
    println("Running diagonal SBP operators for accuracy order $accuracy_order...")
    println("================================================================================")
    println("Running baseline (h)")
    run_h = run_simulation(family = :diagonal,
                           accuracy_order = accuracy_order,
                           N = N,
                           R = R,
                           p = p,
                           target_eltype = target_eltype,
                           boundary_condition = boundary_condition,
                           initial_data_kind = initial_data_kind,
                           enforce_origin = enforce_origin,
                           incoming_amplitude = incoming_amplitude,
                           incoming_center = incoming_center,
                           incoming_radius = incoming_radius,
                           outgoing_amplitude = outgoing_amplitude,
                           gaussian_pi_amplitude = gaussian_pi_amplitude,
                           gaussian_pi_center = gaussian_pi_center,
                           gaussian_pi_width = gaussian_pi_width,
                           regular_gaussian_amplitude = regular_gaussian_amplitude,
                           regular_gaussian_width = regular_gaussian_width,
                           regular_gaussian_center = regular_gaussian_center,
                           alg_name = alg_name,
                           check_provided_dt_stability = check_provided_dt_stability,
                           T_final = T_final, dt = dt_base,
                           save_every = save_every)
    println("Running h/2")
    run_h2 = run_simulation(family = :diagonal,
                            accuracy_order = accuracy_order,
                            N = N * 2,
                            R = R,
                            p = p,
                            target_eltype = target_eltype,
                            boundary_condition = boundary_condition,
                            initial_data_kind = initial_data_kind,
                            enforce_origin = enforce_origin,
                            incoming_amplitude = incoming_amplitude,
                            incoming_center = incoming_center,
                            incoming_radius = incoming_radius,
                            outgoing_amplitude = outgoing_amplitude,
                            gaussian_pi_amplitude = gaussian_pi_amplitude,
                            gaussian_pi_center = gaussian_pi_center,
                            gaussian_pi_width = gaussian_pi_width,
                            regular_gaussian_amplitude = regular_gaussian_amplitude,
                            regular_gaussian_width = regular_gaussian_width,
                            regular_gaussian_center = regular_gaussian_center,
                            alg_name = alg_name,
                            check_provided_dt_stability = check_provided_dt_stability,
                            T_final = T_final, dt = dt_base / 2,
                            save_every = save_every * 2)
    println("Running h/4")
    run_h4 = run_simulation(family = :diagonal,
                            accuracy_order = accuracy_order,
                            N = N * 4,
                            R = R,
                            p = p,
                            target_eltype = target_eltype,
                            boundary_condition = boundary_condition,
                            initial_data_kind = initial_data_kind,
                            enforce_origin = enforce_origin,
                            incoming_amplitude = incoming_amplitude,
                            incoming_center = incoming_center,
                            incoming_radius = incoming_radius,
                            outgoing_amplitude = outgoing_amplitude,
                            gaussian_pi_amplitude = gaussian_pi_amplitude,
                            gaussian_pi_center = gaussian_pi_center,
                            gaussian_pi_width = gaussian_pi_width,
                            regular_gaussian_amplitude = regular_gaussian_amplitude,
                            regular_gaussian_width = regular_gaussian_width,
                            regular_gaussian_center = regular_gaussian_center,
                            alg_name = alg_name,
                            check_provided_dt_stability = check_provided_dt_stability,
                            T_final = T_final, dt = dt_base / 4,
                            save_every = save_every * 4)
    println("Running h/8")
    run_h8 = run_simulation(family = :diagonal,
                            accuracy_order = accuracy_order,
                            N = N * 8,
                            R = R,
                            p = p,
                            target_eltype = target_eltype,
                            boundary_condition = boundary_condition,
                            initial_data_kind = initial_data_kind,
                            enforce_origin = enforce_origin,
                            incoming_amplitude = incoming_amplitude,
                            incoming_center = incoming_center,
                            incoming_radius = incoming_radius,
                            outgoing_amplitude = outgoing_amplitude,
                            gaussian_pi_amplitude = gaussian_pi_amplitude,
                            gaussian_pi_center = gaussian_pi_center,
                            gaussian_pi_width = gaussian_pi_width,
                            regular_gaussian_amplitude = regular_gaussian_amplitude,
                            regular_gaussian_width = regular_gaussian_width,
                            regular_gaussian_center = regular_gaussian_center,
                            alg_name = alg_name,
                            check_provided_dt_stability = check_provided_dt_stability,
                            T_final = T_final, dt = dt_base / 8,
                            save_every = save_every * 8)
    println("Running h/16")
    run_h16 = run_simulation(family = :diagonal,
                             accuracy_order = accuracy_order,
                             N = N * 16,
                             R = R,
                             p = p,
                             target_eltype = target_eltype,
                             boundary_condition = boundary_condition,
                             initial_data_kind = initial_data_kind,
                             enforce_origin = enforce_origin,
                             incoming_amplitude = incoming_amplitude,
                             incoming_center = incoming_center,
                             incoming_radius = incoming_radius,
                             outgoing_amplitude = outgoing_amplitude,
                             gaussian_pi_amplitude = gaussian_pi_amplitude,
                             gaussian_pi_center = gaussian_pi_center,
                             gaussian_pi_width = gaussian_pi_width,
                             regular_gaussian_amplitude = regular_gaussian_amplitude,
                             regular_gaussian_width = regular_gaussian_width,
                             regular_gaussian_center = regular_gaussian_center,
                             alg_name = alg_name,
                             check_provided_dt_stability = check_provided_dt_stability,
                             T_final = T_final, dt = dt_base / 16,
                             save_every = save_every * 16)
    run_h32 = nothing
    if include_h32
        println("Running h/32")
        run_h32 = try
            run_simulation(family = :diagonal,
                           accuracy_order = accuracy_order,
                           N = N * 32,
                           R = R,
                           p = p,
                           target_eltype = target_eltype,
                           boundary_condition = boundary_condition,
                           initial_data_kind = initial_data_kind,
                           enforce_origin = enforce_origin,
                           incoming_amplitude = incoming_amplitude,
                           incoming_center = incoming_center,
                           incoming_radius = incoming_radius,
                           outgoing_amplitude = outgoing_amplitude,
                           gaussian_pi_amplitude = gaussian_pi_amplitude,
                           gaussian_pi_center = gaussian_pi_center,
                           gaussian_pi_width = gaussian_pi_width,
                           regular_gaussian_amplitude = regular_gaussian_amplitude,
                           regular_gaussian_width = regular_gaussian_width,
                           regular_gaussian_center = regular_gaussian_center,
                           alg_name = alg_name,
                           check_provided_dt_stability = check_provided_dt_stability,
                           T_final = T_final, dt = dt_base / 32,
                           save_every = save_every * 32)
        catch e
            println("Failed to run h/32 simulation: ", e)
            nothing
        end
    end
    println("Saving to file...")
    save_path = _gundlach_save_path(accuracy_order,
                                    :diagonal,
                                    run_h.alg,
                                    eltype(run_h.ops.r);
                                    initial_data_kind = run_h.initial_data.kind,
                                    boundary_condition = run_h.wave_config.boundary_condition,
                                    outer_boundary_closure_help = run_h.operator_config.outer_boundary_closure_help)
    mkpath(dirname(save_path))
    println("  path = ", save_path)
    runs = if isnothing(run_h32)
        (; run_h = run_h,
         run_h2 = run_h2,
         run_h4 = run_h4,
         run_h8 = run_h8,
         run_h16 = run_h16)
    else
        (; run_h = run_h,
         run_h2 = run_h2,
         run_h4 = run_h4,
         run_h8 = run_h8,
         run_h16 = run_h16,
         run_h32 = run_h32)
    end
    metadata = _bundle_metadata(save_path, runs)
    metadata_markdown = _bundle_metadata_markdown(save_path, runs)
    if isnothing(run_h32)
        jldsave(save_path;
                metadata = metadata,
                metadata_markdown = metadata_markdown,
                run_h = runs.run_h,
                run_h2 = runs.run_h2,
                run_h4 = runs.run_h4,
                run_h8 = runs.run_h8,
                run_h16 = runs.run_h16)
    else
        jldsave(save_path;
                metadata = metadata,
                metadata_markdown = metadata_markdown,
                run_h = runs.run_h,
                run_h2 = runs.run_h2,
                run_h4 = runs.run_h4,
                run_h8 = runs.run_h8,
                run_h16 = runs.run_h16,
                run_h32 = runs.run_h32)
    end
    println("  metadata = JLD2:/metadata")
    println("Done!")
end

function run_non_diagonal(; accuracy_order::Int = 4, p::Int = 2, dt = 0.01,
                          save_every::Int = 1, T_final::Real = 20.0, N::Int = 25,
                          R::Real = 25.0,
                          target_eltype::Union{Nothing, Type} = DEFAULT_OPERATOR_CONFIG.target_eltype,
                          boundary_condition::Symbol = DEFAULT_WAVE_CONFIG.boundary_condition,
                          initial_data_kind::Symbol = DEFAULT_WAVE_CONFIG.initial_data_kind,
                          enforce_origin::Bool = DEFAULT_WAVE_CONFIG.enforce_origin,
                          check_provided_dt_stability::Bool = DEFAULT_WAVE_CONFIG.check_provided_dt_stability,
                          incoming_amplitude::Real = DEFAULT_WAVE_CONFIG.incoming_amplitude,
                          incoming_center::Real = DEFAULT_WAVE_CONFIG.incoming_center,
                          incoming_radius::Real = DEFAULT_WAVE_CONFIG.incoming_radius,
                          outgoing_amplitude::Real = DEFAULT_WAVE_CONFIG.outgoing_amplitude,
                          gaussian_pi_amplitude::Real = DEFAULT_WAVE_CONFIG.gaussian_pi_amplitude,
                          gaussian_pi_center::Real = DEFAULT_WAVE_CONFIG.gaussian_pi_center,
                          gaussian_pi_width::Real = DEFAULT_WAVE_CONFIG.gaussian_pi_width,
                          regular_gaussian_amplitude::Real = DEFAULT_WAVE_CONFIG.regular_gaussian_amplitude,
                          regular_gaussian_width::Real = DEFAULT_WAVE_CONFIG.regular_gaussian_width,
                          regular_gaussian_center::Real = DEFAULT_WAVE_CONFIG.regular_gaussian_center,
                          alg_name::Symbol = DEFAULT_WAVE_CONFIG.alg_name,
                          experimental::Bool = false,
                          outer_boundary_closure_help::Bool = DEFAULT_OPERATOR_CONFIG.outer_boundary_closure_help,
                          include_h32::Bool = false)
    family = experimental ? :non_diagonal_exp : :non_diagonal
    dt_base = _resolve_convergence_dt(family = family,
                                      source = DEFAULT_OPERATOR_CONFIG.source,
                                      accuracy_order = accuracy_order,
                                      N = N,
                                      R = R,
                                      p = p,
                                      mode = DEFAULT_OPERATOR_CONFIG.mode,
                                      outer_boundary_closure_help = outer_boundary_closure_help,
                                      target_eltype = target_eltype,
                                      dt = dt,
                                      safety_factor = DEFAULT_WAVE_CONFIG.safety_factor,
                                      boundary_condition = boundary_condition,
                                      alg_name = alg_name,
                                      enforce_origin = enforce_origin)
    experimental && accuracy_order != 6 &&
        throw(ArgumentError("`experimental=true` currently requires `accuracy_order = 6`."))
    label = experimental ? "experimental non-diagonal SBP operators" :
            "non-diagonal SBP operators"
    println("Running $label for accuracy order $accuracy_order...")
    println("================================================================================")
    println("Running baseline (h)")
    run_h = run_simulation(family = family,
                           accuracy_order = accuracy_order,
                           N = N,
                           R = R,
                           p = p,
                           outer_boundary_closure_help = outer_boundary_closure_help,
                           target_eltype = target_eltype,
                           boundary_condition = boundary_condition,
                           initial_data_kind = initial_data_kind,
                           enforce_origin = enforce_origin,
                           incoming_amplitude = incoming_amplitude,
                           incoming_center = incoming_center,
                           incoming_radius = incoming_radius,
                           outgoing_amplitude = outgoing_amplitude,
                           gaussian_pi_amplitude = gaussian_pi_amplitude,
                           gaussian_pi_center = gaussian_pi_center,
                           gaussian_pi_width = gaussian_pi_width,
                           regular_gaussian_amplitude = regular_gaussian_amplitude,
                           regular_gaussian_width = regular_gaussian_width,
                           regular_gaussian_center = regular_gaussian_center,
                           alg_name = alg_name,
                           check_provided_dt_stability = check_provided_dt_stability,
                           T_final = T_final, dt = dt_base,
                           save_every = save_every)
    println("Running h/2")
    run_h2 = run_simulation(family = family,
                            accuracy_order = accuracy_order,
                            N = N * 2,
                            R = R,
                            p = p,
                            outer_boundary_closure_help = outer_boundary_closure_help,
                            target_eltype = target_eltype,
                            boundary_condition = boundary_condition,
                            initial_data_kind = initial_data_kind,
                            enforce_origin = enforce_origin,
                            incoming_amplitude = incoming_amplitude,
                            incoming_center = incoming_center,
                            incoming_radius = incoming_radius,
                            outgoing_amplitude = outgoing_amplitude,
                            gaussian_pi_amplitude = gaussian_pi_amplitude,
                            gaussian_pi_center = gaussian_pi_center,
                            gaussian_pi_width = gaussian_pi_width,
                            regular_gaussian_amplitude = regular_gaussian_amplitude,
                            regular_gaussian_width = regular_gaussian_width,
                            regular_gaussian_center = regular_gaussian_center,
                            alg_name = alg_name,
                            check_provided_dt_stability = check_provided_dt_stability,
                            T_final = T_final, dt = dt_base / 2,
                            save_every = save_every * 2)
    println("Running h/4")
    run_h4 = run_simulation(family = family,
                            accuracy_order = accuracy_order,
                            N = N * 4,
                            R = R,
                            p = p,
                            outer_boundary_closure_help = outer_boundary_closure_help,
                            target_eltype = target_eltype,
                            boundary_condition = boundary_condition,
                            initial_data_kind = initial_data_kind,
                            enforce_origin = enforce_origin,
                            incoming_amplitude = incoming_amplitude,
                            incoming_center = incoming_center,
                            incoming_radius = incoming_radius,
                            outgoing_amplitude = outgoing_amplitude,
                            gaussian_pi_amplitude = gaussian_pi_amplitude,
                            gaussian_pi_center = gaussian_pi_center,
                            gaussian_pi_width = gaussian_pi_width,
                            regular_gaussian_amplitude = regular_gaussian_amplitude,
                            regular_gaussian_width = regular_gaussian_width,
                            regular_gaussian_center = regular_gaussian_center,
                            alg_name = alg_name,
                            check_provided_dt_stability = check_provided_dt_stability,
                            T_final = T_final, dt = dt_base / 4,
                            save_every = save_every * 4)
    println("Running h/8")
    run_h8 = run_simulation(family = family,
                            accuracy_order = accuracy_order,
                            N = N * 8,
                            R = R,
                            p = p,
                            outer_boundary_closure_help = outer_boundary_closure_help,
                            target_eltype = target_eltype,
                            boundary_condition = boundary_condition,
                            initial_data_kind = initial_data_kind,
                            enforce_origin = enforce_origin,
                            incoming_amplitude = incoming_amplitude,
                            incoming_center = incoming_center,
                            incoming_radius = incoming_radius,
                            outgoing_amplitude = outgoing_amplitude,
                            gaussian_pi_amplitude = gaussian_pi_amplitude,
                            gaussian_pi_center = gaussian_pi_center,
                            gaussian_pi_width = gaussian_pi_width,
                            regular_gaussian_amplitude = regular_gaussian_amplitude,
                            regular_gaussian_width = regular_gaussian_width,
                            regular_gaussian_center = regular_gaussian_center,
                            alg_name = alg_name,
                            check_provided_dt_stability = check_provided_dt_stability,
                            T_final = T_final, dt = dt_base / 8,
                            save_every = save_every * 8)
    println("Running h/16")
    run_h16 = run_simulation(family = family,
                             accuracy_order = accuracy_order,
                             N = N * 16,
                             R = R,
                             p = p,
                             outer_boundary_closure_help = outer_boundary_closure_help,
                             target_eltype = target_eltype,
                             boundary_condition = boundary_condition,
                             initial_data_kind = initial_data_kind,
                             enforce_origin = enforce_origin,
                             incoming_amplitude = incoming_amplitude,
                             incoming_center = incoming_center,
                             incoming_radius = incoming_radius,
                             outgoing_amplitude = outgoing_amplitude,
                             gaussian_pi_amplitude = gaussian_pi_amplitude,
                             gaussian_pi_center = gaussian_pi_center,
                             gaussian_pi_width = gaussian_pi_width,
                             regular_gaussian_amplitude = regular_gaussian_amplitude,
                             regular_gaussian_width = regular_gaussian_width,
                             regular_gaussian_center = regular_gaussian_center,
                             alg_name = alg_name,
                             check_provided_dt_stability = check_provided_dt_stability,
                             T_final = T_final, dt = dt_base / 16,
                             save_every = save_every * 16)
    run_h32 = nothing
    if include_h32
        println("Running h/32")
        run_h32 = try
            run_simulation(family = family,
                           accuracy_order = accuracy_order,
                           N = N * 32,
                           R = R,
                           p = p,
                           outer_boundary_closure_help = outer_boundary_closure_help,
                           target_eltype = target_eltype,
                           boundary_condition = boundary_condition,
                           initial_data_kind = initial_data_kind,
                           enforce_origin = enforce_origin,
                           incoming_amplitude = incoming_amplitude,
                           incoming_center = incoming_center,
                           incoming_radius = incoming_radius,
                           outgoing_amplitude = outgoing_amplitude,
                           gaussian_pi_amplitude = gaussian_pi_amplitude,
                           gaussian_pi_center = gaussian_pi_center,
                           gaussian_pi_width = gaussian_pi_width,
                           regular_gaussian_amplitude = regular_gaussian_amplitude,
                           regular_gaussian_width = regular_gaussian_width,
                           regular_gaussian_center = regular_gaussian_center,
                           alg_name = alg_name,
                           check_provided_dt_stability = check_provided_dt_stability,
                           T_final = T_final, dt = dt_base / 32,
                           save_every = save_every * 32)
        catch e
            println("Failed to run h/32 simulation: ", e)
            nothing
        end
    end
    println("Saving to file...")
    save_path = _gundlach_save_path(accuracy_order,
                                    family,
                                    run_h.alg,
                                    eltype(run_h.ops.r);
                                    initial_data_kind = run_h.initial_data.kind,
                                    boundary_condition = run_h.wave_config.boundary_condition,
                                    outer_boundary_closure_help = run_h.operator_config.outer_boundary_closure_help)
    mkpath(dirname(save_path))
    println("  path = ", save_path)
    runs = if isnothing(run_h32)
        (; run_h = run_h,
         run_h2 = run_h2,
         run_h4 = run_h4,
         run_h8 = run_h8,
         run_h16 = run_h16)
    else
        (; run_h = run_h,
         run_h2 = run_h2,
         run_h4 = run_h4,
         run_h8 = run_h8,
         run_h16 = run_h16,
         run_h32 = run_h32)
    end
    metadata = _bundle_metadata(save_path, runs)
    metadata_markdown = _bundle_metadata_markdown(save_path, runs)
    if isnothing(run_h32)
        jldsave(save_path;
                metadata = metadata,
                metadata_markdown = metadata_markdown,
                run_h = runs.run_h,
                run_h2 = runs.run_h2,
                run_h4 = runs.run_h4,
                run_h8 = runs.run_h8,
                run_h16 = runs.run_h16)
    else
        jldsave(save_path;
                metadata = metadata,
                metadata_markdown = metadata_markdown,
                run_h = runs.run_h,
                run_h2 = runs.run_h2,
                run_h4 = runs.run_h4,
                run_h8 = runs.run_h8,
                run_h16 = runs.run_h16,
                run_h32 = runs.run_h32)
    end
    println("  metadata = JLD2:/metadata")
    println("Done!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    # result = run_simulation()
    # viewer = interactive_wave_viewer(result)

    println()
    println("Saved time points: ", length(result.sol.t))
    println("Use the slider at the bottom of the GLMakie window to scrub through time.")

    if viewer.screen !== nothing
        wait(viewer.screen)
    end
end
