function _snapshot_series(sol, field::Symbol, time_index::Int)
    U = _wave_field_history(sol, field)
    nt = size(U, 2)
    1 <= time_index <= nt ||
        throw(ArgumentError("`time_index` must satisfy 1 <= time_index <= $nt."))
    return Float64.(sol.r), Float64.(U[:, time_index]), Float64(sol.t[time_index])
end

"""
    plot_wave_snapshot(simulation, time_index)

Plot the `Π` and `Ψ` fields from a single simulation at a chosen time index.
"""
function plot_wave_snapshot(simulation,
                            time_index::Int;
                            title::Union{Nothing, AbstractString} = nothing)
    sol = _extract_wave_solution(simulation)
    r_pi, pi_values, t_value = _snapshot_series(sol, :Π, time_index)
    r_psi, psi_values, _ = _snapshot_series(sol, :Ψ, time_index)

    plot_title = isnothing(title) ?
                 latexstring("t = ", string(round(t_value; digits = 6))) : title

    fig = _with_spectrum_theme() do
        fig = Figure(size = (980, 440))

        ax_pi = Axis(fig[1, 1],
                     xlabel = L"r",
                     ylabel = L"r \Pi(r,t)",
                     title = L"r \,\Pi")
        lines!(ax_pi, r_pi, r_pi .* pi_values; linewidth = 2.5, color = :royalblue4)

        ax_psi = Axis(fig[1, 2],
                      xlabel = L"r",
                      ylabel = L"r \Psi(r,t)",
                      title = L"r \, \Psi")
        lines!(ax_psi, r_psi, r_psi .* psi_values; linewidth = 2.5, color = :seagreen4)

        Label(fig[0, 1:2], plot_title; fontsize = 18, font = :bold)
        fig
    end

    return (fig = fig,
            t = t_value,
            pi = (r = r_pi, values = pi_values),
            psi = (r = r_psi, values = psi_values))
end

"""
    plot_wave_snapshot_resolutions(simulation_h, simulation_h2, simulation_h4, time_index)

Plot `Π` and `Ψ` at a common nested time index from three simulations with resolutions
`dr`, `dr/2`, and `dr/4`.

The provided `time_index` is interpreted on the coarsest common timeline; the medium
and fine solutions are sampled at `2i-1` and `4i-3`, respectively.
"""
function plot_wave_snapshot_resolutions(simulation_h,
                                        simulation_h2,
                                        simulation_h4,
                                        time_index::Int;
                                        labels::NTuple{3, AbstractString} = ("dr", "dr/2",
                                                                             "dr/4"),
                                        title::Union{Nothing, AbstractString} = nothing)
    sol_h = _extract_wave_solution(simulation_h)
    sol_h2 = _extract_wave_solution(simulation_h2)
    sol_h4 = _extract_wave_solution(simulation_h4)

    time_sampling = _nested_common_sampling(length(sol_h.t), length(sol_h2.t),
                                            length(sol_h4.t))
    common_t = _validate_nested_vector_alignment(sol_h.t, sol_h2.t, sol_h4.t, time_sampling;
                                                 name = "time")
    1 <= time_index <= length(common_t) ||
        throw(ArgumentError("`time_index` must satisfy 1 <= time_index <= $(length(common_t))."))

    idx_h = time_sampling.coarse[time_index]
    idx_h2 = time_sampling.medium[time_index]
    idx_h4 = time_sampling.fine[time_index]
    t_value = common_t[time_index]

    r_h, pi_h, _ = _snapshot_series(sol_h, :Π, idx_h)
    r_h2, pi_h2, _ = _snapshot_series(sol_h2, :Π, idx_h2)
    r_h4, pi_h4, _ = _snapshot_series(sol_h4, :Π, idx_h4)

    _, psi_h, _ = _snapshot_series(sol_h, :Ψ, idx_h)
    _, psi_h2, _ = _snapshot_series(sol_h2, :Ψ, idx_h2)
    _, psi_h4, _ = _snapshot_series(sol_h4, :Ψ, idx_h4)

    plot_title = isnothing(title) ?
                 latexstring("\\mathrm{Wave\\ Snapshot\\ Comparison\\ at}\\ t = ",
                             string(round(t_value; digits = 6))) :
                 title

    fig = _with_spectrum_theme() do
        fig = Figure(size = (1020, 460))

        ax_pi = Axis(fig[1, 1],
                     xlabel = L"r",
                     ylabel = L"\Pi(r,t)",
                     title = L"\Pi\ \mathrm{snapshot}")
        lines!(ax_pi, r_h, pi_h; linewidth = 2.5, label = labels[1])
        lines!(ax_pi, r_h2, pi_h2; linewidth = 2.5, linestyle = :dash, label = labels[2])
        lines!(ax_pi, r_h4, pi_h4; linewidth = 2.5, linestyle = :dot, label = labels[3])
        axislegend(ax_pi; position = :rb)

        ax_psi = Axis(fig[1, 2],
                      xlabel = L"r",
                      ylabel = L"\Psi(r,t)",
                      title = L"\Psi\ \mathrm{snapshot}")
        lines!(ax_psi, r_h, psi_h; linewidth = 2.5, label = labels[1])
        lines!(ax_psi, r_h2, psi_h2; linewidth = 2.5, linestyle = :dash, label = labels[2])
        lines!(ax_psi, r_h4, psi_h4; linewidth = 2.5, linestyle = :dot, label = labels[3])
        axislegend(ax_psi; position = :rb)

        Label(fig[0, 1:2], plot_title; fontsize = 18, font = :bold)
        fig
    end

    return (fig = fig,
            t = t_value,
            coarse = (r = r_h, pi = pi_h, psi = psi_h),
            medium = (r = r_h2, pi = pi_h2, psi = psi_h2),
            fine = (r = r_h4, pi = pi_h4, psi = psi_h4))
end
