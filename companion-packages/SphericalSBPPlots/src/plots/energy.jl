"""
    plot_wave_energy_history(simulation; normalize=false, difference_mode=:none)

Plot the discrete energy history for a single simulation.

The input may be a `WaveEvolutionResult` directly or a larger result object that
stores the solution in `.sol`, such as the return value of `run_simulation()`.
"""
const _ENERGY_SUPERTITLE = "Error in Energy conservation."
const _ENERGY_TITLE_SIZE = 24
const _ENERGY_AXIS_LABEL_SIZE = 20
const _ENERGY_TICK_LABEL_SIZE = 16
const _ENERGY_LEGEND_LABEL_SIZE = 16

@inline function _energy_ticklabel(v::Real; digits::Int = 6)
    txt = @sprintf("%.*e", digits, Float64(v))
    txt = replace(txt, r"(\.\d*?[1-9])0+e" => s"\1e")
    txt = replace(txt, r"\.0+e" => "e")
    return txt
end

@inline function _energy_tickformat(values)
    vals = Float64.(values)
    diffs = Float64[]
    for i in 1:(length(vals) - 1)
        Δ = abs(vals[i + 1] - vals[i])
        Δ > 0 && push!(diffs, Δ)
    end

    digits = 6
    if !isempty(diffs)
        scale = maximum(abs, vals)
        if scale > 0
            exponent = floor(Int, log10(scale))
            scaled_step = minimum(diffs) / 10.0^exponent
            digits = clamp(Int(ceil(-log10(scaled_step))) + 2, 6, 12)
        end
    end

    return [_energy_ticklabel(v; digits = digits) for v in vals]
end

function _energy_plot_values(energy::AbstractVector{<:Real};
                             normalize::Bool = false,
                             difference_mode::Symbol = :none)
    if normalize && difference_mode != :none
        throw(ArgumentError("`normalize=true` cannot be combined with `difference_mode=$(difference_mode)`."))
    end

    energy0 = Float64(energy[1])
    values = if normalize
        Float64.(energy ./ energy[1])
    elseif difference_mode == :none
        Float64.(energy)
    elseif difference_mode == :absolute
        Float64.(energy .- energy[1])
    elseif difference_mode == :relative
        abs(energy0) > 0 ||
            throw(ArgumentError("Relative energy difference is undefined because E(0) = 0."))
        Float64.((energy .- energy[1]) ./ energy0)
    else
        throw(ArgumentError("Unsupported `difference_mode=$(difference_mode)`. Use :none, :absolute, or :relative."))
    end

    ylabel = if difference_mode == :absolute
        L"E(t) - E(0)"
    elseif difference_mode == :relative
        L"\frac{E(t) - E(0)}{E(0)}"
    elseif normalize
        L"E(t) / E(0)"
    else
        L"E(t)"
    end

    return values, ylabel
end

function plot_wave_energy_history(simulation;
                                  normalize::Bool = false,
                                  difference_mode::Symbol = :none,
                                  title::AbstractString = "Error in Energy conservation.")
    sol = _extract_wave_solution(simulation)
    energy = Float64.(sol.energy)
    yvals, ylabel = _energy_plot_values(energy; normalize = normalize,
                                        difference_mode = difference_mode)
    fig = _with_spectrum_theme(axis_labelsize = _ENERGY_AXIS_LABEL_SIZE,
                               tick_labelsize = _ENERGY_TICK_LABEL_SIZE,
                               title_size = _ENERGY_TITLE_SIZE,
                               legend_labelsize = _ENERGY_LEGEND_LABEL_SIZE) do
        # Keep the explicit figure padding minimal so the PDF export does not
        # leave a large blank strip to the left of the y-axis label.
        fig = Figure(size = (880, 480), figure_padding = (8, 12, 8, 8))
        Label(fig[0, 1], title; fontsize = _ENERGY_TITLE_SIZE)
        ax = Axis(fig[1, 1], xlabel = L"t", ylabel = ylabel,
                  ytickformat = _energy_tickformat)
        lines!(ax, Float64.(sol.t), yvals; linewidth = 2.5, color = :black)
        xlims!(ax, 0.0, Float64(last(sol.t)))
        fig
    end

    return (fig = fig, t = Float64.(sol.t), energy = yvals)
end

"""
    plot_wave_energy_histories(simulation_h, simulation_h2, simulation_h4;
                               normalize=false, difference_mode=:none)

Plot the energy histories from three simulations with nested resolutions on the same
axes so their energy behavior can be compared directly.
"""
function plot_wave_energy_histories(simulations::AbstractVector;
                                    normalize::Bool = false,
                                    difference_mode::Symbol = :none,
                                    labels::AbstractVector{<:AbstractString} = [string(i) for i in eachindex(simulations)],
                                    title::AbstractString = "Error in Energy conservation.",
                                    legend_position::Symbol = :lb)
    isempty(simulations) &&
        throw(ArgumentError("Need at least one simulation to plot energy histories."))
    length(labels) == length(simulations) ||
        throw(DimensionMismatch("`labels` must have the same length as `simulations`."))

    solutions = [_extract_wave_solution(sim) for sim in simulations]
    histories = map(solutions) do sol
        yvals, ylabel = _energy_plot_values(Float64.(sol.energy); normalize = normalize,
                                            difference_mode = difference_mode)
        return (t = Float64.(sol.t), energy = yvals, ylabel = ylabel)
    end
    ylabel = first(histories).ylabel
    linestyles = (:solid, :dash, :dot, :dashdot, :dashdotdot)

    fig = _with_spectrum_theme(axis_labelsize = _ENERGY_AXIS_LABEL_SIZE,
                               tick_labelsize = _ENERGY_TICK_LABEL_SIZE,
                               title_size = _ENERGY_TITLE_SIZE,
                               legend_labelsize = _ENERGY_LEGEND_LABEL_SIZE) do
        # Keep the explicit figure padding minimal so the PDF export does not
        # leave a large blank strip to the left of the y-axis label.
        fig = Figure(size = (920, 520), figure_padding = (8, 12, 8, 8))
        Label(fig[0, 1], title; fontsize = _ENERGY_TITLE_SIZE)
        ax = Axis(fig[1, 1], xlabel = L"t", ylabel = ylabel,
                  ytickformat = _energy_tickformat)
        for i in eachindex(histories, labels)
            history = histories[i]
            lines!(ax, history.t, history.energy; linewidth = 2.5,
                   linestyle = linestyles[mod1(i, length(linestyles))],
                   label = labels[i])
        end
        xlims!(ax, 0.0, maximum(last(history.t) for history in histories))
        axislegend(ax; position = legend_position)
        fig
    end

    return (fig = fig, histories = histories)
end

function plot_wave_energy_histories(simulation_h,
                                    simulation_h2,
                                    simulation_h4;
                                    normalize::Bool = false,
                                    difference_mode::Symbol = :none,
                                    labels::NTuple{3, AbstractString} = ("dr", "dr/2",
                                                                         "dr/4"),
                                    title::AbstractString = "Error in Energy conservation.",
                                    legend_position::Symbol = :lb)
    plotted = plot_wave_energy_histories([simulation_h, simulation_h2, simulation_h4];
                                         normalize = normalize,
                                         difference_mode = difference_mode,
                                         labels = collect(labels),
                                         title = title,
                                         legend_position = legend_position)
    return (fig = plotted.fig,
            coarse = plotted.histories[1],
            medium = plotted.histories[2],
            fine = plotted.histories[3])
end
