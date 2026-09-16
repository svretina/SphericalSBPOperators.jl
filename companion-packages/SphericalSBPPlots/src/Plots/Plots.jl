module Plots

import DrWatson: plotsdir, savename
import GenericSchur
import MakiePublication: theme_aps
import MultiFloats: Float64x4
import OrdinaryDiffEqLowOrderRK: RK4
import Printf: @printf, @sprintf
import SciMLBase: ODEProblem, solve
import SummationByPartsOperators: derivative_operator, dissipation_operator, grid

using CairoMakie: @lift, Axis, Colorbar, Figure, Label, Observable, RGBAf, Relative,
    axislegend, colgap!, colsize!, heatmap!, hlines!, lines!, record,
    rowsize!, save, scatter!, text!, vlines!, with_theme, ylims!
using LaTeXStrings: @L_str, latexstring
using LinearAlgebra: dot, eigen, eigvals, norm
using SparseArrays: SparseMatrixCSC, sparse

export assemble_block_operator
export rk4_amplification_max, rk4_dt_max_from_spectrum, spectral_analysis
export test_pole_stability, build_folded_sbp_dissipation,
    test_pole_stability_dissipative_overlay
export zero_crossings, oscillation_metrics, dominant_real_projection,
    nullspace_and_checkerboard
export inward_packet_initial_data, wave_rhs_vec!, discrete_energy, reflection_ibvp_test
export save_spectrum_plot, save_spectrum_overlay_plot, save_nullspace_plot
export save_reflection_heatmap, save_energy_trace, save_dashboard, save_reflection_animation
export print_suite_summary

export plot_energy_history, plot_initial_data, plot_field_evolution, animate_field_evolution
export generate_simulation_plots, plot_bc_energy_comparison, theme_prd

include("operators.jl")
include("wave_simulations.jl")

end
