import CairoMakie: save
import SphericalSBPOperators
import SphericalSBPOperators.Plots: plot_energy_history, spectral_analysis

include(joinpath(@__DIR__, "run_wave_evolution.jl"))

# 1) Run from scripts.
result = run_wave_evolution(; N = 32, R = 1.0, p = 2, T_final = 0.2, verbose = true)

# 2) Call plotting functions from the package API.
fig_energy = plot_energy_history(result.sol.t, result.sol.energy)
mkpath("plots")
save("plots/intro_energy.png", fig_energy)

# 3) Call analysis functions from the package API.
spectral = spectral_analysis(
    Matrix(result.ops.D),
    Matrix(result.ops.Geven);
    H = Matrix(result.ops.H),
    B = Matrix(result.ops.B),
    boundary_condition = :absorbing,
    enforce_origin = true,
    throw_on_instability = false
)
println("max Re(lambda) = ", spectral.max_real)
println("saved: plots/intro_energy.png")
