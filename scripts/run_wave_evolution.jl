using SphericalSBPOperators
using OrdinaryDiffEqLowOrderRK: RK4
using OrdinaryDiffEqSDIRK: ImplicitMidpoint
using SummationByPartsOperators: MattssonNordström2004

"""
    run_wave_evolution(; kwargs...)

Build folded spherical SBP operators and run a radial wave evolution.

Keyword variables (what each one means):
- `accuracy_order::Int`: SBP design accuracy order on the full mirrored grid.
- `N::Int`: number of half-intervals on `[0, R]`; half-grid has `N+1` nodes.
- `R::Real`: outer radius of the computational domain.
- `p::Int`: metric power (`0` Cartesian, `1` cylindrical, `2` spherical).
- `T_final::Real`: final simulation time.
- `dt`: fixed timestep. If `nothing`, estimate from linear stability.
- `safety_factor::Real`: multiplies estimated stable timestep (`0 < safety_factor <= 1` typical).
- `boundary_condition::Symbol`: one of `:absorbing`, `:reflecting`, `:dirichlet`, `:none`.
- `alg`: optional SciML integrator algorithm. If omitted, `:reflecting` uses
  `ImplicitMidpoint()`, otherwise `RK4()`.
- initial data:
  - `ϕ0`: optional scalar profile used only when `Ξ0` is omitted.
  - `Π0`: initial even field.
  - `Ξ0`: initial odd field.
  Legacy aliases `phi0`, `pi0`, `xi0` are also accepted.
- `save_every::Int`: save every this many RK steps.
- `build_matrix::Symbol`: derivative extraction mode (`:matrix_if_square` or `:probe`).
- `noise_amplitude::Real`: optional additive uniform step noise amplitude.
- `noise_seed`: optional RNG seed for reproducible noise.
- `verbose::Bool`: print run summary.

Returns a named tuple:
- `ops`: constructed spherical SBP operators.
- `report`: validation report for operators.
- `sol`: wave evolution result (`sol.t`, `sol.pi`, `sol.xi`, `sol.energy`).
"""
function run_wave_evolution(;
                            accuracy_order::Int = 6,
                            N::Int = 64,
                            R::Real = 1.0,
                            p::Int = 2,
                            T_final::Real = 0.2,
                            dt = nothing,
                            alg = nothing,
                            safety_factor::Real = 0.9,
                            boundary_condition::Symbol = :absorbing,
                            ϕ0 = nothing,
                            Π0 = nothing,
                            Ξ0 = nothing,
                            phi0 = nothing,
                            pi0 = nothing,
                            xi0 = nothing,
                            save_every::Int = 1,
                            build_matrix::Symbol = :matrix_if_square,
                            noise_amplitude::Real = 0.0,
                            noise_seed = nothing,
                            verbose::Bool = true)
    # 1) Choose the underlying Cartesian SBP family on the mirrored grid [-R, R].
    source = MattssonNordström2004()

    # 2) Construct folded spherical operators on [0, R].
    ops = spherical_operators(source;
                              accuracy_order = accuracy_order,
                              N = N,
                              R = R,
                              p = p,
                              build_matrix = build_matrix)

    # 3) Optionally validate the operator set before time evolution.
    report = validate(ops; max_monomial_degree = accuracy_order, verbose = false)

    # 4) Time integrator choice.
    bc_reflecting = boundary_condition === :reflecting || boundary_condition === :reflective
    alg_use = alg === nothing ? (bc_reflecting ? ImplicitMidpoint() : RK4()) : alg

    # 5) Evolve the wave system.
    sol = solve_wave_ode(ops;
                         T_final = T_final,
                         dt = dt,
                         alg = alg_use,
                         safety_factor = safety_factor,
                         boundary_condition = boundary_condition,
                         ϕ0 = ϕ0,
                         Π0 = Π0,
                         Ξ0 = Ξ0,
                         phi0 = phi0,
                         pi0 = pi0,
                         xi0 = xi0,
                         save_every = save_every,
                         noise_amplitude = noise_amplitude,
                         noise_seed = noise_seed,
                         verbose = verbose)

    if verbose
        println("\nWave run diagnostics")
        println("  Nh = ", ops.Nh, ", nsteps = ", sol.nsteps, ", dt = ", sol.dt)
        println("  boundary_condition = ", sol.boundary_condition)
        println("  integrator = ", typeof(alg_use))
        println("  sbp_no_origin = ", report.sbp.sbp_no_origin)
        println("  initial_data_consistent = ", sol.initial_data_check.consistent)
        println("  initial origin_residual (Ξ[1]) = ", sol.initial_data_check.origin_residual)
        println("  initial boundary_residual = ", sol.initial_data_check.boundary_residual)
        println("  initial max|Π| interior = ", sol.initial_data_check.max_abs_pi_interior)
        println("  initial Π[end] = ", sol.Π[end, 1], " (SAT BCs are imposed through RHS, not state projection)")
        println("  initial_energy = ", sol.energy[1], ", final_energy = ", sol.energy[end])
    end

    return (ops = ops, report = report, sol = sol)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_wave_evolution()

    # Example post-processing variables:
    # - `result.sol.t`: saved time vector
    # - `result.sol.pi`: Π field history (Nh x Nt)
    # - `result.sol.xi`: Ξ field history (Nh x Nt)
    # - `result.sol.energy`: discrete energy history
    println("\nSaved time points: ", length(result.sol.t))
end
