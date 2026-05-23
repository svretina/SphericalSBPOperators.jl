import SphericalSBPOperators: solve_wave_ode, diagonal_spherical_operators, validate
import SummationByPartsOperators: MattssonNordström2004

using OrdinaryDiffEqLowOrderRK: RK4
using OrdinaryDiffEqSDIRK: ImplicitMidpoint

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
        phi0 = nothing,
        pi0 = nothing,
        xi0 = nothing,
        save_every::Int = 1,
        build_matrix::Symbol = :matrix_if_square,
        noise_amplitude::Real = 0.0,
        noise_seed = nothing,
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

    source = MattssonNordström2004()

    ops = diagonal_spherical_operators(
        source;
        accuracy_order = accuracy_order,
        N = N,
        R = R,
        p = p,
        build_matrix = build_matrix
    )

    report = validate(ops; max_monomial_degree = accuracy_order, verbose = false)

    bc_reflecting = boundary_condition === :reflecting || boundary_condition === :reflective
    alg_use = alg === nothing ? (bc_reflecting ? ImplicitMidpoint() : RK4()) : alg

    sol = solve_wave_ode(
        ops;
        T_final = T_final,
        dt = dt,
        alg = alg_use,
        safety_factor = safety_factor,
        boundary_condition = boundary_condition,
        phi0 = phi0_value,
        pi0 = pi0_value,
        xi0 = xi0_value,
        save_every = save_every,
        noise_amplitude = noise_amplitude,
        noise_seed = noise_seed,
        verbose = verbose
    )

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
        println(
            "  initial Π[end] = ", sol.Π[end, 1],
            " (SAT BCs are imposed through RHS, not state projection)"
        )
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
