"""
    WaveEvolutionResult

Container for the discrete wave evolution on the folded half grid.
"""
struct WaveEvolutionResult{T <: Real}
    t::Vector{T}
    Π::Matrix{T}
    Ξ::Matrix{T}
    energy::Vector{T}
    r::Vector{T}
    dt::T
    nsteps::Int
    boundary_condition::Symbol
    initial_data_check
end

# Backward-compatible aliases.
function Base.getproperty(sol::WaveEvolutionResult, name::Symbol)
    if name === :pi
        return getfield(sol, :Π)
    elseif name === :xi
        return getfield(sol, :Ξ)
    end
    return getfield(sol, name)
end

function Base.propertynames(::WaveEvolutionResult, private::Bool = false)
    base = (:t, :Π, :Ξ, :energy, :r, :dt, :nsteps, :boundary_condition, :initial_data_check)
    return private ? (base..., :pi, :xi) : (base..., :pi, :xi)
end

function _vector_from_input(data, r::Vector{Float64}, name::AbstractString)
    if data === nothing
        return nothing
    end

    values = if data isa Function
        data(r)
    else
        data
    end

    vec = Float64.(collect(values))
    length(vec) == length(r) ||
        throw(DimensionMismatch("`$name` length must match grid size $(length(r))."))
    return vec
end

"""
    compute_stability_limit(alg; tol=1e-3, max_search=-1e9)

Compute the left-most real-axis stability point `z_min` for a one-step method by
binary search on the scalar test equation `u_t = z u`.
"""
function compute_stability_limit(alg; tol::Real = 1e-3, max_search::Real = -1e9)
    tol > 0 || throw(ArgumentError("`tol` must be positive."))
    max_search < 0 || throw(ArgumentError("`max_search` must be negative."))

    prob = ODEProblem((u, p, t) -> p[1] * u, ComplexF64[1.0 + 0im], (0.0, 1.0), ComplexF64[0.0 + 0im])
    z_left = Float64(max_search)
    z_right = 0.0

    while abs(z_right - z_left) > Float64(tol)
        z_mid = 0.5 * (z_left + z_right)
        sol = solve(remake(prob, p = ComplexF64[z_mid + 0im]), alg; adaptive = false, dt = 1.0, verbose = false)
        if abs(sol.u[end][1]) <= 1.0 + Float64(tol)
            z_right = z_mid
        else
            z_left = z_mid
        end
    end

    return z_left
end

"""
    estimate_max_timestep(D::AbstractMatrix, G::AbstractMatrix, alg)

Estimate the maximum stable timestep for `u_t = D*G*u` by combining the method
stability abscissa with the most negative real eigenvalue of `D*G`.
"""
function estimate_max_timestep(D::AbstractMatrix, G::AbstractMatrix, alg)
    z_min = compute_stability_limit(alg)
    λ = eigen(Matrix(D * G)[1:end-1, 1:end-1]).values
    λ_min_real = minimum(real.(λ))
    if λ_min_real >= -1e-12
        return 1.0
    end

    return z_min / λ_min_real
end

"""
    estimate_wave_timestep(ops; alg=RK4(), safety_factor=0.9)

Estimate a stable explicit timestep using the linear operator stability estimate for
`Π_t = D Ξ, Ξ_t = G Π`, mapped to the second-order form `u_t = D G u`.
"""
function estimate_wave_timestep(ops::SphericalOperators;
                                alg = RK4(),
                                safety_factor::Real = 0.9)
    safety_factor > 0 || throw(ArgumentError("`safety_factor` must be positive."))

    Dscaled = Matrix(ops.D)
    G = Matrix(ops.Geven)
    dt_max = estimate_max_timestep(Dscaled, G, alg)
    dt_max > 0 || throw(ArgumentError("Estimated non-positive timestep $dt_max."))
    return Float64(safety_factor) * dt_max
end

@inline _method_stability_function(::RK4, z) = 1 + z + z^2 / 2 + z^3 / 6 + z^4 / 24
@inline _method_stability_function(::ImplicitMidpoint, z) = (1 + z / 2) / (1 - z / 2)

function _method_stability_function(alg, z)
    throw(
          ArgumentError(
                        "No stability-function implementation for algorithm $(typeof(alg)). " *
                        "Provided-`dt` stability checks currently support `RK4()` and `ImplicitMidpoint()`."
                       )
         )
end

function _wave_linear_operator(ops::SphericalOperators;
                               boundary_condition::Symbol,
                               enforce_origin::Bool)
    n = length(ops.r)
    params = WaveODEParams(ops;
                           boundary_condition = boundary_condition,
                           enforce_origin = enforce_origin)
    A = Matrix{Float64}(wave_system_jac_prototype(ops; boundary_condition = boundary_condition))
    wave_system_jac!(A, zeros(Float64, 2 * n), params, 0.0)
    return A
end

function _max_method_amplification(alg,
                                   eigvals::AbstractVector{<:Complex},
                                   dt::Real)
    isempty(eigvals) &&
        return (
                max_amp = 0.0,
                worst_index = 0,
                worst_lambda = 0.0 + 0.0im,
                worst_z = 0.0 + 0.0im,
                worst_R = 1.0 + 0.0im
               )

    zvals = ComplexF64.(Float64(dt) .* eigvals)
    Rvals = _method_stability_function.(Ref(alg), zvals)
    amp = abs.(Rvals)
    idx = argmax(amp)
    return (
            max_amp = amp[idx],
            worst_index = idx,
            worst_lambda = eigvals[idx],
            worst_z = zvals[idx],
            worst_R = Rvals[idx]
           )
end

function _rk4_dt_max_from_spectrum(eigvals::AbstractVector{<:Complex};
                                   tol::Float64 = 1e-6)
    isempty(eigvals) && return Inf
    ρ = maximum(abs.(eigvals))
    ρ <= eps(Float64) && return Inf

    rk4 = RK4()
    lo = 0.0
    hi = 1 / ρ

    while _max_method_amplification(rk4, eigvals, hi).max_amp <= 1 + tol && hi < 1e9
        lo = hi
        hi *= 1.5
    end

    if hi >= 1e9 && _max_method_amplification(rk4, eigvals, hi).max_amp <= 1 + tol
        return hi
    end

    for _ in 1:100
        mid = 0.5 * (lo + hi)
        if _max_method_amplification(rk4, eigvals, mid).max_amp <= 1 + tol
            lo = mid
        else
            hi = mid
        end
    end

    return lo
end

function _check_provided_dt_stability(ops::SphericalOperators,
                                      dt_step::Float64,
                                      alg;
                                      boundary_condition::Symbol,
                                      enforce_origin::Bool,
                                      stability_tol::Float64 = 1e-6,
                                      throw_on_failure::Bool = true)
    A = _wave_linear_operator(ops;
                              boundary_condition = boundary_condition,
                              enforce_origin = enforce_origin)
    eigvals = eigen(A).values
    max_real = maximum(real.(eigvals))

    amp = _max_method_amplification(alg, eigvals, dt_step)
    stable = amp.max_amp <= 1 + stability_tol

    dt_limit = if alg isa RK4
        _rk4_dt_max_from_spectrum(eigvals; tol = stability_tol)
    elseif alg isa ImplicitMidpoint
        max_real <= stability_tol ? Inf : 0.0
    else
        NaN
    end

    if !stable && throw_on_failure
        throw(
              ArgumentError(
                            "Provided `dt`=$dt_step is not stable for $(typeof(alg)) under linear spectral check " *
                            "(boundary_condition=$boundary_condition, enforce_origin=$enforce_origin). " *
                            "max |R(dt*λ)|=$(amp.max_amp) at λ=$(amp.worst_lambda), dt*λ=$(amp.worst_z), " *
                            "max Re(λ)=$max_real, suggested dt ≤ $dt_limit."
                           )
             )
    end

    return (
            stable = stable,
            max_amplification = amp.max_amp,
            worst_eigenvalue = amp.worst_lambda,
            worst_scaled_eigenvalue = amp.worst_z,
            worst_stability_function_value = amp.worst_R,
            max_real_eigenvalue = max_real,
            spectral_radius = maximum(abs.(eigvals)),
            suggested_dt_max = dt_limit
           )
end

function _build_saveat(dt::Float64, nsteps::Int, save_every::Int)
    indices = collect(0:save_every:nsteps)
    if isempty(indices) || indices[end] != nsteps
        push!(indices, nsteps)
    end
    return dt .* Float64.(indices)
end

@inline _lcg_step(x::UInt64) = x * 6364136223846793005 + 1442695040888963407

@inline function _lcg_symmetric_unit!(state::Base.RefValue{UInt64})
    s = _lcg_step(state[])
    state[] = s
    return 2.0 * (Float64(s) / Float64(typemax(UInt64))) - 1.0
end

function _step_noise_callback(noise_amplitude::Real,
                              n::Int,
                              state_layout::Symbol,
                              enforce_origin::Bool;
                              noise_seed::Union{Nothing, Int} = nothing)
    noise_amplitude > 0 || return nothing
    state_layout in (:matrix, :vector) ||
        throw(ArgumentError("`state_layout` must be either `:matrix` or `:vector`."))

    amp = Float64(noise_amplitude)
    seed_u = noise_seed === nothing ? UInt64(time_ns()) : reinterpret(UInt64, Int64(noise_seed))
    state = Ref(seed_u == 0 ? 0x9e3779b97f4a7c15 : seed_u)

    condition = (u, t, integrator) -> integrator.iter > 0

    function affect!(integrator)
        U = integrator.u
        π = nothing
        ξ = nothing
        if state_layout === :vector
            π = @view U[1:n]
            ξ = @view U[(n + 1):(2 * n)]
        else
            π = @view U[:, 1]
            ξ = @view U[:, 2]
        end

        @inbounds for i in 1:n
            π[i] = π[i] + amp * _lcg_symmetric_unit!(state)
            ξ[i] = ξ[i] + amp * _lcg_symmetric_unit!(state)
        end
        apply_symmetry_state!(π, ξ; enforce_origin = enforce_origin)
        return nothing
    end

    return DiscreteCallback(condition, affect!; save_positions = (false, false))
end

function _pick_keyword(primary, legacy, primary_name::AbstractString, legacy_name::AbstractString)
    if primary !== nothing && legacy !== nothing
        throw(ArgumentError("Use either `$primary_name` or `$legacy_name`, not both."))
    end
    return primary === nothing ? legacy : primary
end

@inline function _normalize_initial_data_mode(mode::Symbol)
    if mode === :phi
        return :potential
    elseif mode === :char
        return :characteristic
    elseif mode === :default
        return :auto
    end
    return mode
end

"""
    solve_wave_ode(ops; kwargs...)

Evolve the first-order radial wave system using SciML `ODEProblem` and
`OrdinaryDiffEqLowOrderRK.RK4`.

State variables on `[0, R]`:
- `Π(t, r)` even field (time derivative of scalar potential),
- `Ξ(t, r)` odd radial derivative field.

Semidiscrete equations:
- `∂t Π = D * Ξ`
- `∂t Ξ = Geven * Π`

Constraints are applied once per RHS call, after derivative construction:
- symmetry: `dΞ(0)=0`
- boundary condition on `(dΠ, dΞ)` at `r=R`

Keyword options:
- `T_final`: final time (required).
- `dt`: fixed timestep. If `nothing`, uses `estimate_wave_timestep`.
- `alg`: one-step algorithm (default `RK4()` from `OrdinaryDiffEqLowOrderRK`).
- `state_layout`: `:matrix` (`U[:,1]=Π, U[:,2]=Ξ`) or `:vector` (`U=[Π;Ξ]`).
  Defaults to `:vector` when `alg isa ImplicitMidpoint`, otherwise `:matrix`.
- `boundary_condition`: `:absorbing`, `:reflecting`, `:dirichlet`, or `:none`.
  Aliases: `:radiative` => `:absorbing`, `:reflective` => `:reflecting`.
- initial data: `ϕ0`, `Π0`, `Ξ0` (or legacy aliases `phi0`, `pi0`, `xi0`).
  If `Ξ0` is omitted in `:auto` mode, `Ξ0 = Geven * ϕ0`.
- potential mode: set `initial_data_mode=:potential` and provide `ϕ0` (and optionally
  `Π0`); then `Ξ0` is always built as `Geven * ϕ0`.
- characteristic mode: provide `w_in0`, `w_out0` (or `win0`, `wout0`) or set
  `initial_data_mode=:characteristic`; then
  `Π0 = 0.5*(w_in0+w_out0)`, `Ξ0 = 0.5*(w_in0-w_out0)`.
- `Ξ` growth note: even if `Ξ0=0`, non-constant `Π0` gives immediate
  `Ξ_t = Geven*Π0`, so `Ξ` appears right away. This is expected coupling.
- `save_every`: save every `save_every` RK steps (always saves final state).
- `enforce_origin`: enforce odd symmetry at origin.
- if `dt` is provided explicitly, linear spectral stability is checked against
  the selected one-step integrator.
- `noise_amplitude`: if positive, add uniform random noise in `[-a,a]` to both
  fields after every fixed step.
- `noise_seed`: optional seed for reproducible step-noise.
"""
function solve_wave_ode(ops::SphericalOperators;
                        T_final::Real,
                        dt = nothing,
                        alg = RK4(),
                        state_layout = (alg isa ImplicitMidpoint ? :vector : :matrix),
                        safety_factor::Real = 0.9,
                        boundary_condition::Symbol = :absorbing,
                        ϕ0 = nothing,
                        Π0 = nothing,
                        Ξ0 = nothing,
                        w_in0 = nothing,
                        w_out0 = nothing,
                        phi0 = nothing,
                        pi0 = nothing,
                        xi0 = nothing,
                        win0 = nothing,
                        wout0 = nothing,
                        initial_data_mode::Symbol = :auto,
                        save_every::Int = 1,
                        enforce_origin::Bool = true,
                        check_provided_dt_stability::Bool = true,
                        stability_tol::Real = 1e-6,
                        noise_amplitude::Real = 0.0,
                        noise_seed::Union{Nothing, Int} = nothing,
                        check_initial_data::Bool = true,
                        initial_data_tol = nothing,
                        verbose::Bool = false)
    T_final > 0 || throw(ArgumentError("`T_final` must be positive."))
    save_every > 0 || throw(ArgumentError("`save_every` must be positive."))
    state_layout in (:matrix, :vector) ||
        throw(ArgumentError("`state_layout` must be either `:matrix` or `:vector`."))

    r = Float64.(ops.r)
    n = length(r)
    n > 0 || throw(ArgumentError("Operator grid is empty."))

    ϕ_input = _pick_keyword(ϕ0, phi0, "ϕ0", "phi0")
    Π_input = _pick_keyword(Π0, pi0, "Π0", "pi0")
    Ξ_input = _pick_keyword(Ξ0, xi0, "Ξ0", "xi0")
    w_in_input = _pick_keyword(w_in0, win0, "w_in0", "win0")
    w_out_input = _pick_keyword(w_out0, wout0, "w_out0", "wout0")
    mode = _normalize_initial_data_mode(initial_data_mode)
    has_characteristics = (w_in_input !== nothing) || (w_out_input !== nothing)

    mode in (:auto, :potential, :characteristic) ||
        throw(ArgumentError("`initial_data_mode` must be :auto, :potential, or :characteristic."))
    if has_characteristics && mode === :auto
        mode = :characteristic
    end

    ϕ_init = _vector_from_input(ϕ_input, r, "ϕ0")
    Π_init = _vector_from_input(Π_input, r, "Π0")
    Ξ_init = _vector_from_input(Ξ_input, r, "Ξ0")

    if mode === :characteristic
        if Ξ_input !== nothing || Π_input !== nothing
            throw(ArgumentError("Do not pass `Π0`/`Ξ0` with characteristic initial data. Use `w_in0`,`w_out0` only."))
        end
        w_in_data = w_in_input === nothing ? zeros(Float64, n) : w_in_input
        w_out_data = w_out_input === nothing ? zeros(Float64, n) : w_out_input
        char_data = characteristic_initial_data(
                                          r;
                                          w_in0 = w_in_data,
                                          w_out0 = w_out_data,
                                          enforce_origin = enforce_origin
                                         )
        Π_init = char_data.Π0
        Ξ_init = char_data.Ξ0
        if ϕ_init === nothing
            ϕ_init = zeros(Float64, n)
        end
    elseif mode === :potential
        ϕ_init === nothing && throw(ArgumentError("`initial_data_mode=:potential` requires `ϕ0`."))
        Ξ_input === nothing || throw(ArgumentError("`initial_data_mode=:potential` does not accept explicit `Ξ0`; it is built as `Geven*ϕ0`."))
        if Π_init === nothing
            Π_init = zeros(Float64, n)
        end
        Ξ_init = Float64.(ops.Geven * ϕ_init)
    else
        if ϕ_init === nothing
            ϕ_init = default_wave_profile(r)
        end
        if Π_init === nothing
            Π_init = zeros(Float64, n)
        end
        if Ξ_init === nothing
            Ξ_init = Float64.(ops.Geven * ϕ_init)
        end
    end

    bc_norm = _normalize_boundary_condition(boundary_condition)
    initialize_wave_state!(Π_init, Ξ_init; enforce_origin = enforce_origin)

    initial_data_check_wave = check_wave_data_consistency(
                                                          Π_init,
                                                          Ξ_init;
                                                          boundary_condition = bc_norm,
                                                          enforce_origin = enforce_origin,
                                                          require_boundary = false,
                                                          tol = initial_data_tol
                                                         )
    potential_check = check_potential_consistency(
                                                  ops,
                                                  Π_init,
                                                  Ξ_init;
                                                  tol = initial_data_tol
                                                 )
    initial_data_check = merge(initial_data_check_wave, (potential = potential_check, mode = mode))
    if check_initial_data && !initial_data_check_wave.consistent
        throw(
              ArgumentError(
                            "Initial data inconsistent with constraints: " *
                            "origin_residual=$(initial_data_check_wave.origin_residual), " *
                            "boundary_residual=$(initial_data_check_wave.boundary_residual), " *
                            "tol=$(initial_data_check_wave.tol)"
                           )
             )
    end
    if check_initial_data && mode === :potential && !potential_check.residual_ok
        throw(
              ArgumentError(
                            "Potential initial data failed consistency check: " *
                            "residual_l2_abs=$(potential_check.residual_l2_abs), " *
                            "residual_l2_rel=$(potential_check.residual_l2_rel), " *
                            "tol=$(potential_check.tol)"
                           )
             )
    end

    dt_raw = dt === nothing ? estimate_wave_timestep(ops; alg = alg, safety_factor = safety_factor) : Float64(dt)
    dt_raw > 0 || throw(ArgumentError("`dt` must be positive."))
    stability_tol > 0 || throw(ArgumentError("`stability_tol` must be positive."))
    noise_amplitude >= 0 || throw(ArgumentError("`noise_amplitude` must be nonnegative."))

    nsteps = max(1, ceil(Int, Float64(T_final) / dt_raw))
    dt_step = Float64(T_final) / nsteps
    saveat = _build_saveat(dt_step, nsteps, save_every)

    params = WaveODEParams(ops;
                           boundary_condition = bc_norm,
                           enforce_origin = enforce_origin)

    dt_stability = nothing
    if dt !== nothing && check_provided_dt_stability
        dt_stability = _check_provided_dt_stability(
                                                  ops,
                                                  dt_step,
                                                  alg;
                                                  boundary_condition = bc_norm,
                                                  enforce_origin = enforce_origin,
                                                  stability_tol = Float64(stability_tol)
                                                 )
    end

    noise_cb = _step_noise_callback(
                                    noise_amplitude,
                                    n,
                                    state_layout,
                                    enforce_origin;
                                    noise_seed = noise_seed
                                   )

    sol = if state_layout === :vector
        U0_vec = vcat(Π_init, Ξ_init)
        if alg isa ImplicitMidpoint
            Jproto = wave_system_jac_prototype(ops; boundary_condition = bc_norm)
            f = ODEFunction(wave_system_ode_vec!; jac = wave_system_jac!, jac_prototype = Jproto)
            prob = ODEProblem(f, U0_vec, (0.0, Float64(T_final)), params)
            if noise_cb === nothing
                solve(prob,
                      ImplicitMidpoint(concrete_jac = true, linsolve = KLUFactorization());
                      dt = dt_step,
                      adaptive = false,
                      saveat = saveat,
                      save_start = true,
                      dense = false)
            else
                solve(prob,
                      ImplicitMidpoint(concrete_jac = true, linsolve = KLUFactorization());
                      dt = dt_step,
                      adaptive = false,
                      saveat = saveat,
                      save_start = true,
                      dense = false,
                      callback = noise_cb)
            end
        else
            prob = ODEProblem(wave_system_ode_vec!, U0_vec, (0.0, Float64(T_final)), params)
            if noise_cb === nothing
                solve(prob, alg;
                      dt = dt_step,
                      adaptive = false,
                      saveat = saveat,
                      save_start = true,
                      dense = false)
            else
                solve(prob, alg;
                      dt = dt_step,
                      adaptive = false,
                      saveat = saveat,
                      save_start = true,
                      dense = false,
                      callback = noise_cb)
            end
        end
    else
        U0_mat = hcat(Π_init, Ξ_init)
        prob = ODEProblem(wave_system_ode!, U0_mat, (0.0, Float64(T_final)), params)
        if noise_cb === nothing
            solve(prob, alg;
                  dt = dt_step,
                  adaptive = false,
                  saveat = saveat,
                  save_start = true,
                  dense = false)
        else
            solve(prob, alg;
                  dt = dt_step,
                  adaptive = false,
                  saveat = saveat,
                  save_start = true,
                  dense = false,
                  callback = noise_cb)
        end
    end

    nsave = length(sol.t)
    Π_hist = Matrix{Float64}(undef, n, nsave)
    Ξ_hist = Matrix{Float64}(undef, n, nsave)
    energy = Vector{Float64}(undef, nsave)

    if state_layout === :vector
        for k in 1:nsave
            Uk = sol.u[k]
            Πk = @view Uk[1:n]
            Ξk = @view Uk[(n + 1):(2 * n)]
            Π_hist[:, k] .= Πk
            Ξ_hist[:, k] .= Ξk
            energy[k] = Float64(wave_energy(ops, Πk, Ξk))
        end
    else
        for k in 1:nsave
            Uk = sol.u[k]
            Πk = @view Uk[:, 1]
            Ξk = @view Uk[:, 2]
            Π_hist[:, k] .= Πk
            Ξ_hist[:, k] .= Ξk
            energy[k] = Float64(wave_energy(ops, Πk, Ξk))
        end
    end

    t_hist = Float64.(sol.t)

    if verbose
        println("Wave solve completed: n=$(n), steps=$(nsteps), dt=$(dt_step), saved=$(nsave), bc=$(bc_norm), layout=$(state_layout)")
        println("  initial_data_mode = ", mode)
        println("  potential residual ||Geven*phi_hat - Xi||₂ = ",
                initial_data_check.potential.residual_l2_abs, " (rel ",
                initial_data_check.potential.residual_l2_rel, ")")
        println("  max|Geven*Pi0| = ", initial_data_check.potential.max_abs_GPi)
        if initial_data_check.potential.xi_growth_expected_from_pi
            println("  note: ", initial_data_check.potential.growth_explanation)
        end
        for msg in initial_data_check.potential.warnings
            println("  warning: ", msg)
        end
        if dt_stability !== nothing
            println("  dt spectral stability check: max|R(dt*λ)| = ", dt_stability.max_amplification,
                    ", ρ(L) = ", dt_stability.spectral_radius,
                    ", max Re(λ) = ", dt_stability.max_real_eigenvalue)
        end
        if noise_amplitude > 0
            println("  additive step noise amplitude = ", noise_amplitude,
                    ", noise seed = ", noise_seed === nothing ? "auto" : string(noise_seed))
        end
    end

    return WaveEvolutionResult(
                               t_hist,
                               Π_hist,
                               Ξ_hist,
                               energy,
                               r,
                               dt_step,
                               nsteps,
                               bc_norm,
                               initial_data_check
                              )
end

const solve_wave = solve_wave_ode

function _energy_behavior(sol::WaveEvolutionResult)
    dE = diff(sol.energy)
    return (
            E0 = sol.energy[1],
            Ef = sol.energy[end],
            Ef_over_E0 = sol.energy[end] / sol.energy[1],
            max_step_increase = isempty(dE) ? NaN : maximum(dE),
            max_step_decrease = isempty(dE) ? NaN : minimum(dE)
           )
end

"""
    benchmark_wave_integrators(ops; kwargs...)

Run a microbenchmark of `RK4()` vs `ImplicitMidpoint()` at the same fixed `dt` and
report runtime and energy behavior summaries.
"""
function benchmark_wave_integrators(ops::SphericalOperators;
                                    T_final::Real,
                                    dt::Real,
                                    boundary_condition::Symbol = :reflecting,
                                    ϕ0 = nothing,
                                    Π0 = nothing,
                                    Ξ0 = nothing,
                                    phi0 = nothing,
                                    pi0 = nothing,
                                    xi0 = nothing,
                                    save_every::Int = 1,
                                    enforce_origin::Bool = true,
                                    check_initial_data::Bool = true,
                                    initial_data_tol = nothing,
                                    warmup::Bool = true,
                                    verbose::Bool = true)
    T_final > 0 || throw(ArgumentError("`T_final` must be positive."))
    dt > 0 || throw(ArgumentError("`dt` must be positive."))

    common = (
              T_final = T_final,
              dt = dt,
              boundary_condition = boundary_condition,
              ϕ0 = ϕ0,
              Π0 = Π0,
              Ξ0 = Ξ0,
              phi0 = phi0,
              pi0 = pi0,
              xi0 = xi0,
              save_every = save_every,
              enforce_origin = enforce_origin,
              check_initial_data = check_initial_data,
              initial_data_tol = initial_data_tol,
              verbose = false
             )

    if warmup
        solve_wave_ode(ops; common..., T_final = min(T_final, dt), alg = RK4(), state_layout = :matrix)
        solve_wave_ode(ops; common..., T_final = min(T_final, dt), alg = ImplicitMidpoint(), state_layout = :vector)
    end

    elapsed_rk4 = @elapsed sol_rk4 = solve_wave_ode(
                                                 ops;
                                                 common...,
                                                 alg = RK4(),
                                                 state_layout = :matrix
                                                )
    elapsed_implicit = @elapsed sol_implicit = solve_wave_ode(
                                                           ops;
                                                           common...,
                                                           alg = ImplicitMidpoint(),
                                                           state_layout = :vector
                                                          )

    rk4_energy = _energy_behavior(sol_rk4)
    implicit_energy = _energy_behavior(sol_implicit)

    report = (
              config = (
                        T_final = Float64(T_final),
                        dt = Float64(dt),
                        boundary_condition = _normalize_boundary_condition(boundary_condition),
                        n = length(ops.r)
                       ),
              rk4 = (
                     elapsed_seconds = elapsed_rk4,
                     nsteps = sol_rk4.nsteps,
                     energy = rk4_energy
                    ),
              implicit_midpoint = (
                                   elapsed_seconds = elapsed_implicit,
                                   nsteps = sol_implicit.nsteps,
                                   energy = implicit_energy
                                  ),
              speedup_rk4_over_implicit = elapsed_implicit > 0 ? elapsed_rk4 / elapsed_implicit : NaN
             )

    if verbose
        println("Integrator microbenchmark")
        println("  config: n=", report.config.n, ", dt=", report.config.dt,
                ", T_final=", report.config.T_final, ", bc=", report.config.boundary_condition)
        println("  RK4")
        println("    elapsed = ", report.rk4.elapsed_seconds, " s, steps = ", report.rk4.nsteps)
        println("    E_final/E0 = ", report.rk4.energy.Ef_over_E0,
                ", max ΔE(step) = ", report.rk4.energy.max_step_increase)
        println("  ImplicitMidpoint(concrete_jac=true, linsolve=KLUFactorization())")
        println("    elapsed = ", report.implicit_midpoint.elapsed_seconds, " s, steps = ", report.implicit_midpoint.nsteps)
        println("    E_final/E0 = ", report.implicit_midpoint.energy.Ef_over_E0,
                ", max ΔE(step) = ", report.implicit_midpoint.energy.max_step_increase)
        println("  elapsed ratio RK4/Implicit = ", report.speedup_rk4_over_implicit)
    end

    return report
end

function _make_vector_data(data, r::Vector{Float64}, name::AbstractString)
    if data === nothing
        return nothing
    end
    values = data isa Function ? data(r) : data
    vec = Float64.(collect(values))
    length(vec) == length(r) ||
        throw(DimensionMismatch("`$name` length must match grid size $(length(r))."))
    return vec
end

@inline function _maxabs_vec(v::AbstractArray{<:Real})
    isempty(v) && return 0.0
    m = 0.0
    @inbounds for x in v
        ax = abs(Float64(x))
        if ax > m
            m = ax
        end
    end
    return m
end

function _maxabs_matrix_with_arg(A::AbstractMatrix{<:Real})
    best = -1.0
    bi = 1
    bj = 1
    @inbounds for j in axes(A, 2), i in axes(A, 1)
        av = abs(Float64(A[i, j]))
        if av > best
            best = av
            bi = i
            bj = j
        end
    end
    return (maxabs = max(0.0, best), i = bi, j = bj, value = A[bi, bj])
end

function _decode_block_index(idx::Int, n::Int, r::Vector{Float64})
    if idx <= n
        return (field = :Pi, block_index = idx, r_index = idx, r = r[idx])
    end
    ridx = idx - n
    return (field = :Xi, block_index = idx, r_index = ridx, r = r[ridx])
end

function _energy_phys(ops::SphericalOperators, Pi::AbstractVector, Xi::AbstractVector)
    return wave_energy(ops, Pi, Xi)
end

function _estimate_initial_profile_metrics(r::Vector{Float64}, Pi0::Vector{Float64})
    amp = maximum(abs.(Pi0))
    w = abs.(Pi0)
    denom = sum(w)
    center_est = denom > 0 ? sum(r .* w) / denom : 0.0
    width_est = denom > 0 ? sqrt(sum(((r .- center_est) .^ 2) .* w) / denom) : 0.0

    dx = length(r) > 1 ? minimum(diff(r)) : 1.0
    nyquist = pi / max(dx, 1e-14)
    k_proxy = _maxabs_vec(diff(Pi0)) / max(amp * dx, 1e-14)
    hf_ratio = k_proxy / max(nyquist, 1e-14)
    high_freq = hf_ratio > 0.25
    return (
            amplitude = amp,
            center_estimate = center_est,
            width_estimate = width_est,
            dx = dx,
            nyquist = nyquist,
            k_proxy = k_proxy,
            high_frequency_ratio = hf_ratio,
            high_frequency = high_freq
           )
end

function _run_wave_case_with_monitor(ops::SphericalOperators,
                                     Pi_seed::Vector{Float64},
                                     Xi_seed::Vector{Float64};
                                     T_final::Real,
                                     dt,
                                     alg,
                                     adaptive::Bool,
                                     reltol::Real,
                                     abstol::Real,
                                     save_every::Int,
                                     enforce_origin::Bool)
    Pi0 = copy(Pi_seed)
    Xi0 = copy(Xi_seed)
    initialize_wave_state!(Pi0, Xi0; enforce_origin = enforce_origin)
    xi1_after_init = Xi0[1]

    params = WaveODEParams(ops;
                           boundary_condition = :reflecting,
                           enforce_origin = enforce_origin)
    U0 = hcat(Pi0, Xi0)

    rhs_evals = Ref(0)
    max_abs_dxi1 = Ref(0.0)
    function rhs_wrapped!(dU, U, p, t)
        wave_system_ode!(dU, U, p, t)
        rhs_evals[] += 1
        if eltype(dU) <: AbstractFloat
            v = abs(dU[1, 2])
            if v > max_abs_dxi1[]
                max_abs_dxi1[] = v
            end
        end
        return nothing
    end

    prob = ODEProblem(rhs_wrapped!, U0, (0.0, Float64(T_final)), params)
    dt_used = dt === nothing ? estimate_wave_timestep(ops; alg = alg, safety_factor = 0.9) : Float64(dt)
    dt_stability = nothing

    if adaptive
        sol = solve(prob, alg;
                    adaptive = true,
                    reltol = Float64(reltol),
                    abstol = Float64(abstol),
                    save_start = true)
        t_hist = Float64.(sol.t)
        dt_internal = length(t_hist) >= 2 ? diff(t_hist) : Float64[]
        dt_info = (
                   adaptive = true,
                   dt_requested = dt,
                   dt_min = isempty(dt_internal) ? NaN : minimum(dt_internal),
                   dt_max = isempty(dt_internal) ? NaN : maximum(dt_internal),
                   dt_mean = isempty(dt_internal) ? NaN : sum(dt_internal) / length(dt_internal),
                   reltol = Float64(reltol),
                   abstol = Float64(abstol)
                  )
    else
        nsteps = max(1, ceil(Int, Float64(T_final) / dt_used))
        dt_step = Float64(T_final) / nsteps
        if dt !== nothing
            dt_stability = _check_provided_dt_stability(
                                                        ops,
                                                        dt_step,
                                                        alg;
                                                        boundary_condition = :reflecting,
                                                        enforce_origin = enforce_origin,
                                                        stability_tol = 1e-6,
                                                        throw_on_failure = false
                                                       )
        end
        saveat = _build_saveat(dt_step, nsteps, save_every)
        sol = solve(prob, alg;
                    dt = dt_step,
                    adaptive = false,
                    saveat = saveat,
                    save_start = true,
                    dense = false)
        t_hist = Float64.(sol.t)
        dt_info = (
                   adaptive = false,
                   dt_requested = dt,
                   dt_used = dt_step,
                   dt_stability = dt_stability,
                   reltol = NaN,
                   abstol = NaN
                  )
    end

    nsave = length(sol.t)
    n = length(ops.r)
    Pi_hist = Matrix{Float64}(undef, n, nsave)
    Xi_hist = Matrix{Float64}(undef, n, nsave)
    E = Vector{Float64}(undef, nsave)
    Ephys = Vector{Float64}(undef, nsave)

    for k in 1:nsave
        Uk = sol.u[k]
        Pik = @view Uk[:, 1]
        Xik = @view Uk[:, 2]
        Pi_hist[:, k] .= Pik
        Xi_hist[:, k] .= Xik
        E[k] = Float64(wave_energy(ops, Pik, Xik))
        Ephys[k] = Float64(_energy_phys(ops, Pik, Xik))
    end

    xi1 = Xi_hist[1, :]
    xi1_max = _maxabs_vec(xi1)
    xi1_exact_zero = all(x -> x == 0.0, xi1)

    return (
            t = t_hist,
            Pi = Pi_hist,
            Xi = Xi_hist,
            E = E,
            Ephys = Ephys,
            dt_info = dt_info,
            rhs_evals = rhs_evals[],
            max_abs_dxi1 = max_abs_dxi1[],
            xi1_after_init = xi1_after_init,
            xi1_max = xi1_max,
            xi1_exact_zero = xi1_exact_zero
           )
end

function _first_steps_energy_table(t::Vector{Float64}, E::Vector{Float64}; nshow::Int = 20)
    m = min(nshow, length(E))
    rows = NamedTuple[]
    for k in 1:m
        dE = k < length(E) ? E[k + 1] - E[k] : NaN
        push!(rows, (k = k, t = t[k], E = E[k], dE = dE))
    end
    return rows
end

function _early_delta_metrics(t::Vector{Float64}, E::Vector{Float64}, K::Int)
    if length(E) < 2
        return (
                K_used = 0,
                dE = Float64[],
                max_dE = NaN,
                max_abs_dE = NaN,
                min_dE = NaN,
                argmax_dE = 0,
                t_at_argmax_dE = NaN
               )
    end
    dE = diff(E)
    K_used = min(K, length(dE))
    dEk = dE[1:K_used]
    max_dE, arg = findmax(dEk)
    return (
            K_used = K_used,
            dE = dEk,
            max_dE = max_dE,
            max_abs_dE = _maxabs_vec(dEk),
            min_dE = minimum(dEk),
            argmax_dE = arg,
            t_at_argmax_dE = t[arg]
           )
end

function _dt_halving_study(ops::SphericalOperators,
                           Pi0::Vector{Float64},
                           Xi0::Vector{Float64};
                           T_final::Real,
                           dt::Real,
                           alg,
                           save_every::Int,
                           enforce_origin::Bool,
                           K::Int)
    dts = (Float64(dt), Float64(dt) / 2, Float64(dt) / 4)
    metrics = NamedTuple[]
    maxabs_vals = Float64[]
    for dti in dts
        run = _run_wave_case_with_monitor(
                                          ops,
                                          Pi0,
                                          Xi0;
                                          T_final = T_final,
                                          dt = dti,
                                          alg = alg,
                                          adaptive = false,
                                          reltol = 1e-9,
                                          abstol = 1e-12,
                                          save_every = save_every,
                                          enforce_origin = enforce_origin
                                         )
        early = _early_delta_metrics(run.t, run.E, K)
        push!(
              metrics,
              (
               dt = dti,
               max_abs_dE = early.max_abs_dE,
               max_dE = early.max_dE,
               min_dE = early.min_dE,
               K_used = early.K_used
              )
             )
        push!(maxabs_vals, early.max_abs_dE)
    end

    r1 = maxabs_vals[2] > 0 ? maxabs_vals[1] / maxabs_vals[2] : NaN
    r2 = maxabs_vals[3] > 0 ? maxabs_vals[2] / maxabs_vals[3] : NaN
    q1 = isfinite(r1) && r1 > 0 ? log2(r1) : NaN
    q2 = isfinite(r2) && r2 > 0 ? log2(r2) : NaN
    return (
            mode = :fixed_dt,
            metrics = metrics,
            ratio_dt_to_dt2 = r1,
            ratio_dt2_to_dt4 = r2,
            q_dt_to_dt2 = q1,
            q_dt2_to_dt4 = q2
           )
end

function _adaptive_tol_study(ops::SphericalOperators,
                             Pi0::Vector{Float64},
                             Xi0::Vector{Float64};
                             T_final::Real,
                             dt,
                             alg,
                             reltol::Real,
                             abstol::Real,
                             save_every::Int,
                             enforce_origin::Bool,
                             K::Int)
    tolerances = (
                  (Float64(reltol), Float64(abstol)),
                  (Float64(reltol) / 10, Float64(abstol) / 10),
                  (Float64(reltol) / 100, Float64(abstol) / 100)
                 )
    metrics = NamedTuple[]
    maxabs_vals = Float64[]
    for (rt, at) in tolerances
        run = _run_wave_case_with_monitor(
                                          ops,
                                          Pi0,
                                          Xi0;
                                          T_final = T_final,
                                          dt = dt,
                                          alg = alg,
                                          adaptive = true,
                                          reltol = rt,
                                          abstol = at,
                                          save_every = save_every,
                                          enforce_origin = enforce_origin
                                         )
        early = _early_delta_metrics(run.t, run.E, K)
        push!(
              metrics,
              (
               reltol = rt,
               abstol = at,
               max_abs_dE = early.max_abs_dE,
               max_dE = early.max_dE,
               min_dE = early.min_dE,
               K_used = early.K_used
              )
             )
        push!(maxabs_vals, early.max_abs_dE)
    end

    r1 = maxabs_vals[2] > 0 ? maxabs_vals[1] / maxabs_vals[2] : NaN
    r2 = maxabs_vals[3] > 0 ? maxabs_vals[2] / maxabs_vals[3] : NaN
    return (
            mode = :adaptive_tol,
            metrics = metrics,
            ratio_tol_to_tol10 = r1,
            ratio_tol10_to_tol100 = r2
           )
end

function _skew_adjointness_report(ops::SphericalOperators; bigfloat_check::Bool)
    n = length(ops.r)
    H = Matrix{Float64}(ops.H)
    D = Matrix{Float64}(ops.D)
    G = Matrix{Float64}(ops.Geven)
    Z = zeros(Float64, n, n)
    Hblk = [H Z; Z H]
    keep = setdiff(collect(1:(2 * n)), [1, n + 1])

    function summarize_S(S::Matrix{Float64})
        full = _maxabs_matrix_with_arg(S)
        S_no_origin = S[keep, keep]
        no_origin = _maxabs_matrix_with_arg(S_no_origin)
        no_origin_i = keep[no_origin.i]
        no_origin_j = keep[no_origin.j]
        rows_2end_max = _maxabs_vec(S[2:end, :])
        full_i_desc = _decode_block_index(full.i, n, Float64.(ops.r))
        full_j_desc = _decode_block_index(full.j, n, Float64.(ops.r))
        no_i_desc = _decode_block_index(no_origin_i, n, Float64.(ops.r))
        no_j_desc = _decode_block_index(no_origin_j, n, Float64.(ops.r))
        return (
                maxabs_full = full.maxabs,
                maxabs_rows_2end = rows_2end_max,
                maxabs_no_origin = no_origin.maxabs,
                argmax_full = (i = full.i, j = full.j, value = full.value, i_desc = full_i_desc, j_desc = full_j_desc),
                argmax_no_origin = (i = no_origin_i, j = no_origin_j, value = no_origin.value, i_desc = no_i_desc, j_desc = no_j_desc)
               )
    end

    A_interior = [Z D; G Z]
    S_interior = transpose(A_interior) * Hblk + Hblk * A_interior
    interior = summarize_S(S_interior)

    Sat = zeros(Float64, n, n)
    Sat[end, end] = -Float64(ops.B[end, end]) / Float64(ops.H[end, end])
    A_reflecting_sat = [Z D + Sat; G Z]
    S_reflecting_sat = transpose(A_reflecting_sat) * Hblk + Hblk * A_reflecting_sat
    reflecting_sat = summarize_S(S_reflecting_sat)

    sbp_residual = H * D + transpose(G) * H - Matrix{Float64}(ops.B)
    sbp_full = _maxabs_vec(sbp_residual)
    sbp_no_origin = _maxabs_vec(sbp_residual[2:end, :])

    bigfloat_part = if bigfloat_check
        Ab = BigFloat.(A_reflecting_sat)
        Hb = BigFloat.(Hblk)
        Sb = transpose(Ab) * Hb + Hb * Ab
        maxb = _maxabs_vec(Sb)
        (enabled = true, maxabs = maxb)
    else
        (enabled = false, maxabs = NaN)
    end

    return (
            n = n,
            interior = interior,
            reflecting_sat = reflecting_sat,
            sbp_full = sbp_full,
            sbp_no_origin = sbp_no_origin,
            bigfloat = bigfloat_part
           )
end

"""
    diagnose_reflecting_energy_bump(; kwargs...)

Diagnose early-time energy bumps for the reflecting SAT radial-wave setup.

This runs:
1. integrator/step metadata and early-step energy table,
2. dt-halving (fixed-step) or tolerance-scaling (adaptive) study,
3. semidiscrete skew-adjointness check (`A' Hblk + Hblk A`),
4. energy-definition comparison (`E` vs `Ephys`),
5. boundary-hit timing estimate vs bump timing,
6. parity-enforcement checks (`Xi[1]` and `dXi[1]` instrumentation).

Returns a `NamedTuple` and prints a concise diagnostics summary.
"""
function diagnose_reflecting_energy_bump(;
                                         source = MattssonNordström2004(),
                                         accuracy_order::Int = 6,
                                         N::Int = 64,
                                         R::Real = 1.0,
                                         p::Int = 2,
                                         mode = FastMode(),
                                         build_matrix::Symbol = :matrix_if_square,
                                         alg = RK4(),
                                         adaptive::Bool = false,
                                         dt = nothing,
                                         reltol::Real = 1e-8,
                                         abstol::Real = 1e-10,
                                         T_final::Real = 2.0,
                                         save_every::Int = 1,
                                         K::Int = 100,
                                         ϕ0 = nothing,
                                         Π0 = nothing,
                                         Ξ0 = nothing,
                                         center = nothing,
                                         width = nothing,
                                         amplitude::Real = 1.0,
                                         enforce_origin::Bool = true,
                                         bigfloat_check::Bool = false,
                                         verbose::Bool = true)
    ops = spherical_operators(
                              source;
                              accuracy_order = accuracy_order,
                              N = N,
                              R = R,
                              p = p,
                              mode = mode,
                              build_matrix = build_matrix
                             )
    r = Float64.(ops.r)

    Pi_seed = _make_vector_data(Π0, r, "Π0")
    Xi_seed = _make_vector_data(Ξ0, r, "Ξ0")
    phi_seed = _make_vector_data(ϕ0, r, "ϕ0")

    if Pi_seed === nothing
        if phi_seed !== nothing
            Pi_seed = phi_seed
        else
            if center === nothing && width === nothing
                Pi_seed = amplitude .* exp.(-0.5 .* (r .^ 2))
            else
                center_val = center === nothing ? 0.0 : Float64(center)
                width_val = width === nothing ? 1.0 : Float64(width)
                Pi_seed = amplitude .* exp.(-((r .- center_val) .^ 2) ./ (2 * width_val^2))
            end
        end
    end
    if Xi_seed === nothing
        Xi_seed = zeros(Float64, length(r))
    end

    profile = _estimate_initial_profile_metrics(r, Pi_seed)
    peak_center = r[argmax(abs.(Pi_seed))]
    center_for_hit = center === nothing ? peak_center : Float64(center)
    width_for_info = width === nothing ? profile.width_estimate : Float64(width)

    run_main = _run_wave_case_with_monitor(
                                           ops,
                                           Pi_seed,
                                           Xi_seed;
                                           T_final = T_final,
                                           dt = dt,
                                           alg = alg,
                                           adaptive = adaptive,
                                           reltol = reltol,
                                           abstol = abstol,
                                           save_every = save_every,
                                           enforce_origin = enforce_origin
                                          )

    early_E = _early_delta_metrics(run_main.t, run_main.E, K)
    early_Ephys = _early_delta_metrics(run_main.t, run_main.Ephys, K)
    first20 = _first_steps_energy_table(run_main.t, run_main.E; nshow = 20)

    scaling = if adaptive
        _adaptive_tol_study(
                            ops,
                            Pi_seed,
                            Xi_seed;
                            T_final = T_final,
                            dt = dt,
                            alg = alg,
                            reltol = reltol,
                            abstol = abstol,
                            save_every = save_every,
                            enforce_origin = enforce_origin,
                            K = K
                           )
    else
        dt_base = run_main.dt_info.adaptive ? (dt === nothing ? NaN : Float64(dt)) : run_main.dt_info.dt_used
        _dt_halving_study(
                          ops,
                          Pi_seed,
                          Xi_seed;
                          T_final = T_final,
                          dt = dt_base,
                          alg = alg,
                          save_every = save_every,
                          enforce_origin = enforce_origin,
                          K = K
                         )
    end

    skew = _skew_adjointness_report(ops; bigfloat_check = bigfloat_check)

    t_hit = max(0.0, Float64(R) - center_for_hit)
    bump_time = early_E.t_at_argmax_dE
    bump_pre_boundary = isfinite(t_hit) && isfinite(bump_time) && (bump_time < t_hit)

    parity = (
              xi1_after_init = run_main.xi1_after_init,
              xi1_max = run_main.xi1_max,
              xi1_exact_zero = run_main.xi1_exact_zero,
              dxi1_max = run_main.max_abs_dxi1,
              rhs_evals = run_main.rhs_evals,
              enforce_origin = enforce_origin
             )

    conclusions = String[]
    if parity.xi1_max > 1e-14 || parity.dxi1_max > 1e-14
        push!(conclusions, "Parity enforcement bypassed or numerically violated at origin.")
    else
        push!(conclusions, "Parity enforcement active: Xi[1] and dXi[1] remain at machine zero.")
    end

    if skew.reflecting_sat.maxabs_no_origin > 1e-10
        push!(conclusions, "Reflecting-SAT semidiscrete operator is not skew-adjoint away from origin; this can drive true energy drift.")
    else
        push!(conclusions, "Reflecting-SAT skew-adjointness holds away from origin (no-origin block near machine precision).")
    end

    if skew.interior.maxabs_no_origin > 1e-8 && skew.reflecting_sat.maxabs_no_origin <= 1e-10
        push!(conclusions, "Raw interior block has boundary flux (expected); reflecting SAT cancels it in the no-origin block.")
    end

    if skew.sbp_no_origin <= 1e-10 && skew.sbp_full > 1e-8
        push!(conclusions, "Energy defect is dominated by the unconstrained origin row (H[1,1]=0 degeneracy), not outer SAT.")
    end

    if scaling.mode == :fixed_dt && length(scaling.metrics) == 3
        if isfinite(scaling.q_dt_to_dt2) && isfinite(scaling.q_dt2_to_dt4) &&
           scaling.q_dt_to_dt2 > 1.0 && scaling.q_dt2_to_dt4 > 1.0
            push!(conclusions, "Integrator-induced energy wobble likely (early-step |ΔE| decreases under dt-halving).")
        else
            push!(conclusions, "Early-step |ΔE| does not strongly improve under dt-halving; inspect semidiscrete structure.")
        end
    elseif scaling.mode == :adaptive_tol
        if isfinite(scaling.ratio_tol_to_tol10) && scaling.ratio_tol_to_tol10 > 1.2
            push!(conclusions, "Integrator/tolerance contribution likely (|ΔE| decreases with tighter tolerances).")
        else
            push!(conclusions, "Tolerance tightening does not strongly reduce |ΔE|; semidiscrete effects may dominate.")
        end
    end

    push!(conclusions, "Wave speed is fixed to c=1; E and Ephys coincide.")

    if bump_pre_boundary
        push!(conclusions, "Bump occurs before boundary interaction time; likely interior/origin/integrator, not boundary reflection event.")
    else
        push!(conclusions, "Bump timing is not clearly pre-boundary; boundary interaction may contribute.")
    end

    integrator = (
                  algorithm = string(typeof(alg)),
                  adaptive = adaptive,
                  dt_info = run_main.dt_info,
                  reltol = adaptive ? Float64(reltol) : NaN,
                  abstol = adaptive ? Float64(abstol) : NaN
                 )
    initial_data = (
                    description = "Pi even, Xi odd-compatible; Xi initialized to zero unless provided.",
                    amplitude = profile.amplitude,
                    center_peak = peak_center,
                    center_l1 = profile.center_estimate,
                    center_used_for_hit = center_for_hit,
                    width = width_for_info,
                    high_frequency_ratio = profile.high_frequency_ratio,
                    high_frequency = profile.high_frequency
                   )
    boundary_timing = (
                      t_hit_estimate = t_hit,
                      bump_time = bump_time,
                      bump_pre_boundary = bump_pre_boundary
                     )
    energy_compare = (
                      max_abs_dE_E = early_E.max_abs_dE,
                      max_abs_dE_Ephys = early_Ephys.max_abs_dE,
                      max_dE_E = early_E.max_dE,
                      max_dE_Ephys = early_Ephys.max_dE,
                      min_dE_E = early_E.min_dE,
                      min_dE_Ephys = early_Ephys.min_dE
                     )

    report = (
              integrator = integrator,
              initial_data = initial_data,
              early_energy = (
                              first20 = first20,
                              K_used = early_E.K_used,
                              max_dE = early_E.max_dE,
                              max_abs_dE = early_E.max_abs_dE,
                              min_dE = early_E.min_dE,
                              t_at_max_dE = early_E.t_at_argmax_dE
                             ),
              scaling = scaling,
              skew_adjointness = skew,
              energy_definition = energy_compare,
              boundary_timing = boundary_timing,
              parity = parity,
              conclusions = conclusions
             )

    if verbose
        println("Diagnostics summary")
        println("  Integrator")
        println("    algorithm = ", report.integrator.algorithm)
        println("    adaptive = ", report.integrator.adaptive)
        if report.integrator.adaptive
            println("    dt range = [", report.integrator.dt_info.dt_min, ", ", report.integrator.dt_info.dt_max, "]")
            println("    reltol/abstol = ", report.integrator.reltol, " / ", report.integrator.abstol)
        else
            println("    fixed dt = ", report.integrator.dt_info.dt_used)
        end

        println("  Initial data")
        println("    amplitude = ", report.initial_data.amplitude)
        println("    center_peak = ", report.initial_data.center_peak,
                ", center_l1 = ", report.initial_data.center_l1,
                ", center_used_for_hit = ", report.initial_data.center_used_for_hit,
                ", width = ", report.initial_data.width)
        println("    high_frequency_ratio = ", report.initial_data.high_frequency_ratio,
                " (high_frequency = ", report.initial_data.high_frequency, ")")

        println("  First 20 energy rows (k, t, E, dE)")
        for row in report.early_energy.first20
            println("    ", row.k, "  ", row.t, "  ", row.E, "  ", row.dE)
        end
        println("  Early-step bump")
        println("    K_used = ", report.early_energy.K_used)
        println("    max dE = ", report.early_energy.max_dE, " at t = ", report.early_energy.t_at_max_dE)
        println("    max |dE| = ", report.early_energy.max_abs_dE, ", min dE = ", report.early_energy.min_dE)

        if report.scaling.mode == :fixed_dt
            println("  dt-halving study")
            for row in report.scaling.metrics
                println("    dt=", row.dt, " max|dE|=", row.max_abs_dE, " max dE=", row.max_dE, " min dE=", row.min_dE)
            end
            println("    ratios = ", report.scaling.ratio_dt_to_dt2, ", ", report.scaling.ratio_dt2_to_dt4)
            println("    q estimates = ", report.scaling.q_dt_to_dt2, ", ", report.scaling.q_dt2_to_dt4)
        else
            println("  tolerance-scaling study")
            for row in report.scaling.metrics
                println("    reltol=", row.reltol, " abstol=", row.abstol, " max|dE|=", row.max_abs_dE)
            end
            println("    ratios = ", report.scaling.ratio_tol_to_tol10, ", ", report.scaling.ratio_tol10_to_tol100)
        end

        println("  Skew-adjointness")
        println("    interior A: maxabs(S) = ", report.skew_adjointness.interior.maxabs_full)
        println("    interior A: maxabs(S rows 2:end) = ", report.skew_adjointness.interior.maxabs_rows_2end)
        println("    interior A: maxabs(S no-origin) = ", report.skew_adjointness.interior.maxabs_no_origin)
        println("    interior A: argmax full = ", report.skew_adjointness.interior.argmax_full)
        println("    interior A: argmax no-origin = ", report.skew_adjointness.interior.argmax_no_origin)
        println("    reflecting-SAT A: maxabs(S) = ", report.skew_adjointness.reflecting_sat.maxabs_full)
        println("    reflecting-SAT A: maxabs(S rows 2:end) = ", report.skew_adjointness.reflecting_sat.maxabs_rows_2end)
        println("    reflecting-SAT A: maxabs(S no-origin) = ", report.skew_adjointness.reflecting_sat.maxabs_no_origin)
        println("    reflecting-SAT A: argmax full = ", report.skew_adjointness.reflecting_sat.argmax_full)
        println("    reflecting-SAT A: argmax no-origin = ", report.skew_adjointness.reflecting_sat.argmax_no_origin)
        println("    SBP residual full/no-origin = ", report.skew_adjointness.sbp_full, " / ", report.skew_adjointness.sbp_no_origin)
        if report.skew_adjointness.bigfloat.enabled
            println("    BigFloat maxabs(S) = ", report.skew_adjointness.bigfloat.maxabs)
        end

        println("  Energy definition")
        println("    max|dE| = ", report.energy_definition.max_abs_dE_E,
                ", max|dEphys| = ", report.energy_definition.max_abs_dE_Ephys)

        println("  Boundary timing")
        println("    t_hit ≈ ", report.boundary_timing.t_hit_estimate,
                ", bump time = ", report.boundary_timing.bump_time,
                ", bump_pre_boundary = ", report.boundary_timing.bump_pre_boundary)

        println("  Parity enforcement")
        println("    Xi[1] after init = ", report.parity.xi1_after_init)
        println("    max|Xi[1]| = ", report.parity.xi1_max,
                ", exact_zero_saved = ", report.parity.xi1_exact_zero)
        println("    max|dXi[1]| = ", report.parity.dxi1_max,
                ", rhs_evals = ", report.parity.rhs_evals)

        println("  Conclusions")
        for cmsg in report.conclusions
            println("    - ", cmsg)
        end
    end

    return report
end

"""
    energy_rate(ops, Π, Ξ, dΠ, dΞ)

Semidiscrete energy rate

`dE/dt = Π' * H * dΠ + Ξ' * H * dΞ`.
"""
function energy_rate(ops::SphericalOperators,
                     Π::AbstractVector,
                     Ξ::AbstractVector,
                     dΠ::AbstractVector,
                     dΞ::AbstractVector)
    n = length(ops.r)
    length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    length(Ξ) == n || throw(DimensionMismatch("`Ξ` length must match grid size $(n)."))
    length(dΠ) == n || throw(DimensionMismatch("`dΠ` length must match grid size $(n)."))
    length(dΞ) == n || throw(DimensionMismatch("`dΞ` length must match grid size $(n)."))
    return dot(Π, ops.H * dΠ) + dot(Ξ, ops.H * dΞ)
end

@inline function _boundary_flux_term(ops::SphericalOperators, Π::AbstractVector, Ξ::AbstractVector)
    return Float64(ops.B[end, end]) * Float64(Π[end]) * Float64(Ξ[end])
end

function _runtime_rhs_energy_diagnostics(ops::SphericalOperators,
                                         params::WaveODEParams,
                                         Π::AbstractVector{<:Real},
                                         Ξ::AbstractVector{<:Real},
                                         t::Real;
                                         sbp_residual_matrix = nothing)
    n = length(ops.r)
    length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    length(Ξ) == n || throw(DimensionMismatch("`Ξ` length must match grid size $(n)."))

    Πf = Float64.(Π)
    Ξf = Float64.(Ξ)

    U = hcat(Πf, Ξf)
    dU = zeros(Float64, n, 2)
    wave_system_ode!(dU, U, params, t)
    dΠ = @view dU[:, 1]
    dΞ = @view dU[:, 2]

    dΠ_interior = Vector{Float64}(ops.D * Ξf)
    dΞ_interior = Vector{Float64}(ops.Geven * Πf)
    if params.enforce_origin
        dΞ_interior[1] = 0.0
    end

    dE = Float64(energy_rate(ops, Πf, Ξf, dΠ, dΞ))
    dE_interior = Float64(energy_rate(ops, Πf, Ξf, dΠ_interior, dΞ_interior))
    dE_sat = dE - dE_interior

    flux = _boundary_flux_term(ops, Πf, Ξf)
    R_sbp = sbp_residual_matrix === nothing ?
            (ops.H * ops.D + transpose(ops.Geven) * ops.H - ops.B) :
            sbp_residual_matrix
    sbp_residual_term = Float64(dot(Πf, R_sbp * Ξf))
    chars = boundary_characteristics(Πf[end], Ξf[end])
    w_in = Float64(chars.w_in)
    w_out = Float64(chars.w_out)
    win_minus_wout = w_in - w_out
    win_plus_wout = w_in + w_out

    return (
            dE = dE,
            dE_interior = dE_interior,
            dE_sat = dE_sat,
            flux = flux,
            interior_minus_flux = dE_interior - flux,
            sbp_residual_term = sbp_residual_term,
            interior_minus_flux_minus_sbp = (dE_interior - flux) - sbp_residual_term,
            total_plus_flux = dE + flux,
            total_minus_flux = dE - flux,
            w_in = w_in,
            w_out = w_out,
            win_minus_wout = win_minus_wout,
            win_plus_wout = win_plus_wout
           )
end

function _classify_energy_rate(values::Vector{Float64}; tol::Float64 = 1e-12)
    isempty(values) && return :empty
    vmin = minimum(values)
    vmax = maximum(values)
    if vmin >= -tol && vmax <= tol
        return :near_zero
    elseif vmin >= -tol && vmax > tol
        return :positive_bias
    elseif vmax <= tol && vmin < -tol
        return :negative_bias
    end
    return :oscillatory
end

function _run_reflecting_case_energy_diagnostics(ops::SphericalOperators;
                                                 Π0::Vector{Float64},
                                                 Ξ0::Vector{Float64},
                                                 dt::Float64,
                                                 T_final::Float64,
                                                 K::Int,
                                                 alg,
                                                 enforce_origin::Bool)
    Πseed = copy(Π0)
    Ξseed = copy(Ξ0)
    initialize_wave_state!(Πseed, Ξseed; enforce_origin = enforce_origin)

    sol = solve_wave_ode(
                         ops;
                         T_final = T_final,
                         dt = dt,
                         alg = alg,
                         boundary_condition = :reflecting,
                         Π0 = Πseed,
                         Ξ0 = Ξseed,
                         save_every = 1,
                         enforce_origin = enforce_origin,
                         check_initial_data = true,
                         verbose = false
                        )

    params = WaveODEParams(ops; boundary_condition = :reflecting, enforce_origin = enforce_origin)
    R_sbp = ops.H * ops.D + transpose(ops.Geven) * ops.H - ops.B
    K_used = min(K, length(sol.t))
    rows = NamedTuple[]
    dE_vals = Vector{Float64}(undef, K_used)
    dE_interior_vals = Vector{Float64}(undef, K_used)
    flux_vals = Vector{Float64}(undef, K_used)
    sbp_term_vals = Vector{Float64}(undef, K_used)
    sbp_formula_err_vals = Vector{Float64}(undef, K_used)
    win_vals = Vector{Float64}(undef, K_used)
    wout_vals = Vector{Float64}(undef, K_used)
    wm_vals = Vector{Float64}(undef, K_used)
    wp_vals = Vector{Float64}(undef, K_used)

    for k in 1:K_used
        Πk = @view sol.Π[:, k]
        Ξk = @view sol.Ξ[:, k]
        entry = _runtime_rhs_energy_diagnostics(
                                                ops,
                                                params,
                                                Πk,
                                                Ξk,
                                                sol.t[k];
                                                sbp_residual_matrix = R_sbp
                                               )

        dE_vals[k] = entry.dE
        dE_interior_vals[k] = entry.dE_interior
        flux_vals[k] = entry.flux
        sbp_term_vals[k] = entry.sbp_residual_term
        sbp_formula_err_vals[k] = entry.interior_minus_flux_minus_sbp
        win_vals[k] = entry.w_in
        wout_vals[k] = entry.w_out
        wm_vals[k] = entry.win_minus_wout
        wp_vals[k] = entry.win_plus_wout

        push!(
              rows,
              (
               k = k,
               t = sol.t[k],
               dE = entry.dE,
               dE_interior = entry.dE_interior,
               dE_sat = entry.dE_sat,
               flux = entry.flux,
               interior_minus_flux = entry.interior_minus_flux,
               sbp_residual_term = entry.sbp_residual_term,
               interior_minus_flux_minus_sbp = entry.interior_minus_flux_minus_sbp,
               total_plus_flux = entry.total_plus_flux,
               total_minus_flux = entry.total_minus_flux,
               w_in = entry.w_in,
               w_out = entry.w_out,
               win_minus_wout = entry.win_minus_wout,
               win_plus_wout = entry.win_plus_wout
              )
             )
    end

    dE_steps = diff(sol.energy)
    K_steps = min(K, length(dE_steps))
    dE_steps_early = K_steps > 0 ? dE_steps[1:K_steps] : Float64[]

    return (
            sol = sol,
            K_used = K_used,
            rows = rows,
            dE_vals = dE_vals,
            dE_interior_vals = dE_interior_vals,
            flux_vals = flux_vals,
            sbp_residual_term_vals = sbp_term_vals,
            sbp_formula_error_vals = sbp_formula_err_vals,
            w_in_vals = win_vals,
            w_out_vals = wout_vals,
            win_minus_wout_vals = wm_vals,
            win_plus_wout_vals = wp_vals,
            dE_sign = _classify_energy_rate(dE_vals),
            dE0 = isempty(dE_vals) ? NaN : dE_vals[1],
            max_abs_dE = isempty(dE_vals) ? NaN : _maxabs_vec(dE_vals),
            max_abs_dE_interior = isempty(dE_interior_vals) ? NaN : _maxabs_vec(dE_interior_vals),
            max_abs_flux = isempty(flux_vals) ? NaN : _maxabs_vec(flux_vals),
            max_abs_sbp_residual_term = isempty(sbp_term_vals) ? NaN : _maxabs_vec(sbp_term_vals),
            max_abs_interior_minus_flux_minus_sbp = isempty(sbp_formula_err_vals) ? NaN : _maxabs_vec(sbp_formula_err_vals),
            max_abs_interior_minus_flux = isempty(rows) ? NaN : _maxabs_vec(getfield.(rows, :interior_minus_flux)),
            max_abs_total_plus_flux = isempty(rows) ? NaN : _maxabs_vec(getfield.(rows, :total_plus_flux)),
            max_abs_total_minus_flux = isempty(rows) ? NaN : _maxabs_vec(getfield.(rows, :total_minus_flux)),
            max_abs_win_minus_wout = isempty(wm_vals) ? NaN : _maxabs_vec(wm_vals),
            max_abs_win_plus_wout = isempty(wp_vals) ? NaN : _maxabs_vec(wp_vals),
            max_step_energy_increase = isempty(dE_steps_early) ? NaN : maximum(dE_steps_early),
            min_step_energy_change = isempty(dE_steps_early) ? NaN : minimum(dE_steps_early)
           )
end

function _reflecting_constraint_diagnostics()
    rA = Float64(boundary_characteristic_residual([1.0], [0.0]; bc = :reflecting))
    rB = Float64(boundary_characteristic_residual([0.0], [1.0]; bc = :reflecting))
    a = 0.5 * (rA + rB)
    b = 0.5 * (rA - rB)

    kind = if isapprox(a, 1.0; atol = 1e-12, rtol = 0.0) && isapprox(b, -1.0; atol = 1e-12, rtol = 0.0)
        :w_in_minus_w_out
    elseif isapprox(a, 1.0; atol = 1e-12, rtol = 0.0) && isapprox(b, 1.0; atol = 1e-12, rtol = 0.0)
        :w_in_plus_w_out
    elseif isapprox(a, 1.0; atol = 1e-12, rtol = 0.0) && isapprox(b, 0.0; atol = 1e-12, rtol = 0.0)
        :w_in_only
    elseif isapprox(a, 0.0; atol = 1e-12, rtol = 0.0) && isapprox(b, 1.0; atol = 1e-12, rtol = 0.0)
        :w_out_only
    else
        :unknown_linear_combination
    end

    target = if kind === :w_in_minus_w_out
        "w_in - w_out = 0 (equivalently Xi(R)=0, i.e. w_in = +w_out)"
    elseif kind === :w_in_plus_w_out
        "w_in + w_out = 0 (equivalently Pi(R)=0, i.e. w_in = -w_out)"
    elseif kind === :w_in_only
        "w_in = 0 (incoming characteristic kill)"
    elseif kind === :w_out_only
        "w_out = 0 (outgoing characteristic kill)"
    else
        "unrecognized characteristic relation"
    end

    return (a = a, b = b, kind = kind, target_relation = target)
end

function _probe_runtime_operator_matrix(ops::SphericalOperators;
                                        boundary_condition::Symbol,
                                        enforce_origin::Bool)
    n = length(ops.r)
    m = 2 * n
    A = Matrix{Float64}(undef, m, m)
    e = zeros(Float64, m)
    f = zeros(Float64, m)
    params = WaveODEParams(ops;
                           boundary_condition = boundary_condition,
                           enforce_origin = enforce_origin)
    for j in 1:m
        fill!(e, 0.0)
        e[j] = 1.0
        wave_system_ode_vec!(f, e, params, 0.0)
        @inbounds A[:, j] .= f
    end
    return A
end

function _skew_report_from_runtime_A(ops::SphericalOperators,
                                     A_runtime::Matrix{Float64})
    n = length(ops.r)
    m = 2 * n
    H = Matrix{Float64}(ops.H)
    Z = zeros(Float64, n, n)
    Hblk = [H Z; Z H]
    S = transpose(A_runtime) * Hblk + Hblk * A_runtime

    full = _maxabs_matrix_with_arg(S)
    origin_idx = [1, n + 1]
    no_origin_idx = setdiff(collect(1:m), origin_idx)
    S_no_origin = S[no_origin_idx, no_origin_idx]
    no_origin = _maxabs_matrix_with_arg(S_no_origin)
    no_origin_i = no_origin_idx[no_origin.i]
    no_origin_j = no_origin_idx[no_origin.j]

    closure_right = max(0, ops.closure_width)
    r_safe = (2 <= n - closure_right) ? collect(2:(n - closure_right)) : Int[]
    safe_idx = vcat(r_safe, r_safe .+ n)
    safe = if isempty(safe_idx)
        (maxabs = NaN, i = 0, j = 0, value = NaN)
    else
        _maxabs_matrix_with_arg(S[safe_idx, safe_idx])
    end

    return (
            S = S,
            maxabs_full = full.maxabs,
            maxabs_no_origin = no_origin.maxabs,
            maxabs_safe = safe.maxabs,
            argmax_full = (i = full.i, j = full.j, value = full.value),
            argmax_no_origin = (
                                i = no_origin_i,
                                j = no_origin_j,
                                value = no_origin.value,
                                i_desc = _decode_block_index(no_origin_i, n, Float64.(ops.r)),
                                j_desc = _decode_block_index(no_origin_j, n, Float64.(ops.r))
                               ),
            origin_indices = origin_idx,
            no_origin_indices = no_origin_idx,
            safe_indices = safe_idx
           )
end

function _runtime_skew_energy_partition(case_data::NamedTuple,
                                        skew::NamedTuple)
    sol = case_data.sol
    K_used = case_data.K_used
    if K_used == 0
        return (
                K_used = 0,
                max_abs_rhs_minus_quadratic = NaN,
                max_abs_quadratic_no_origin = NaN,
                max_abs_quadratic_origin_coupling = NaN,
                rows = NamedTuple[]
               )
    end

    n = size(sol.Π, 1)
    S = skew.S
    origin_idx = skew.origin_indices
    no_origin_idx = skew.no_origin_indices
    S_no_origin = S[no_origin_idx, no_origin_idx]
    S_origin = S[origin_idx, origin_idx]

    U = zeros(Float64, 2 * n)
    rhs_minus_quad = Vector{Float64}(undef, K_used)
    quad_no_origin = Vector{Float64}(undef, K_used)
    quad_origin_coupling = Vector{Float64}(undef, K_used)
    rows = NamedTuple[]

    for k in 1:K_used
        @views U[1:n] .= sol.Π[:, k]
        @views U[(n + 1):(2 * n)] .= sol.Ξ[:, k]

        q_full = 0.5 * dot(U, S * U)
        U_no_origin = U[no_origin_idx]
        U_origin = U[origin_idx]
        q_no_origin = 0.5 * dot(U_no_origin, S_no_origin * U_no_origin)
        q_origin_only = 0.5 * dot(U_origin, S_origin * U_origin)
        q_origin_coupling = q_full - q_no_origin - q_origin_only
        rhs_err = case_data.dE_vals[k] - q_full

        rhs_minus_quad[k] = rhs_err
        quad_no_origin[k] = q_no_origin
        quad_origin_coupling[k] = q_origin_coupling

        push!(
              rows,
              (
               k = k,
               t = sol.t[k],
               dE_rhs = case_data.dE_vals[k],
               dE_quadratic = q_full,
               rhs_minus_quadratic = rhs_err,
               dE_quadratic_no_origin = q_no_origin,
               dE_quadratic_origin_coupling = q_origin_coupling
              )
             )
    end

    return (
            K_used = K_used,
            max_abs_rhs_minus_quadratic = _maxabs_vec(rhs_minus_quad),
            max_abs_quadratic_no_origin = _maxabs_vec(quad_no_origin),
            max_abs_quadratic_origin_coupling = _maxabs_vec(quad_origin_coupling),
            rows = rows
           )
end

function _bigfloat_reflecting_rate(ops::SphericalOperators,
                                   Π::AbstractVector{<:Real},
                                   Ξ::AbstractVector{<:Real};
                                   enforce_origin::Bool)
    Πb = BigFloat.(Π)
    Ξb = BigFloat.(Ξ)
    D = BigFloat.(Matrix(ops.D))
    G = BigFloat.(Matrix(ops.Geven))
    H = BigFloat.(Matrix(ops.H))
    dΠ = D * Ξb
    dΞ = G * Πb
    if enforce_origin && !isempty(dΞ)
        dΞ[1] = big"0"
    end
    BNN = BigFloat(ops.B[end, end])
    HNN = BigFloat(ops.H[end, end])
    dΠ[end] -= (BNN / HNN) * Ξb[end]
    dE = dot(Πb, H * dΠ) + dot(Ξb, H * dΞ)
    return dE
end

function _dt_scaling_reflecting_dE(ops::SphericalOperators;
                                   Π0::Vector{Float64},
                                   Ξ0::Vector{Float64},
                                   dt::Float64,
                                   T_final::Float64,
                                   K::Int,
                                   alg,
                                   enforce_origin::Bool)
    dts = (dt, dt / 2)
    rows = NamedTuple[]
    for dti in dts
        case = _run_reflecting_case_energy_diagnostics(
                                                       ops;
                                                       Π0 = Π0,
                                                       Ξ0 = Ξ0,
                                                       dt = dti,
                                                       T_final = T_final,
                                                       K = K,
                                                       alg = alg,
                                                       enforce_origin = enforce_origin
                                                      )
        push!(
              rows,
              (
               dt = dti,
               max_abs_dE_rhs = case.max_abs_dE,
               dE_sign = case.dE_sign,
               max_step_energy_increase = case.max_step_energy_increase
              )
             )
    end
    ratio = rows[2].max_abs_dE_rhs > 0 ? rows[1].max_abs_dE_rhs / rows[2].max_abs_dE_rhs : NaN
    return (rows = rows, ratio_dt_to_dt2 = ratio)
end

"""
    diagnose_reflecting_sat_energy_drift(; kwargs...)

Focused reflecting-BC diagnostic that checks whether observed energy drift is caused
by semidiscrete bias or by time integration.

Checks included:
1. direct semidiscrete `dE/dt` from the runtime RHS (`wave_system_ode!`),
2. interior flux vs SAT cancellation (`Π[end]*Ξ[end]*B[end,end]`),
3. characteristic behavior at the outer boundary (`w_in`, `w_out`, `w_in±w_out`),
4. runtime-probed block operator `A_runtime` and skew defect `S = A' Hblk + Hblk A`,
5. mismatch check between `A_runtime` and the analytic Jacobian path,
6. decomposition of `dE` into no-origin and origin-coupling contributions via `0.5*U'SU`,
7. initial-data potential consistency and expected `Ξ` growth via `max|Geven*Π0|`.

The routine runs two initial-data cases on the same operators:
- Gaussian: `Π = exp(-r^2/2)`, `Ξ = 0`,
- Compact bump: `Π = bumpb((r-center)/radius)`, `Ξ = 0`.

Returns a `NamedTuple` and prints a concise diagnostics summary.
"""
function diagnose_reflecting_sat_energy_drift(;
                                              source = MattssonNordström2004(),
                                              accuracy_order::Int = 6,
                                              N::Union{Int, Nothing} = nothing,
                                              R::Real = 16.0,
                                              dr::Real = 0.1,
                                              p::Int = 2,
                                              mode = FastMode(),
                                              build_matrix::Symbol = :matrix_if_square,
                                              dt = nothing,
                                              cfl::Real = 0.25,
                                              T_final::Real = 2.5,
                                              K::Int = 100,
                                              alg = ImplicitMidpoint(),
                                              enforce_origin::Bool = true,
                                              bump_radius::Real = 1.0,
                                              bump_center::Real = 0.0,
                                              run_dt_scaling::Bool = false,
                                              bigfloat_check::Bool = false,
                                              verbose::Bool = true)
    dr > 0 || throw(ArgumentError("`dr` must be positive."))
    cfl > 0 || throw(ArgumentError("`cfl` must be positive."))
    K > 0 || throw(ArgumentError("`K` must be positive."))
    T_final > 0 || throw(ArgumentError("`T_final` must be positive."))

    N_use = if N === nothing
        N_guess = round(Int, Float64(R) / Float64(dr))
        isapprox(N_guess * Float64(dr), Float64(R); atol = 1e-12, rtol = 1e-12) ||
            throw(ArgumentError("`R/dr` must be an integer when `N` is omitted. Got R=$R, dr=$dr."))
        N_guess
    else
        Int(N)
    end
    N_use > 0 || throw(ArgumentError("`N` must be positive."))

    dt_use = dt === nothing ? Float64(cfl) * Float64(dr) : Float64(dt)
    dt_use > 0 || throw(ArgumentError("`dt` must be positive."))
    T_use = max(Float64(T_final), Float64(K) * dt_use)

    ops = spherical_operators(
                              source;
                              accuracy_order = accuracy_order,
                              N = N_use,
                              R = R,
                              p = p,
                              mode = mode,
                              build_matrix = build_matrix
                             )
    r = Float64.(ops.r)
    n = length(r)

    Π0_gaussian = exp.(-0.5 .* (r .^ 2))
    Π0_bump = bumpb_profile(r; amplitude = 1.0, center = bump_center, radius = bump_radius)
    Ξ0 = zeros(Float64, n)

    case_gaussian = _run_reflecting_case_energy_diagnostics(
                                                     ops;
                                                     Π0 = Π0_gaussian,
                                                     Ξ0 = Ξ0,
                                                     dt = dt_use,
                                                     T_final = T_use,
                                                     K = K,
                                                     alg = alg,
                                                     enforce_origin = enforce_origin
                                                    )
    case_bump = _run_reflecting_case_energy_diagnostics(
                                                         ops;
                                                         Π0 = Π0_bump,
                                                         Ξ0 = Ξ0,
                                                         dt = dt_use,
                                                         T_final = T_use,
                                                         K = K,
                                                         alg = alg,
                                                         enforce_origin = enforce_origin
                                                        )

    potential_gaussian = check_potential_consistency(
                                                   ops,
                                                   case_gaussian.sol.Π[:, 1],
                                                   case_gaussian.sol.Ξ[:, 1]
                                                  )
    potential_bump = check_potential_consistency(
                                               ops,
                                               case_bump.sol.Π[:, 1],
                                               case_bump.sol.Ξ[:, 1]
                                              )

    constraint = _reflecting_constraint_diagnostics()

    A_runtime = _probe_runtime_operator_matrix(
                                                ops;
                                                boundary_condition = :reflecting,
                                                enforce_origin = enforce_origin
                                               )
    nstate = 2 * n
    J_runtime = zeros(Float64, nstate, nstate)
    wave_system_jac!(
                     J_runtime,
                     zeros(Float64, nstate),
                     WaveODEParams(ops; boundary_condition = :reflecting, enforce_origin = enforce_origin),
                     0.0
                    )

    A_diff = A_runtime .- J_runtime
    A_diff_report = _maxabs_matrix_with_arg(A_diff)
    skew = _skew_report_from_runtime_A(ops, A_runtime)
    skew_partition_gaussian = _runtime_skew_energy_partition(case_gaussian, skew)
    skew_partition_bump = _runtime_skew_energy_partition(case_bump, skew)

    bigfloat = if bigfloat_check
        (
         enabled = true,
         gaussian_t0_dE = _bigfloat_reflecting_rate(ops, case_gaussian.sol.Π[:, 1], case_gaussian.sol.Ξ[:, 1];
                                                     enforce_origin = enforce_origin),
         bump_t0_dE = _bigfloat_reflecting_rate(ops, case_bump.sol.Π[:, 1], case_bump.sol.Ξ[:, 1];
                                                 enforce_origin = enforce_origin)
        )
    else
        (enabled = false, gaussian_t0_dE = big"NaN", bump_t0_dE = big"NaN")
    end

    dt_scaling = if run_dt_scaling
        _dt_scaling_reflecting_dE(
                                  ops;
                                  Π0 = Π0_gaussian,
                                  Ξ0 = Ξ0,
                                  dt = dt_use,
                                  T_final = T_use,
                                  K = K,
                                  alg = alg,
                                  enforce_origin = enforce_origin
                                 )
    else
        (rows = NamedTuple[], ratio_dt_to_dt2 = NaN)
    end

    conclusions = String[]
    tol_energy = 1e-10
    tol_skew = 1e-10
    tol_consistency = 1e-10

    if constraint.kind === :w_in_minus_w_out || constraint.kind === :w_in_plus_w_out
        push!(conclusions, "Reflecting SAT enforces a no-flux characteristic relation ($(constraint.target_relation)).")
    else
        push!(conclusions, "Reflecting SAT is not enforcing no-flux; fix BC to w_in = ± w_out.")
    end

    if skew.maxabs_no_origin > tol_skew
        push!(conclusions, "Semidiscrete runtime operator is not skew-adjoint in the no-origin block; this can cause true energy drift.")
    else
        push!(conclusions, "Runtime-probed no-origin skew defect is near machine precision.")
    end

    if A_diff_report.maxabs > tol_consistency
        push!(conclusions, "Runtime RHS differs from analytic A used in skew check.")
    else
        push!(conclusions, "Runtime RHS matches analytic Jacobian operator (no hidden SAT/projection mismatch).")
    end

    max_abs_dE_semidiscrete = max(case_gaussian.max_abs_dE, case_bump.max_abs_dE)
    max_abs_flux_mismatch = max(case_gaussian.max_abs_interior_minus_flux, case_bump.max_abs_interior_minus_flux)
    max_abs_sbp_formula_mismatch = max(case_gaussian.max_abs_interior_minus_flux_minus_sbp,
                                       case_bump.max_abs_interior_minus_flux_minus_sbp)
    max_abs_rhs_minus_quadratic = max(skew_partition_gaussian.max_abs_rhs_minus_quadratic,
                                      skew_partition_bump.max_abs_rhs_minus_quadratic)
    max_abs_origin_coupling = max(skew_partition_gaussian.max_abs_quadratic_origin_coupling,
                                  skew_partition_bump.max_abs_quadratic_origin_coupling)
    if max_abs_sbp_formula_mismatch > tol_energy
        push!(conclusions, "SAT coefficient scaling mismatch (B/H factors) or flux-sign inconsistency detected.")
    elseif max_abs_flux_mismatch > tol_energy
        push!(conclusions, "Interior dE-flux mismatch is fully explained by the discrete SBP residual term.")
    end

    if max_abs_rhs_minus_quadratic > tol_consistency
        push!(conclusions, "Runtime dE from RHS does not match quadratic-form prediction from runtime-probed S.")
    end

    if max_abs_dE_semidiscrete <= tol_energy && skew.maxabs_no_origin <= tol_skew
        push!(conclusions, "Semidiscrete dE≈0; observed drift is likely a time-integrator artifact.")
    elseif max_abs_dE_semidiscrete > tol_energy && skew.maxabs_no_origin <= tol_skew
        if max_abs_origin_coupling > tol_energy
            push!(conclusions, "Nonzero semidiscrete dE is dominated by origin-coupling terms (Π[1] interactions), not reflecting SAT.")
        else
            push!(conclusions, "Nonzero semidiscrete dE detected from runtime RHS despite small skew defect; inspect boundary forcing/state consistency.")
        end
    end
    for wmsg in potential_gaussian.warnings
        push!(conclusions, "Gaussian initial-data warning: " * wmsg)
    end
    for wmsg in potential_bump.warnings
        push!(conclusions, "Bump initial-data warning: " * wmsg)
    end

    report = (
              config = (
                        accuracy_order = accuracy_order,
                        N = N_use,
                        R = Float64(R),
                        dr = Float64(dr),
                        p = p,
                        dt = dt_use,
                        cfl = Float64(cfl),
                        T_final = T_use,
                        K = K,
                        algorithm = string(typeof(alg))
                       ),
              cases = (
                       gaussian = (
                                   dE_t0 = case_gaussian.dE0,
                                   max_abs_dE = case_gaussian.max_abs_dE,
                                   dE_sign = case_gaussian.dE_sign,
                                   max_abs_dE_interior = case_gaussian.max_abs_dE_interior,
                                   max_abs_flux = case_gaussian.max_abs_flux,
                                   max_abs_sbp_residual_term = case_gaussian.max_abs_sbp_residual_term,
                                   max_abs_interior_minus_flux_minus_sbp = case_gaussian.max_abs_interior_minus_flux_minus_sbp,
                                   max_abs_interior_minus_flux = case_gaussian.max_abs_interior_minus_flux,
                                   max_abs_total_plus_flux = case_gaussian.max_abs_total_plus_flux,
                                   max_abs_total_minus_flux = case_gaussian.max_abs_total_minus_flux,
                                   max_abs_win_minus_wout = case_gaussian.max_abs_win_minus_wout,
                                   max_abs_win_plus_wout = case_gaussian.max_abs_win_plus_wout,
                                   max_abs_GPi0 = potential_gaussian.max_abs_GPi,
                                   xi_growth_expected_from_pi0 = potential_gaussian.xi_growth_expected_from_pi,
                                   potential_consistency = potential_gaussian,
                                   max_step_energy_increase = case_gaussian.max_step_energy_increase,
                                   min_step_energy_change = case_gaussian.min_step_energy_change,
                                   early_rows = case_gaussian.rows
                                  ),
                       bump = (
                               dE_t0 = case_bump.dE0,
                               max_abs_dE = case_bump.max_abs_dE,
                               dE_sign = case_bump.dE_sign,
                               max_abs_dE_interior = case_bump.max_abs_dE_interior,
                               max_abs_flux = case_bump.max_abs_flux,
                               max_abs_sbp_residual_term = case_bump.max_abs_sbp_residual_term,
                               max_abs_interior_minus_flux_minus_sbp = case_bump.max_abs_interior_minus_flux_minus_sbp,
                               max_abs_interior_minus_flux = case_bump.max_abs_interior_minus_flux,
                               max_abs_total_plus_flux = case_bump.max_abs_total_plus_flux,
                               max_abs_total_minus_flux = case_bump.max_abs_total_minus_flux,
                               max_abs_win_minus_wout = case_bump.max_abs_win_minus_wout,
                               max_abs_win_plus_wout = case_bump.max_abs_win_plus_wout,
                               max_abs_GPi0 = potential_bump.max_abs_GPi,
                               xi_growth_expected_from_pi0 = potential_bump.xi_growth_expected_from_pi,
                               potential_consistency = potential_bump,
                               max_step_energy_increase = case_bump.max_step_energy_increase,
                               min_step_energy_change = case_bump.min_step_energy_change,
                               early_rows = case_bump.rows
                              )
                      ),
              characteristic_constraint = constraint,
              operator_consistency = (
                                      runtime_vs_jacobian_maxabs = A_diff_report.maxabs,
                                      runtime_vs_jacobian_argmax = (i = A_diff_report.i, j = A_diff_report.j, value = A_diff_report.value),
                                      skew_runtime = skew,
                                      skew_energy_partition = (
                                                               gaussian = skew_partition_gaussian,
                                                               bump = skew_partition_bump
                                                              )
                                     ),
              dt_scaling = dt_scaling,
              bigfloat = bigfloat,
              conclusions = conclusions
             )

    if verbose
        println("Reflecting SAT energy-drift diagnostics")
        println("  config: acc=", report.config.accuracy_order, ", N=", report.config.N,
                ", R=", report.config.R, ", dr=", report.config.dr,
                ", dt=", report.config.dt, ", p=", report.config.p,
                ", alg=", report.config.algorithm)
        println("  reflecting characteristic form: ", report.characteristic_constraint.kind,
                "  (", report.characteristic_constraint.target_relation, ")")

        for (name, case_rep) in ((:gaussian, report.cases.gaussian), (:bump, report.cases.bump))
            println("  case=", name)
            println("    dE(t0) = ", case_rep.dE_t0, ", max|dE| (first K states) = ", case_rep.max_abs_dE,
                    ", sign = ", case_rep.dE_sign)
            println("    max|dE_interior - flux| = ", case_rep.max_abs_interior_minus_flux)
            println("    max|SBP residual term| = ", case_rep.max_abs_sbp_residual_term,
                    ", max|(dE_interior - flux) - SBP| = ", case_rep.max_abs_interior_minus_flux_minus_sbp)
            println("    max|dE + flux| = ", case_rep.max_abs_total_plus_flux,
                    ", max|dE - flux| = ", case_rep.max_abs_total_minus_flux)
            println("    max|w_in - w_out| = ", case_rep.max_abs_win_minus_wout,
                    ", max|w_in + w_out| = ", case_rep.max_abs_win_plus_wout)
            println("    max|G*Pi0| = ", case_rep.max_abs_GPi0,
                    ", xi_growth_expected_from_pi0 = ", case_rep.xi_growth_expected_from_pi0)
            println("    potential residual l2 abs/rel = ",
                    case_rep.potential_consistency.residual_l2_abs, " / ",
                    case_rep.potential_consistency.residual_l2_rel)
            for wmsg in case_rep.potential_consistency.warnings
                println("    warning: ", wmsg)
            end
            println("    max step ΔE = ", case_rep.max_step_energy_increase,
                    ", min step ΔE = ", case_rep.min_step_energy_change)

            nshow = min(6, length(case_rep.early_rows))
            println("    early rows (k, t, dE, flux, dE+flux, dE_interior-flux, SBP, w_in, w_out, w_in-w_out)")
            for i in 1:nshow
                row = case_rep.early_rows[i]
                println("      ", row.k, "  ", row.t, "  ", row.dE, "  ", row.flux,
                        "  ", row.total_plus_flux, "  ", row.interior_minus_flux, "  ",
                        row.sbp_residual_term, "  ", row.w_in, "  ", row.w_out, "  ", row.win_minus_wout)
            end
        end

        println("  runtime operator checks")
        println("    max|A_runtime - J_runtime| = ", report.operator_consistency.runtime_vs_jacobian_maxabs,
                " at ", report.operator_consistency.runtime_vs_jacobian_argmax)
        println("    max|S| full = ", report.operator_consistency.skew_runtime.maxabs_full)
        println("    max|S| no-origin = ", report.operator_consistency.skew_runtime.maxabs_no_origin)
        println("    max|S| safe = ", report.operator_consistency.skew_runtime.maxabs_safe)
        println("    max|dE_rhs - 0.5*U'SU| gaussian/bump = ",
                report.operator_consistency.skew_energy_partition.gaussian.max_abs_rhs_minus_quadratic, " / ",
                report.operator_consistency.skew_energy_partition.bump.max_abs_rhs_minus_quadratic)
        println("    max|origin-coupling dE term| gaussian/bump = ",
                report.operator_consistency.skew_energy_partition.gaussian.max_abs_quadratic_origin_coupling, " / ",
                report.operator_consistency.skew_energy_partition.bump.max_abs_quadratic_origin_coupling)

        if run_dt_scaling
            println("  dt scaling (gaussian)")
            for row in report.dt_scaling.rows
                println("    dt=", row.dt, "  max|dE_rhs|=", row.max_abs_dE_rhs,
                        "  dE_sign=", row.dE_sign, "  max step ΔE=", row.max_step_energy_increase)
            end
            println("    ratio max|dE_rhs|(dt)/max|dE_rhs|(dt/2) = ", report.dt_scaling.ratio_dt_to_dt2)
        end

        if report.bigfloat.enabled
            println("  BigFloat t=0 dE")
            println("    gaussian: ", report.bigfloat.gaussian_t0_dE)
            println("    bump:     ", report.bigfloat.bump_t0_dE)
        end

        println("  conclusions")
        for cmsg in report.conclusions
            println("    - ", cmsg)
        end
    end

    return report
end
