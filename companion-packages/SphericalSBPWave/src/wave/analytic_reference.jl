function _analytic_wave_solution_free(time::Real, r, reference::NamedTuple)
    kind = reference.kind
    if kind === :even_gaussian_pi_zero_psi
        return even_gaussian_pi_zero_psi_solution(time, r;
                                                  amplitude = reference.amplitude,
                                                  r0 = reference.r0,
                                                  d = reference.d)
    elseif kind === :regular_left_moving_spherical_gaussian
        return regular_left_moving_spherical_gaussian_solution(time, r;
                                                               amplitude = reference.amplitude,
                                                               R = reference.R,
                                                               d = reference.d,
                                                               rc = reference.rc)
    end

    throw(ArgumentError("No analytic solution dispatcher is defined for initial-data kind `$kind`."))
end

function _analytic_boundary_state(exact, r)
    if r isa AbstractVector
        isempty(r) && throw(ArgumentError("`r` must be non-empty."))
        ΠN = exact.Π[end]
        ΨN = exact.Ψ[end]
        rN = r[end]
    else
        ΠN = exact.Π
        ΨN = exact.Ψ
        rN = r
    end

    return (ΠN = Float64(ΠN), ΨN = Float64(ΨN), rN = Float64(rN))
end

function analytic_wave_solution(time::Real,
                                r,
                                reference::NamedTuple;
                                validate_boundary::Bool = true,
                                boundary_atol::Real = 1.0e-8,
                                boundary_rtol::Real = 1.0e-8)
    boundary_atol >= 0 || throw(ArgumentError("`boundary_atol` must be non-negative."))
    boundary_rtol >= 0 || throw(ArgumentError("`boundary_rtol` must be non-negative."))

    exact = _analytic_wave_solution_free(time, r, reference)
    bc = hasproperty(reference, :boundary_condition) ? getproperty(reference, :boundary_condition) :
         :none
    bc_norm = _normalize_boundary_condition(Symbol(bc))

    state = _analytic_boundary_state(exact, r)
    residual = abs(boundary_characteristic_residual([state.ΠN], [state.ΨN]; bc = bc_norm))
    scale = max(abs(state.ΠN), abs(state.ΨN), 1.0)
    tol = Float64(boundary_atol) + Float64(boundary_rtol) * scale
    boundary_ok = bc_norm === :none || residual <= tol

    if validate_boundary && !boundary_ok
        throw(ArgumentError("The available analytic solution for initial-data kind `$(reference.kind)` " *
                            "does not satisfy boundary_condition=`$bc_norm` at t=$(Float64(time)) " *
                            "and r=R=$(state.rN): residual=$(residual), tolerance=$(tol). " *
                            "A boundary-aware closed-domain analytic solution has not been implemented " *
                            "for this case yet."))
    end

    return merge(exact,
                 (kind = reference.kind,
                  boundary_condition = bc_norm,
                  boundary_residual = residual,
                  boundary_tolerance = tol,
                  boundary_ok = boundary_ok))
end

function analytic_wave_solution(time::Real,
                                r;
                                initial_data_kind::Symbol,
                                boundary_condition::Symbol = :none,
                                amplitude::Real = 1.0,
                                r0::Real = 5.0,
                                d::Real = 2.0,
                                R::Real = 40.0,
                                rc = nothing,
                                validate_boundary::Bool = true,
                                boundary_atol::Real = 1.0e-8,
                                boundary_rtol::Real = 1.0e-8)
    reference = if initial_data_kind === :even_gaussian_pi_zero_psi
        (kind = initial_data_kind,
         amplitude = Float64(amplitude),
         r0 = Float64(r0),
         d = Float64(d),
         boundary_condition = boundary_condition)
    elseif initial_data_kind === :regular_left_moving_spherical_gaussian
        (kind = initial_data_kind,
         amplitude = Float64(amplitude),
         R = Float64(R),
         d = Float64(d),
         rc = rc === nothing ? 0.5 * Float64(R) : Float64(rc),
         boundary_condition = boundary_condition)
    else
        throw(ArgumentError("Unsupported initial-data kind `$initial_data_kind` for analytic wave solution."))
    end

    return analytic_wave_solution(time, r, reference;
                                  validate_boundary = validate_boundary,
                                  boundary_atol = boundary_atol,
                                  boundary_rtol = boundary_rtol)
end
