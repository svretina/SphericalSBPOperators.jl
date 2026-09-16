function _regular_left_moving_spherical_gaussian_parameters(; amplitude::Real = 1.0,
                                                            R::Real = 40.0,
                                                            d::Real = 2.0,
                                                            rc = nothing,
                                                            origin_tol::Real = 1.0e-12)
    d > 0 || throw(ArgumentError("`d` must be positive."))
    origin_tol > 0 || throw(ArgumentError("`origin_tol` must be positive."))

    Rf = Float64(R)
    rcf = rc === nothing ? 0.5 * Rf : Float64(rc)
    return (A = Float64(amplitude),
            R = Rf,
            d = Float64(d),
            rc = rcf,
            origin_tol = Float64(origin_tol))
end

@inline function regular_left_moving_spherical_gaussian_profile(x::Real;
                                                                amplitude::Real = 1.0,
                                                                R::Real = 40.0,
                                                                d::Real = 2.0,
                                                                rc = nothing,
                                                                origin_tol::Real = 1.0e-12)
    params = _regular_left_moving_spherical_gaussian_parameters(; amplitude = amplitude,
                                                                R = R,
                                                                d = d,
                                                                rc = rc,
                                                                origin_tol = origin_tol)
    xf = Float64(x)
    return params.A * exp(-((xf - params.rc)^2) / params.d^2)
end

@inline function regular_left_moving_spherical_gaussian_profile_prime(x::Real;
                                                                      amplitude::Real = 1.0,
                                                                      R::Real = 40.0,
                                                                      d::Real = 2.0,
                                                                      rc = nothing,
                                                                      origin_tol::Real = 1.0e-12)
    params = _regular_left_moving_spherical_gaussian_parameters(; amplitude = amplitude,
                                                                R = R,
                                                                d = d,
                                                                rc = rc,
                                                                origin_tol = origin_tol)
    return _regular_left_moving_spherical_gaussian_profile_prime(x, params)
end

@inline function _regular_left_moving_spherical_gaussian_profile_prime(x::Real, params)
    xf = Float64(x)
    return -2.0 * (xf - params.rc) / params.d^2 *
           regular_left_moving_spherical_gaussian_profile(xf;
                                                          amplitude = params.A,
                                                          R = params.R,
                                                          d = params.d,
                                                          rc = params.rc,
                                                          origin_tol = params.origin_tol)
end

@inline function regular_left_moving_spherical_gaussian_profile_second(x::Real;
                                                                       amplitude::Real = 1.0,
                                                                       R::Real = 40.0,
                                                                       d::Real = 2.0,
                                                                       rc = nothing,
                                                                       origin_tol::Real = 1.0e-12)
    params = _regular_left_moving_spherical_gaussian_parameters(; amplitude = amplitude,
                                                                R = R,
                                                                d = d,
                                                                rc = rc,
                                                                origin_tol = origin_tol)
    return _regular_left_moving_spherical_gaussian_profile_second(x, params)
end

@inline function _regular_left_moving_spherical_gaussian_profile_second(x::Real, params)
    xf = Float64(x)
    y = xf - params.rc
    return (4.0 * y^2 / params.d^4 - 2.0 / params.d^2) *
           regular_left_moving_spherical_gaussian_profile(xf;
                                                          amplitude = params.A,
                                                          R = params.R,
                                                          d = params.d,
                                                          rc = params.rc,
                                                          origin_tol = params.origin_tol)
end

function regular_left_moving_spherical_gaussian_phi_exact(t::Real, r::Real;
                                                          amplitude::Real = 1.0,
                                                          R::Real = 40.0,
                                                          d::Real = 2.0,
                                                          rc = nothing,
                                                          origin_tol::Real = 1.0e-12)
    params = _regular_left_moving_spherical_gaussian_parameters(; amplitude = amplitude,
                                                                R = R,
                                                                d = d,
                                                                rc = rc,
                                                                origin_tol = origin_tol)
    return _regular_left_moving_spherical_gaussian_phi_exact(t, r, params)
end

function _regular_left_moving_spherical_gaussian_phi_exact(t::Real, r::Real, params)
    tf = Float64(t)
    rf = Float64(r)
    if abs(rf) < params.origin_tol
        return 2.0 * _regular_left_moving_spherical_gaussian_profile_prime(tf, params)
    end
    return (regular_left_moving_spherical_gaussian_profile(rf + tf;
                                                           amplitude = params.A,
                                                           R = params.R,
                                                           d = params.d,
                                                           rc = params.rc,
                                                           origin_tol = params.origin_tol) -
            regular_left_moving_spherical_gaussian_profile(tf - rf;
                                                           amplitude = params.A,
                                                           R = params.R,
                                                           d = params.d,
                                                           rc = params.rc,
                                                           origin_tol = params.origin_tol)) / rf
end

function regular_left_moving_spherical_gaussian_pi_exact(t::Real, r::Real;
                                                         amplitude::Real = 1.0,
                                                         R::Real = 40.0,
                                                         d::Real = 2.0,
                                                         rc = nothing,
                                                         origin_tol::Real = 1.0e-12)
    params = _regular_left_moving_spherical_gaussian_parameters(; amplitude = amplitude,
                                                                R = R,
                                                                d = d,
                                                                rc = rc,
                                                                origin_tol = origin_tol)
    return _regular_left_moving_spherical_gaussian_pi_exact(t, r, params)
end

function _regular_left_moving_spherical_gaussian_pi_exact(t::Real, r::Real, params)
    tf = Float64(t)
    rf = Float64(r)
    if abs(rf) < params.origin_tol
        return 2.0 * _regular_left_moving_spherical_gaussian_profile_second(tf, params)
    end
    return (_regular_left_moving_spherical_gaussian_profile_prime(rf + tf, params) -
            _regular_left_moving_spherical_gaussian_profile_prime(tf - rf, params)) / rf
end

function regular_left_moving_spherical_gaussian_psi_exact(t::Real, r::Real;
                                                          amplitude::Real = 1.0,
                                                          R::Real = 40.0,
                                                          d::Real = 2.0,
                                                          rc = nothing,
                                                          origin_tol::Real = 1.0e-12)
    params = _regular_left_moving_spherical_gaussian_parameters(; amplitude = amplitude,
                                                                R = R,
                                                                d = d,
                                                                rc = rc,
                                                                origin_tol = origin_tol)
    return _regular_left_moving_spherical_gaussian_psi_exact(t, r, params)
end

function _regular_left_moving_spherical_gaussian_psi_exact(t::Real, r::Real, params)
    tf = Float64(t)
    rf = Float64(r)
    if abs(rf) < params.origin_tol
        return 0.0
    end

    num = regular_left_moving_spherical_gaussian_profile(rf + tf;
                                                         amplitude = params.A,
                                                         R = params.R,
                                                         d = params.d,
                                                         rc = params.rc,
                                                         origin_tol = params.origin_tol) -
          regular_left_moving_spherical_gaussian_profile(tf - rf;
                                                         amplitude = params.A,
                                                         R = params.R,
                                                         d = params.d,
                                                         rc = params.rc,
                                                         origin_tol = params.origin_tol)
    num_r = _regular_left_moving_spherical_gaussian_profile_prime(rf + tf, params) +
            _regular_left_moving_spherical_gaussian_profile_prime(tf - rf, params)
    return num_r / rf - num / rf^2
end

"""
    regular_left_moving_spherical_gaussian_solution(t, r; amplitude=1.0, R=40.0, d=2.0, rc=R/2)

Return the analytic wave fields for the regular left-moving spherical Gaussian pulse.
For scalar `r`, this returns `(Φ = ..., Π = ..., Ψ = ...)`. For vector `r`, it returns
vector-valued fields evaluated at each point.
"""
function regular_left_moving_spherical_gaussian_solution(t::Real,
                                                         r::Real;
                                                         amplitude::Real = 1.0,
                                                         R::Real = 40.0,
                                                         d::Real = 2.0,
                                                         rc = nothing,
                                                         origin_tol::Real = 1.0e-12)
    params = _regular_left_moving_spherical_gaussian_parameters(; amplitude = amplitude,
                                                                R = R,
                                                                d = d,
                                                                rc = rc,
                                                                origin_tol = origin_tol)
    return (Φ = _regular_left_moving_spherical_gaussian_phi_exact(t, r, params),
            Π = _regular_left_moving_spherical_gaussian_pi_exact(t, r, params),
            Ψ = _regular_left_moving_spherical_gaussian_psi_exact(t, r, params))
end

function regular_left_moving_spherical_gaussian_solution(t::Real,
                                                         r::AbstractVector;
                                                         amplitude::Real = 1.0,
                                                         R::Real = 40.0,
                                                         d::Real = 2.0,
                                                         rc = nothing,
                                                         origin_tol::Real = 1.0e-12)
    params = _regular_left_moving_spherical_gaussian_parameters(; amplitude = amplitude,
                                                                R = R,
                                                                d = d,
                                                                rc = rc,
                                                                origin_tol = origin_tol)
    rf = Float64.(r)
    Φ = Vector{Float64}(undef, length(rf))
    Π = Vector{Float64}(undef, length(rf))
    Ψ = Vector{Float64}(undef, length(rf))
    for i in eachindex(rf)
        Φ[i] = _regular_left_moving_spherical_gaussian_phi_exact(t, rf[i], params)
        Π[i] = _regular_left_moving_spherical_gaussian_pi_exact(t, rf[i], params)
        Ψ[i] = _regular_left_moving_spherical_gaussian_psi_exact(t, rf[i], params)
    end
    return (Φ = Φ, Π = Π, Ψ = Ψ)
end
