function _even_gaussian_pi_zero_psi_parameters(; amplitude::Real = 1.0,
                                               r0::Real = 5.0,
                                               d::Real = 2.0)
    d > 0 || throw(ArgumentError("`d` must be positive."))
    return (A = Float64(amplitude), r0 = Float64(r0), d = Float64(d))
end

@inline function even_gaussian_pi_zero_psi_profile(x::Real;
                                                   amplitude::Real = 1.0,
                                                   r0::Real = 5.0,
                                                   d::Real = 2.0)
    params = _even_gaussian_pi_zero_psi_parameters(; amplitude = amplitude, r0 = r0, d = d)
    return params.A * (exp(-((Float64(x) - params.r0)^2) / params.d^2) +
                       exp(-((Float64(x) + params.r0)^2) / params.d^2))
end

@inline function _even_gaussian_pi_zero_psi_F(x::Real, params)
    xf = Float64(x)
    return xf * even_gaussian_pi_zero_psi_profile(xf;
                                                  amplitude = params.A,
                                                  r0 = params.r0,
                                                  d = params.d)
end

function _even_gaussian_pi_zero_psi_Fprime(x::Real, params)
    xf = Float64(x)
    Ep = exp(-((xf - params.r0)^2) / params.d^2)
    Em = exp(-((xf + params.r0)^2) / params.d^2)
    gp = params.A * (Ep + Em)
    dgp = params.A * (Ep * (-2 * (xf - params.r0) / params.d^2) +
                      Em * (-2 * (xf + params.r0) / params.d^2))
    return gp + xf * dgp
end

function _even_gaussian_pi_zero_psi_primitive_F(x::Real, params)
    xf = Float64(x)
    term1 = -(params.d^2 / 2) * exp(-((xf - params.r0)^2) / params.d^2)
    term2 = (params.r0 * params.d * sqrt(pi) / 2) * erf((xf - params.r0) / params.d)
    term3 = -(params.d^2 / 2) * exp(-((xf + params.r0)^2) / params.d^2)
    term4 = -(params.r0 * params.d * sqrt(pi) / 2) * erf((xf + params.r0) / params.d)
    return params.A * (term1 + term2 + term3 + term4)
end

@inline function _even_gaussian_pi_zero_psi_integral_F(a::Real, b::Real, params)
    return _even_gaussian_pi_zero_psi_primitive_F(b, params) -
           _even_gaussian_pi_zero_psi_primitive_F(a, params)
end

function _even_gaussian_pi_zero_psi_pi(t::Real, r::Real, params)
    tf = Float64(t)
    rf = Float64(r)
    if abs(rf) < 1e-12
        return _even_gaussian_pi_zero_psi_Fprime(tf, params)
    end
    return (_even_gaussian_pi_zero_psi_F(rf + tf, params) +
            _even_gaussian_pi_zero_psi_F(rf - tf, params)) / (2 * rf)
end

function _even_gaussian_pi_zero_psi_psi(t::Real, r::Real, params)
    tf = Float64(t)
    rf = Float64(r)
    if abs(rf) < 1e-12
        return 0.0
    end
    u_r_over_r = (_even_gaussian_pi_zero_psi_F(rf + tf, params) -
                  _even_gaussian_pi_zero_psi_F(rf - tf, params)) / (2 * rf)
    u_over_r2 = _even_gaussian_pi_zero_psi_integral_F(rf - tf, rf + tf, params) /
                (2 * rf^2)
    return u_r_over_r - u_over_r2
end

"""
    even_gaussian_pi_zero_psi_solution(t, r; amplitude=1.0, r0=5.0, d=2.0)

Return the analytic wave fields for the `:even_gaussian_pi_zero_psi` initial data.
For scalar `r`, this returns `(Π = ..., Ψ = ...)`. For vector `r`, it returns
vector-valued fields evaluated at each point.
"""
function even_gaussian_pi_zero_psi_solution(t::Real,
                                            r::Real;
                                            amplitude::Real = 1.0,
                                            r0::Real = 5.0,
                                            d::Real = 2.0)
    params = _even_gaussian_pi_zero_psi_parameters(; amplitude = amplitude, r0 = r0, d = d)
    return (Π = _even_gaussian_pi_zero_psi_pi(t, r, params),
            Ψ = _even_gaussian_pi_zero_psi_psi(t, r, params))
end

function even_gaussian_pi_zero_psi_solution(t::Real,
                                            r::AbstractVector;
                                            amplitude::Real = 1.0,
                                            r0::Real = 5.0,
                                            d::Real = 2.0)
    params = _even_gaussian_pi_zero_psi_parameters(; amplitude = amplitude, r0 = r0, d = d)
    rf = Float64.(r)
    Π = Vector{Float64}(undef, length(rf))
    Ψ = Vector{Float64}(undef, length(rf))
    for i in eachindex(rf)
        Π[i] = _even_gaussian_pi_zero_psi_pi(t, rf[i], params)
        Ψ[i] = _even_gaussian_pi_zero_psi_psi(t, rf[i], params)
    end
    return (Π = Π, Ψ = Ψ)
end
