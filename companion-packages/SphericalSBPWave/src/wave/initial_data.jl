"""
    default_wave_profile(r; amplitude=1.0, center=nothing, width=nothing)

Default smooth initial scalar profile on `[0, R]` used by `solve_wave_ode`.
"""
function default_wave_profile(r::AbstractVector; amplitude::Real = 1.0, center = nothing,
                              width = nothing)
    n = length(r)
    n == 0 && throw(ArgumentError("`r` must be non-empty."))

    rf = Float64.(r)
    R = rf[end]
    center_val = center === nothing ? 0.35 * R : Float64(center)
    width_val = width === nothing ? max(1.0e-12, 0.08 * max(R, 1.0)) : Float64(width)

    return amplitude .* exp.(-((rf .- center_val) .^ 2) ./ (2 * width_val^2))
end

"""
    bumpb(x)

Compact-support smooth bump function

`Ψ(x) = exp(1 / (x^2 - 1))` for `|x| < 1`, and `0` for `|x| ≥ 1`.
"""
@inline function bumpb(x::Real)
    ax = abs(x)
    if ax < one(x)
        return exp(inv(x * x - one(x)))
    end
    return zero(x)
end

"""
    bumpb_profile(r; amplitude=1.0, center=0.0, radius=1.0)

Evaluate a shifted/scaled compact-support bump profile on grid `r`:

`Π(r) = amplitude * bumpb((r - center)/radius)`.

For parity-compatible even `Π` on `[0,R]`, keep `center=0.0`.
"""
function bumpb_profile(r::AbstractVector;
                       amplitude::Real = 1.0,
                       center::Real = 0.0,
                       radius::Real = 1.0)
    radius > 0 || throw(ArgumentError("`radius` must be positive."))
    rf = Float64.(r)
    ξ = (rf .- Float64(center)) ./ Float64(radius)
    return Float64(amplitude) .* bumpb.(ξ)
end

function _profile_vector(data, r::AbstractVector, name::AbstractString)
    values = data isa Function ? data(r) : data
    vec = Float64.(collect(values))
    length(vec) == length(r) ||
        throw(DimensionMismatch("`$name` length must match grid size $(length(r))."))
    return vec
end

"""
    characteristic_initial_data(r; w_in0, w_out0, enforce_origin=true)

Construct wave initial data from characteristic profiles:

- `pi0 = 0.5 * (w_in0 + w_out0)`
- `psi0 = 0.5 * (w_in0 - w_out0)`

`w_in0` and `w_out0` may be vectors or callables evaluated on `r`.
When `enforce_origin=true`, odd-origin parity is imposed via `psi0[1]=0`.
"""
function characteristic_initial_data(r::AbstractVector;
                                     w_in0,
                                     w_out0,
                                     enforce_origin::Bool = true,
                                     has_origin_node::Bool = true)
    n = length(r)
    n > 0 || throw(ArgumentError("`r` must be non-empty."))

    w_in = _profile_vector(w_in0, r, "w_in0")
    w_out = _profile_vector(w_out0, r, "w_out0")

    Π0 = 0.5 .* (w_in .+ w_out)
    Ψ0 = 0.5 .* (w_in .- w_out)
    apply_symmetry_state!(Π0,
                          Ψ0;
                          enforce_origin = enforce_origin,
                          has_origin_node = has_origin_node)

    return (pi0 = Π0, psi0 = Ψ0, Π0 = Π0, Ψ0 = Ψ0, w_in0 = w_in, w_out0 = w_out)
end

"""
    check_potential_consistency(ops, Π, Ψ; tol=nothing, gauge=:origin)

Check consistency of a wave state with the potential constraint `Ψ ≈ Geven * ϕ`.

The routine reconstructs `ϕ̂` by least squares and reports absolute/relative residuals
for `Geven*ϕ̂ - Ψ`. It also reports `max|Geven*Π|`, which is the immediate source term
for `Ψ_t`; thus non-constant `Π` produces immediate `Ψ` growth through `Ψ_t = Geven*Π`.
"""
function check_potential_consistency(ops::WaveOperators,
                                     Π::AbstractVector,
                                     Ψ::AbstractVector;
                                     tol = nothing,
                                     gauge::Symbol = :origin)
    n = length(ops.r)
    length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    length(Ψ) == n || throw(DimensionMismatch("`Ψ` length must match grid size $(n)."))
    n > 0 || throw(ArgumentError("State vectors must be non-empty."))

    gauge in (:origin, :mean) ||
        throw(ArgumentError("`gauge` must be :origin or :mean."))

    Πf = Float64.(Π)
    Ψf = Float64.(Ψ)
    G = Matrix{Float64}(ops.Geven)

    ϕ̂ = zeros(Float64, n)
    if gauge === :origin
        if n >= 2
            A = @view G[:, 2:n]
            ϕ̂[2:n] .= A \ Ψf
        end
    else
        A = [G; fill(1.0 / n, 1, n)]
        b = vcat(Ψf, 0.0)
        ϕ̂ .= A \ b
    end

    Ψ_fit = G * ϕ̂
    resid = Ψ_fit .- Ψf

    resid_l2_abs = norm(resid)
    resid_linf_abs = isempty(resid) ? 0.0 : maximum(abs.(resid))
    psi_l2 = norm(Ψf)
    psi_linf = isempty(Ψf) ? 0.0 : maximum(abs.(Ψf))
    resid_l2_rel = resid_l2_abs / max(psi_l2, eps(Float64))
    resid_linf_rel = resid_linf_abs / max(psi_linf, eps(Float64))

    tol_use = tol === nothing ? 1.0e-10 : Float64(tol)
    residual_ok = resid_l2_abs <= tol_use || resid_l2_rel <= tol_use

    GΠ = Vector{Float64}(ops.Geven * Πf)
    max_abs_GPi = isempty(GΠ) ? 0.0 : maximum(abs.(GΠ))
    growth_tol = max(tol_use, 100 * eps(Float64))
    psi_growth_expected_from_pi = max_abs_GPi > growth_tol

    psi_near_zero = psi_linf <= growth_tol
    pi_range = isempty(Πf) ? 0.0 : (maximum(Πf) - minimum(Πf))
    pi_spatially_constant = pi_range <= growth_tol

    warnings = String[]
    if psi_near_zero && !pi_spatially_constant
        push!(warnings,
              "This initial state is inconsistent with Ψ=φ_r for any scalar φ; you are evolving a general first-order system and may excite constraint-violating modes.")
    end

    return (gauge = gauge,
            phi_hat = ϕ̂,
            residual_l2_abs = resid_l2_abs,
            residual_l2_rel = resid_l2_rel,
            residual_linf_abs = resid_linf_abs,
            residual_linf_rel = resid_linf_rel,
            residual_ok = residual_ok,
            tol = tol_use,
            psi_norm_l2 = psi_l2,
            psi_norm_linf = psi_linf,
            max_abs_GPi = max_abs_GPi,
            psi_growth_expected_from_pi = psi_growth_expected_from_pi,
            psi_near_zero = psi_near_zero,
            pi_spatially_constant = pi_spatially_constant,
            warnings = warnings,
            growth_explanation = "Ψ appears immediately if Π0 varies because Ψ_t = G Π.",
            note = "Π can be prescribed independently as φ_t; Ψ_t = Geven*Π couples Π variations into Ψ immediately.")
end
