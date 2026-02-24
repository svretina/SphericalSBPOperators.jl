"""
    default_wave_profile(r; amplitude=1.0, center=nothing, width=nothing)

Default smooth initial scalar profile on `[0, R]` used by `solve_wave_ode`.
"""
function default_wave_profile(r::AbstractVector; amplitude::Real = 1.0, center = nothing, width = nothing)
    n = length(r)
    n == 0 && throw(ArgumentError("`r` must be non-empty."))

    rf = Float64.(r)
    R = rf[end]
    center_val = center === nothing ? 0.35 * R : Float64(center)
    width_val = width === nothing ? max(1e-12, 0.08 * max(R, 1.0)) : Float64(width)

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

- `Π0 = 0.5 * (w_in0 + w_out0)`
- `Ξ0 = 0.5 * (w_in0 - w_out0)`

`w_in0` and `w_out0` may be vectors or callables evaluated on `r`.
When `enforce_origin=true`, odd-origin parity is imposed via `Ξ0[1]=0`.
"""
function characteristic_initial_data(r::AbstractVector;
                                     w_in0,
                                     w_out0,
                                     enforce_origin::Bool = true)
    n = length(r)
    n > 0 || throw(ArgumentError("`r` must be non-empty."))

    w_in = _profile_vector(w_in0, r, "w_in0")
    w_out = _profile_vector(w_out0, r, "w_out0")

    Π0 = 0.5 .* (w_in .+ w_out)
    Ξ0 = 0.5 .* (w_in .- w_out)
    apply_symmetry_state!(Π0, Ξ0; enforce_origin = enforce_origin)

    return (Π0 = Π0, Ξ0 = Ξ0, w_in0 = w_in, w_out0 = w_out)
end

@inline function _normalize_boundary_condition(boundary_condition::Symbol)
    if boundary_condition === :radiative
        return :absorbing
    elseif boundary_condition === :reflective
        return :reflecting
    end
    return boundary_condition
end

"""
    apply_symmetry_state!(Π, Ξ; enforce_origin=true)

Mutating symmetry constraints on the state variables.

- `Π` is even at the origin (no direct point constraint needed here).
- `Ξ` is odd at the origin, so `Ξ(0)=0` is imposed by setting `Ξ[1]=0`.
"""
function apply_symmetry_state!(Π::AbstractVector,
                               Ξ::AbstractVector;
                               enforce_origin::Bool = true)
    n = length(Π)
    length(Ξ) == n || throw(DimensionMismatch("`Π` and `Ξ` must have matching lengths."))

    if enforce_origin && n >= 1
        Ξ[1] = zero(eltype(Ξ))
    end

    return nothing
end

"""
    apply_symmetry_rhs!(dΠ, dΞ; enforce_origin=true)

Mutating symmetry constraints on the RHS variables.

For stability with metric-weighted mass (`H[1,1]=0` for `p>0`), enforce invariance
of the odd subspace at the origin by imposing `dΞ(0)=0`.
"""
function apply_symmetry_rhs!(dΠ::AbstractVector,
                             dΞ::AbstractVector;
                             enforce_origin::Bool = true)
    n = length(dΠ)
    length(dΞ) == n || throw(DimensionMismatch("`dΠ` and `dΞ` must have matching lengths."))

    if enforce_origin && n >= 1
        dΞ[1] = zero(eltype(dΞ))
    end

    return nothing
end

"""
    initialize_wave_state!(Π, Ξ; enforce_origin=true)

Apply required symmetry constraints to initial state data.
"""
function initialize_wave_state!(Π::AbstractVector,
                                Ξ::AbstractVector;
                                enforce_origin::Bool = true)
    apply_symmetry_state!(Π, Ξ; enforce_origin = enforce_origin)
    return nothing
end

"""
    boundary_characteristics(ΠN, ΞN)

Characteristic variables at the outer boundary:

- `w_in  = ΠN + ΞN`
- `w_out = ΠN - ΞN`
"""
function boundary_characteristics(ΠN::Real, ΞN::Real)
    T = promote_type(typeof(ΠN), typeof(ΞN))
    ΠT = convert(T, ΠN)
    ΞT = convert(T, ΞN)
    w_in = ΠT + ΞT
    w_out = ΠT - ΞT
    return (w_in = w_in, w_out = w_out)
end

"""
    boundary_characteristic_residual(Π, Ξ; bc=:absorbing)

Return the characteristic residual at `r=R` for a boundary condition:

- `:absorbing`  -> `ρ = w_in`
- `:reflecting` -> `ρ = w_in - w_out` (equivalent to `Ξ(R)=0`)
- `:dirichlet`  -> `ρ = w_in + w_out` (equivalent to `Π(R)=0`)
- `:none`       -> `ρ = 0`

Notes:
- `:radiative` is treated as alias for `:absorbing`.
- `:reflective` is treated as alias for `:reflecting`.
"""
function boundary_characteristic_residual(Π::AbstractVector,
                                          Ξ::AbstractVector;
                                          bc::Symbol = :absorbing)
    n = length(Π)
    length(Ξ) == n || throw(DimensionMismatch("`Π` and `Ξ` must have matching lengths."))
    n > 0 || throw(ArgumentError("State vectors must be non-empty."))

    bc_norm = _normalize_boundary_condition(bc)
    chars = boundary_characteristics(Π[end], Ξ[end])

    if bc_norm === :none
        T = promote_type(typeof(chars.w_in), typeof(chars.w_out))
        return zero(T)
    elseif bc_norm === :absorbing
        return chars.w_in
    elseif bc_norm === :reflecting
        return chars.w_in - chars.w_out
    elseif bc_norm === :dirichlet
        return chars.w_in + chars.w_out
    else
        throw(ArgumentError("Unsupported boundary condition `$bc_norm`. Use :absorbing, :reflecting, :dirichlet, or :none."))
    end
end

"""
    apply_characteristic_bc_sat!(dΠ, dΞ, Π, Ξ, ops; bc=:absorbing)

Apply SBP-SAT boundary penalties at the outer boundary `r=R`.

Penalty form:
- `dΠ[end] += -σΠ * invHN * ρ`
- `dΞ[end] += -σΞ * invHN * ρ`

where `HN = H[end,end]` and `ρ` is a characteristic residual.

Implemented conditions:
- `:absorbing`: `ρ = w_in = Π + Ξ`, with
  `σΠ = BNN/2`, `σΞ = BNN/2`, giving dissipative boundary contribution.
- `:reflecting`: `ρ = w_in - w_out = 2Ξ` (equivalent to `Ξ(R)=0`), with
  `σΠ = BNN/2`, `σΞ = 0`, giving energy-conserving boundary cancellation.
- `:dirichlet`: `ρ = w_in + w_out = 2Π`, with
  `σΠ = 0`, `σΞ = BNN/2`, also energy-conserving.
- `:none`: no boundary SAT term.

Aliases:
- `:radiative` -> `:absorbing`
- `:reflective` -> `:reflecting`
"""
function apply_characteristic_bc_sat!(dΠ::AbstractVector,
                                      dΞ::AbstractVector,
                                      Π::AbstractVector,
                                      Ξ::AbstractVector,
                                      ops::SphericalOperators;
                                      bc::Symbol = :absorbing)
    n = length(ops.r)
    length(dΠ) == n || throw(DimensionMismatch("`dΠ` length must match grid size $(n)."))
    length(dΞ) == n || throw(DimensionMismatch("`dΞ` length must match grid size $(n)."))
    length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    length(Ξ) == n || throw(DimensionMismatch("`Ξ` length must match grid size $(n)."))

    bc_norm = _normalize_boundary_condition(bc)
    if bc_norm === :none
        return nothing
    end

    T = promote_type(
                     eltype(dΠ),
                     eltype(dΞ),
                     eltype(Π),
                     eltype(Ξ),
                     eltype(ops.r)
                    )
    twoT = convert(T, 2)
    BNN = convert(T, ops.B[end, end])
    HNN = convert(T, ops.H[end, end])
    HNN == zero(T) && throw(ArgumentError("`H[end,end]` must be nonzero for SAT penalties."))

    invHN = one(T) / HNN
    chars = boundary_characteristics(convert(T, Π[end]), convert(T, Ξ[end]))
    w_in = chars.w_in
    w_out = chars.w_out

    ρ = zero(T)
    σΠ = zero(T)
    σΞ = zero(T)

    if bc_norm === :absorbing
        ρ = w_in
        σΠ = BNN / twoT
        σΞ = BNN / twoT
    elseif bc_norm === :reflecting
        ρ = w_in - w_out
        σΠ = BNN / twoT
        σΞ = zero(T)
    elseif bc_norm === :dirichlet
        ρ = w_in + w_out
        σΠ = zero(T)
        σΞ = BNN / twoT
    else
        throw(ArgumentError("Unsupported boundary condition `$bc_norm`. Use :absorbing, :reflecting, :dirichlet, or :none."))
    end

    dΠ[end] -= convert(eltype(dΠ), σΠ * invHN * ρ)
    dΞ[end] -= convert(eltype(dΞ), σΞ * invHN * ρ)

    return nothing
end

"""
    apply_symmetry_constraints!(Π, Ξ; enforce_origin=true)

Backward-compatible alias for `apply_symmetry_state!`.
"""
function apply_symmetry_constraints!(Π::AbstractVector,
                                     Ξ::AbstractVector;
                                     enforce_origin::Bool = true)
    apply_symmetry_state!(Π, Ξ; enforce_origin = enforce_origin)
    return nothing
end

"""
    apply_boundary_conditions!(Π, Ξ; boundary_condition=:absorbing)

Backward-compatible state-projection boundary helper.

This is not used by the SAT-based default time evolution path; SAT penalties are
applied in `wave_system_ode!` through `apply_characteristic_bc_sat!`.
"""
function apply_boundary_conditions!(Π::AbstractVector,
                                    Ξ::AbstractVector;
                                    boundary_condition::Symbol = :absorbing)
    n = length(Π)
    length(Ξ) == n || throw(DimensionMismatch("`Π` and `Ξ` must have matching lengths."))
    n == 0 && return nothing

    bc_norm = _normalize_boundary_condition(boundary_condition)

    if bc_norm === :none
        return nothing
    elseif bc_norm === :absorbing
        Π[end] = -Ξ[end]
    elseif bc_norm === :reflecting
        Ξ[end] = zero(eltype(Ξ))
    elseif bc_norm === :dirichlet
        Π[end] = zero(eltype(Π))
    else
        throw(ArgumentError("Unsupported boundary condition `$bc_norm`. Use :absorbing, :reflecting, :dirichlet, or :none."))
    end

    return nothing
end

"""
    apply_wave_constraints!(Π, Ξ, ops; boundary_condition=:absorbing,
                            enforce_origin=true)

Backward-compatible convenience wrapper applying symmetry and state-projection BC.
The SAT-based wave solver path does not call this function.
"""
function apply_wave_constraints!(Π::AbstractVector,
                                 Ξ::AbstractVector,
                                 ops::SphericalOperators;
                                 boundary_condition::Symbol = :absorbing,
                                 enforce_origin::Bool = true)
    n = length(ops.r)
    length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    length(Ξ) == n || throw(DimensionMismatch("`Ξ` length must match grid size $(n)."))

    apply_symmetry_state!(Π, Ξ; enforce_origin = enforce_origin)
    apply_boundary_conditions!(Π, Ξ; boundary_condition = boundary_condition)
    return nothing
end

function _default_wave_check_tol(::Type{T}) where {T <: AbstractFloat}
    return T(100) * eps(T)
end

function _default_wave_check_tol(::Type{T}) where {T <: Real}
    return zero(T)
end

"""
    check_wave_data_consistency(Π, Ξ; boundary_condition=:absorbing,
                                enforce_origin=true, require_boundary=false,
                                tol=nothing)

Check whether a wave-state pair `(Π, Ξ)` is consistent with imposed parity and, when
requested, boundary-condition residual tolerance.

For SAT evolution, `require_boundary=false` is generally appropriate for initial data,
since boundary conditions are imposed weakly through the RHS, not by state projection.
"""
function check_wave_data_consistency(Π::AbstractVector,
                                     Ξ::AbstractVector;
                                     boundary_condition::Symbol = :absorbing,
                                     enforce_origin::Bool = true,
                                     require_boundary::Bool = false,
                                     tol = nothing)
    n = length(Π)
    length(Ξ) == n || throw(DimensionMismatch("`Π` and `Ξ` must have matching lengths."))
    n > 0 || throw(ArgumentError("State vectors must be non-empty."))

    bc_norm = _normalize_boundary_condition(boundary_condition)

    T = promote_type(eltype(Π), eltype(Ξ))
    tolT = tol === nothing ? _default_wave_check_tol(T) : convert(T, tol)

    max_abs_pi = maximum(abs.(Π))
    max_abs_pi_interior = n > 1 ? maximum(abs.(view(Π, 1:(n - 1)))) : max_abs_pi
    max_abs_xi = maximum(abs.(Ξ))

    origin_residual = enforce_origin ? abs(Ξ[1]) : zero(T)
    origin_ok = !enforce_origin || origin_residual <= tolT

    boundary_residual = abs(boundary_characteristic_residual(Π, Ξ; bc = bc_norm))
    boundary_ok = bc_norm === :none || boundary_residual <= tolT

    finite_ok = all(isfinite, Π) && all(isfinite, Ξ)
    consistent = finite_ok && origin_ok && (!require_boundary || boundary_ok)

    return (
            consistent = consistent,
            finite_ok = finite_ok,
            origin_ok = origin_ok,
            boundary_ok = boundary_ok,
            require_boundary = require_boundary,
            origin_residual = origin_residual,
            boundary_residual = boundary_residual,
            max_abs_pi = max_abs_pi,
            max_abs_pi_interior = max_abs_pi_interior,
            max_abs_xi = max_abs_xi,
            boundary_condition = bc_norm,
            enforce_origin = enforce_origin,
            tol = tolT
           )
end

"""
    check_potential_consistency(ops, Π, Ξ; tol=nothing, gauge=:origin)

Check consistency of a wave state with the potential constraint `Ξ ≈ Geven * ϕ`.

The routine reconstructs `ϕ̂` by least squares and reports absolute/relative residuals
for `Geven*ϕ̂ - Ξ`. It also reports `max|Geven*Π|`, which is the immediate source term
for `Ξ_t`; thus non-constant `Π` produces immediate `Ξ` growth through `Ξ_t = Geven*Π`.
"""
function check_potential_consistency(ops::SphericalOperators,
                                     Π::AbstractVector,
                                     Ξ::AbstractVector;
                                     tol = nothing,
                                     gauge::Symbol = :origin)
    n = length(ops.r)
    length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    length(Ξ) == n || throw(DimensionMismatch("`Ξ` length must match grid size $(n)."))
    n > 0 || throw(ArgumentError("State vectors must be non-empty."))

    gauge in (:origin, :mean) ||
        throw(ArgumentError("`gauge` must be :origin or :mean."))

    Πf = Float64.(Π)
    Ξf = Float64.(Ξ)
    G = Matrix{Float64}(ops.Geven)

    ϕ̂ = zeros(Float64, n)
    if gauge === :origin
        if n >= 2
            A = @view G[:, 2:n]
            ϕ̂[2:n] .= A \ Ξf
        end
    else
        A = [G; fill(1.0 / n, 1, n)]
        b = vcat(Ξf, 0.0)
        ϕ̂ .= A \ b
    end

    Ξ_fit = G * ϕ̂
    resid = Ξ_fit .- Ξf

    resid_l2_abs = norm(resid)
    resid_linf_abs = isempty(resid) ? 0.0 : maximum(abs.(resid))
    xi_l2 = norm(Ξf)
    xi_linf = isempty(Ξf) ? 0.0 : maximum(abs.(Ξf))
    resid_l2_rel = resid_l2_abs / max(xi_l2, eps(Float64))
    resid_linf_rel = resid_linf_abs / max(xi_linf, eps(Float64))

    tol_use = tol === nothing ? 1e-10 : Float64(tol)
    residual_ok = resid_l2_abs <= tol_use || resid_l2_rel <= tol_use

    GΠ = Vector{Float64}(ops.Geven * Πf)
    max_abs_GPi = isempty(GΠ) ? 0.0 : maximum(abs.(GΠ))
    growth_tol = max(tol_use, 100 * eps(Float64))
    xi_growth_expected_from_pi = max_abs_GPi > growth_tol

    xi_near_zero = xi_linf <= growth_tol
    pi_range = isempty(Πf) ? 0.0 : (maximum(Πf) - minimum(Πf))
    pi_spatially_constant = pi_range <= growth_tol

    warnings = String[]
    if xi_near_zero && !pi_spatially_constant
        push!(warnings, "This initial state is inconsistent with Ξ=φ_r for any scalar φ; you are evolving a general first-order system and may excite constraint-violating modes.")
    end

    return (
            gauge = gauge,
            phi_hat = ϕ̂,
            residual_l2_abs = resid_l2_abs,
            residual_l2_rel = resid_l2_rel,
            residual_linf_abs = resid_linf_abs,
            residual_linf_rel = resid_linf_rel,
            residual_ok = residual_ok,
            tol = tol_use,
            xi_norm_l2 = xi_l2,
            xi_norm_linf = xi_linf,
            max_abs_GPi = max_abs_GPi,
            xi_growth_expected_from_pi = xi_growth_expected_from_pi,
            xi_near_zero = xi_near_zero,
            pi_spatially_constant = pi_spatially_constant,
            warnings = warnings,
            growth_explanation = "Ξ appears immediately if Π0 varies because Ξ_t = G Π.",
            note = "Π can be prescribed independently as φ_t; Ξ_t = Geven*Π couples Π variations into Ξ immediately."
           )
end

"""
    wave_rhs!(dΠ, dΞ, Π, Ξ, ops)

Low-level semidiscrete first-order radial wave system on `[0, R]`:

`∂t Π = D*Ξ`,
`∂t Ξ = Geven*Π`.
"""
function wave_rhs!(dΠ::AbstractVector,
                   dΞ::AbstractVector,
                   Π::AbstractVector,
                   Ξ::AbstractVector,
                   ops::SphericalOperators)
    n = length(ops.r)
    length(dΠ) == n || throw(DimensionMismatch("`dΠ` length must match grid size $(n)."))
    length(dΞ) == n || throw(DimensionMismatch("`dΞ` length must match grid size $(n)."))
    length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    length(Ξ) == n || throw(DimensionMismatch("`Ξ` length must match grid size $(n)."))

    mul!(dΠ, ops.D, Ξ)

    mul!(dΞ, ops.Geven, Π)
    return nothing
end

function _sat_boundary_jacobian_entries(ops::SphericalOperators;
                                        bc::Symbol)
    bc_norm = _normalize_boundary_condition(bc)
    if bc_norm === :none
        return (
                apply = false,
                dPi_dPi = 0.0,
                dPi_dXi = 0.0,
                dXi_dPi = 0.0,
                dXi_dXi = 0.0
               )
    end

    BNN = Float64(ops.B[end, end])
    HNN = Float64(ops.H[end, end])
    HNN == 0.0 && throw(ArgumentError("`H[end,end]` must be nonzero for SAT Jacobian terms."))
    invHN = 1.0 / HNN

    dPi_dPi = 0.0
    dPi_dXi = 0.0
    dXi_dPi = 0.0
    dXi_dXi = 0.0

    if bc_norm === :absorbing
        coeff_pi = -(BNN / 2.0) * invHN
        coeff_xi = -(BNN / 2.0) * invHN
        dPi_dPi += coeff_pi
        dPi_dXi += coeff_pi
        dXi_dPi += coeff_xi
        dXi_dXi += coeff_xi
    elseif bc_norm === :reflecting
        dPi_dXi += -(BNN * invHN)
    elseif bc_norm === :dirichlet
        dXi_dPi += -(BNN * invHN)
    else
        throw(ArgumentError("Unsupported boundary condition `$bc_norm`. Use :absorbing, :reflecting, :dirichlet, or :none."))
    end

    return (
            apply = true,
            dPi_dPi = dPi_dPi,
            dPi_dXi = dPi_dXi,
            dXi_dPi = dXi_dPi,
            dXi_dXi = dXi_dXi
           )
end

"""
    WaveODEParams(ops; boundary_condition=:absorbing, enforce_origin=true)

Parameters for `wave_system_ode!` in SciML `f!(du,u,p,t)` style.
"""
struct WaveODEParams{T <: Real, Ti <: Integer}
    ops::SphericalOperators{T, Ti}
    boundary_condition::Symbol
    enforce_origin::Bool
end

function WaveODEParams(ops::SphericalOperators{T, Ti};
                       boundary_condition::Symbol = :absorbing,
                       enforce_origin::Bool = true) where {T <: Real, Ti <: Integer}
    return WaveODEParams{T, Ti}(ops, _normalize_boundary_condition(boundary_condition), enforce_origin)
end

"""
    wave_system_ode!(dU, U, p, t)

SciML-compatible RHS for the radial wave system with SAT boundary conditions.

State layout (matrix form):
- `U[:,1] = Π`
- `U[:,2] = Ξ`
"""
function wave_system_ode!(dU::AbstractMatrix,
                          U::AbstractMatrix,
                          p::WaveODEParams,
                          t)
    _ = t
    n = length(p.ops.r)
    size(U, 1) == n || throw(DimensionMismatch("`U` first dimension must be $(n)."))
    size(dU, 1) == n || throw(DimensionMismatch("`dU` first dimension must be $(n)."))
    size(U, 2) == 2 || throw(DimensionMismatch("`U` must have exactly 2 columns (Π, Ξ)."))
    size(dU, 2) == 2 || throw(DimensionMismatch("`dU` must have exactly 2 columns (dΠ, dΞ)."))

    Π = @view U[:, 1]
    Ξ = @view U[:, 2]
    dΠ = @view dU[:, 1]
    dΞ = @view dU[:, 2]

    wave_rhs!(dΠ, dΞ, Π, Ξ, p.ops)
    apply_symmetry_rhs!(dΠ, dΞ; enforce_origin = p.enforce_origin)
    apply_characteristic_bc_sat!(dΠ, dΞ, Π, Ξ, p.ops; bc = p.boundary_condition)

    return nothing
end

"""
    wave_system_ode_vec!(dU, U, p, t)

SciML-compatible RHS for the radial wave system in stacked vector form:

- `U[1:n] = Π`
- `U[n+1:2n] = Ξ`
"""
function wave_system_ode_vec!(dU::AbstractVector,
                              U::AbstractVector,
                              p::WaveODEParams,
                              t)
    _ = t
    n = length(p.ops.r)
    length(U) == 2 * n || throw(DimensionMismatch("`U` length must be $(2 * n)."))
    length(dU) == 2 * n || throw(DimensionMismatch("`dU` length must be $(2 * n)."))

    Π = @view U[1:n]
    Ξ = @view U[(n + 1):(2 * n)]
    dΠ = @view dU[1:n]
    dΞ = @view dU[(n + 1):(2 * n)]

    wave_rhs!(dΠ, dΞ, Π, Ξ, p.ops)
    apply_symmetry_rhs!(dΠ, dΞ; enforce_origin = p.enforce_origin)
    apply_characteristic_bc_sat!(dΠ, dΞ, Π, Ξ, p.ops; bc = p.boundary_condition)

    return nothing
end

"""
    wave_system_jac!(J, U, p, t)

Analytic Jacobian for `wave_system_ode_vec!` in stacked layout `[Π; Ξ]`.
"""
function wave_system_jac!(J::AbstractMatrix,
                          U::AbstractVector,
                          p::WaveODEParams,
                          t)
    _ = U
    _ = t
    n = length(p.ops.r)
    size(J, 1) == 2 * n || throw(DimensionMismatch("Jacobian must have $(2 * n) rows."))
    size(J, 2) == 2 * n || throw(DimensionMismatch("Jacobian must have $(2 * n) cols."))

    fill!(J, zero(eltype(J)))

    I_D, J_D, V_D = findnz(p.ops.D)
    @inbounds for k in eachindex(V_D)
        J[I_D[k], n + J_D[k]] = convert(eltype(J), V_D[k])
    end

    I_G, J_G, V_G = findnz(p.ops.Geven)
    @inbounds for k in eachindex(V_G)
        J[n + I_G[k], J_G[k]] = convert(eltype(J), V_G[k])
    end

    sat = _sat_boundary_jacobian_entries(p.ops; bc = p.boundary_condition)
    if sat.apply
        J[n, n] += convert(eltype(J), sat.dPi_dPi)
        J[n, 2 * n] += convert(eltype(J), sat.dPi_dXi)
        J[2 * n, n] += convert(eltype(J), sat.dXi_dPi)
        J[2 * n, 2 * n] += convert(eltype(J), sat.dXi_dXi)
    end

    if p.enforce_origin
        fill!(@view(J[n + 1, :]), zero(eltype(J)))
    end

    return nothing
end

"""
    wave_system_jac_prototype(ops; boundary_condition=:absorbing)

Sparse Jacobian nonzero pattern for `wave_system_jac!` in stacked layout `[Π; Ξ]`.
"""
function wave_system_jac_prototype(ops::SphericalOperators;
                                   boundary_condition::Symbol = :absorbing)
    n = length(ops.r)
    bc_norm = _normalize_boundary_condition(boundary_condition)

    I = Int[]
    J = Int[]

    I_D, J_D, _ = findnz(ops.D)
    append!(I, I_D)
    append!(J, J_D .+ n)

    I_G, J_G, _ = findnz(ops.Geven)
    append!(I, I_G .+ n)
    append!(J, J_G)

    if bc_norm === :absorbing
        append!(I, (n, n, 2 * n, 2 * n))
        append!(J, (n, 2 * n, n, 2 * n))
    elseif bc_norm === :reflecting
        append!(I, (n,))
        append!(J, (2 * n,))
    elseif bc_norm === :dirichlet
        append!(I, (2 * n,))
        append!(J, (n,))
    elseif bc_norm !== :none
        throw(ArgumentError("Unsupported boundary condition `$bc_norm`. Use :absorbing, :reflecting, :dirichlet, or :none."))
    end

    V = ones(Float64, length(I))
    return sparse(I, J, V, 2 * n, 2 * n)
end

"""
    wave_energy(ops, Π, Ξ)

Discrete wave energy

`E = 0.5 * (Π' * H * Π + Ξ' * H * Ξ)`.
"""
function wave_energy(ops::SphericalOperators, Π::AbstractVector, Ξ::AbstractVector)
    n = length(ops.r)
    length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    length(Ξ) == n || throw(DimensionMismatch("`Ξ` length must match grid size $(n)."))

    return 0.5 * (dot(Π, ops.H * Π) + dot(Ξ, ops.H * Ξ))
end
