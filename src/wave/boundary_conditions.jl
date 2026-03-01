@inline function _normalize_boundary_condition(boundary_condition::Symbol)
    if boundary_condition === :radiative
        return :absorbing
    elseif boundary_condition === :reflective
        return :reflecting
    end
    return boundary_condition
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
function boundary_characteristic_residual(
        Π::AbstractVector,
        Ξ::AbstractVector;
        bc::Symbol = :absorbing
    )
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
function apply_characteristic_bc_sat!(
        dΠ::AbstractVector,
        dΞ::AbstractVector,
        Π::AbstractVector,
        Ξ::AbstractVector,
        ops::WaveOperators;
        bc::Symbol = :absorbing
    )
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
    HNN == zero(T) &&
        throw(ArgumentError("`H[end,end]` must be nonzero for SAT penalties."))

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
    apply_boundary_conditions!(Π, Ξ; boundary_condition=:absorbing)

Backward-compatible state-projection boundary helper.

This is not used by the SAT-based default time evolution path; SAT penalties are
applied in `wave_system_ode!` through `apply_characteristic_bc_sat!`.
"""
function apply_boundary_conditions!(
        Π::AbstractVector,
        Ξ::AbstractVector;
        boundary_condition::Symbol = :absorbing
    )
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
function apply_wave_constraints!(
        Π::AbstractVector,
        Ξ::AbstractVector,
        ops::WaveOperators;
        boundary_condition::Symbol = :absorbing,
        enforce_origin::Bool = true
    )
    n = length(ops.r)
    length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    length(Ξ) == n || throw(DimensionMismatch("`Ξ` length must match grid size $(n)."))

    apply_symmetry_state!(
        Π,
        Ξ;
        enforce_origin = enforce_origin,
        has_origin_node = _has_origin_node(ops)
    )
    apply_boundary_conditions!(Π, Ξ; boundary_condition = boundary_condition)
    return nothing
end
