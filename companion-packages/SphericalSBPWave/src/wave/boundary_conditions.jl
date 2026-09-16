@inline function _normalize_boundary_condition(boundary_condition::Symbol)
    if boundary_condition === :radiative
        return :absorbing
    elseif boundary_condition === :reflective
        return :reflecting
    end
    return boundary_condition
end

"""
    boundary_characteristics(ΠN, ΨN)

Characteristic variables at the outer boundary:

- `w_in  = ΠN + ΨN`
- `w_out = ΠN - ΨN`
"""
function boundary_characteristics(ΠN::Real, ΨN::Real)
    T = promote_type(typeof(ΠN), typeof(ΨN))
    ΠT = convert(T, ΠN)
    ΨT = convert(T, ΨN)
    w_in = ΠT + ΨT
    w_out = ΠT - ΨT
    return (w_in = w_in, w_out = w_out)
end

"""
    boundary_characteristic_residual(Π, Ψ; bc=:absorbing)

Return the characteristic residual at `r=R` for a boundary condition:

- `:absorbing`  -> `ρ = w_in`
- `:reflecting` -> `ρ = w_in - w_out` (equivalent to `Ψ(R)=0`)
- `:dirichlet`  -> `ρ = w_in + w_out` (equivalent to `Π(R)=0`)
- `:none`       -> `ρ = 0`

Notes:
- `:radiative` is treated as alias for `:absorbing`.
- `:reflective` is treated as alias for `:reflecting`.
"""
function boundary_characteristic_residual(Π::AbstractVector,
                                          Ψ::AbstractVector;
                                          bc::Symbol = :absorbing)
    n = length(Π)
    length(Ψ) == n || throw(DimensionMismatch("`Π` and `Ψ` must have matching lengths."))
    n > 0 || throw(ArgumentError("State vectors must be non-empty."))

    bc_norm = _normalize_boundary_condition(bc)
    chars = boundary_characteristics(Π[end], Ψ[end])

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
    apply_characteristic_bc_sat!(dΠ, dΨ, Π, Ψ, ops; bc=:absorbing)

Apply SBP-SAT boundary penalties at the outer boundary `r=R`.

Penalty form:
- `dΠ[end] += -σΠ * S^{-1}_{NN} * ρ`
- `dΨ[end] += -σΨ * V^{-1}_{NN} * ρ`

where `SNN = S[end,end]`, `VNN = V[end,end]`, and `ρ` is a characteristic residual.

Implemented conditions:
- `:absorbing`: `ρ = w_in = Π + Ψ`, with
  `σΠ = BNN/2`, `σΨ = BNN/2`, giving dissipative boundary contribution.
- `:reflecting`: `ρ = w_in - w_out = 2Ψ` (equivalent to `Ψ(R)=0`), with
  `σΠ = BNN/2`, `σΨ = 0`, giving energy-conserving boundary cancellation.
- `:dirichlet`: `ρ = w_in + w_out = 2Π`, with
  `σΠ = 0`, `σΨ = BNN/2`, also energy-conserving.
- `:none`: no boundary SAT term.

Aliases:
- `:radiative` -> `:absorbing`
- `:reflective` -> `:reflecting`
"""
function apply_characteristic_bc_sat!(dΠ::AbstractVector,
                                      dΨ::AbstractVector,
                                      Π::AbstractVector,
                                      Ψ::AbstractVector,
                                      ops::WaveOperators;
                                      bc::Symbol = :absorbing)
    n = length(ops.r)
    length(dΠ) == n || throw(DimensionMismatch("`dΠ` length must match grid size $(n)."))
    length(dΨ) == n || throw(DimensionMismatch("`dΨ` length must match grid size $(n)."))
    length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    length(Ψ) == n || throw(DimensionMismatch("`Ψ` length must match grid size $(n)."))

    bc_norm = _normalize_boundary_condition(bc)
    if bc_norm === :none
        return nothing
    end

    T = promote_type(eltype(dΠ),
                     eltype(dΨ),
                     eltype(Π),
                     eltype(Ψ),
                     eltype(ops.r),
                     eltype(ops.S),
                     eltype(ops.V),
                     eltype(ops.B))

    BNN = convert(T, ops.B[end, end])      # = R^p at outer boundary
    SNN = convert(T, ops.S[end, end])      # scalar norm endpoint weight
    VNN = convert(T, ops.V[end, end])      # vector norm endpoint weight

    # SNN == zero(T) &&
    #     throw(ArgumentError("`S[end,end]` must be nonzero for SAT penalties."))
    # VNN == zero(T) &&
    #     throw(ArgumentError("`V[end,end]` must be nonzero for SAT penalties."))

    invSN = one(T) / SNN
    invVN = one(T) / VNN
    halfB = BNN / convert(T, 2)
    quarterB = BNN / convert(T, 4)

    ΠN = convert(T, Π[end])
    ΨN = convert(T, Ψ[end])

    # For the semidiscrete system
    #   Π_t = DΨ,  Ψ_t = GΠ,
    # the right-boundary incoming/outgoing characteristics are
    #   w_in  = Π + Ψ
    #   w_out = Π - Ψ
    w_in = ΠN + ΨN
    w_out = ΠN - ΨN

    if bc_norm === :absorbing
        # Impose incoming characteristic = 0:
        #   w_in = Π + Ψ = 0
        #
        # SATs:
        #   Π_t <- Π_t - (B/4) S^{-1} e_N w_in
        #   Ψ_t <- Ψ_t - (B/4) V^{-1} e_N w_in
        #
        # Then
        #   dE/dt = B Π_N Ψ_N - (B/4)(Π_N+Ψ_N)^2 - (B/4)(Π_N+Ψ_N)^2
        #         = -(B/4)(Π_N-Ψ_N)^?  [No: for this sign convention]
        #         = -(B/4)(Π_N+Ψ_N)^2 <= 0
        dΠ[end] -= convert(eltype(dΠ), quarterB * invSN * w_in)
        dΨ[end] -= convert(eltype(dΨ), quarterB * invVN * w_in)

    elseif bc_norm === :reflecting
        # Reflecting wall: Ψ_N = 0  <=>  w_in - w_out = 2Ψ_N = 0
        #
        # Energy-conserving SAT:
        #   Π_t <- Π_t - B S^{-1} e_N Ψ_N
        #
        # Equivalent characteristic form:
        #   Π_t <- Π_t - (B/2) S^{-1} e_N (w_in - w_out)
        #
        # Then dE/dt = B Π_N Ψ_N - B Π_N Ψ_N = 0.
        ρ = w_in - w_out   # = 2 Ψ_N
        dΠ[end] -= convert(eltype(dΠ), halfB * invSN * ρ)

    elseif bc_norm === :dirichlet
        # Homogeneous Π boundary condition: Π_N = 0
        # <=> w_in + w_out = 2Π_N = 0
        #
        # Energy-conserving SAT:
        #   Ψ_t <- Ψ_t - B V^{-1} e_N Π_N
        #
        # Equivalent characteristic form:
        #   Ψ_t <- Ψ_t - (B/2) V^{-1} e_N (w_in + w_out)
        #
        # Then dE/dt = B Π_N Ψ_N - B Π_N Ψ_N = 0.
        ρ = w_in + w_out   # = 2 Π_N
        dΨ[end] -= convert(eltype(dΨ), halfB * invVN * ρ)

    else
        throw(ArgumentError("Unsupported boundary condition `$bc_norm`. " *
                            "Use :absorbing, :reflecting, :dirichlet, or :none."))
    end

    return nothing
end
# function apply_characteristic_bc_sat!(
#         dΠ::AbstractVector,
#         dΨ::AbstractVector,
#         Π::AbstractVector,
#         Ψ::AbstractVector,
#         ops::WaveOperators;
#         bc::Symbol = :absorbing
#     )
#     n = length(ops.r)
#     length(dΠ) == n || throw(DimensionMismatch("`dΠ` length must match grid size $(n)."))
#     length(dΨ) == n || throw(DimensionMismatch("`dΨ` length must match grid size $(n)."))
#     length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
#     length(Ψ) == n || throw(DimensionMismatch("`Ψ` length must match grid size $(n)."))

#     bc_norm = _normalize_boundary_condition(bc)
#     if bc_norm === :none
#         return nothing
#     end

#     T = promote_type(
#         eltype(dΠ),
#         eltype(dΨ),
#         eltype(Π),
#         eltype(Ψ),
#         eltype(ops.r)
#     )
#     twoT = convert(T, 2)
#     BNN = convert(T, ops.B[end, end])
#     HNN = convert(T, ops.H[end, end])
#     HNN == zero(T) &&
#         throw(ArgumentError("`H[end,end]` must be nonzero for SAT penalties."))

#     invHN = one(T) / HNN
#     chars = boundary_characteristics(convert(T, Π[end]), convert(T, Ψ[end]))
#     w_in = chars.w_in
#     w_out = chars.w_out

#     ρ = zero(T)
#     σΠ = zero(T)
#     σΨ = zero(T)

#     if bc_norm === :absorbing
#         ρ = w_in
#         σΠ = BNN / twoT
#         σΨ = BNN / twoT
#     elseif bc_norm === :reflecting
#         ρ = w_in - w_out
#         σΠ = BNN / twoT
#         σΨ = zero(T)
#     elseif bc_norm === :dirichlet
#         ρ = w_in + w_out
#         σΠ = zero(T)
#         σΨ = BNN / twoT
#     else
#         throw(ArgumentError("Unsupported boundary condition `$bc_norm`. Use :absorbing, :reflecting, :dirichlet, or :none."))
#     end

#     dΠ[end] -= convert(eltype(dΠ), σΠ * invHN * ρ)
#     dΨ[end] -= convert(eltype(dΨ), σΨ * invHN * ρ)

#     return nothing
# end

"""
    apply_boundary_conditions!(Π, Ψ; boundary_condition=:absorbing)

Backward-compatible state-projection boundary helper.

This is not used by the SAT-based default time evolution path; SAT penalties are
applied in `wave_system_ode!` through `apply_characteristic_bc_sat!`.
"""
function apply_boundary_conditions!(Π::AbstractVector,
                                    Ψ::AbstractVector;
                                    boundary_condition::Symbol = :absorbing)
    n = length(Π)
    length(Ψ) == n || throw(DimensionMismatch("`Π` and `Ψ` must have matching lengths."))
    n == 0 && return nothing

    bc_norm = _normalize_boundary_condition(boundary_condition)

    if bc_norm === :none
        return nothing
    elseif bc_norm === :absorbing
        Π[end] = -Ψ[end]
    elseif bc_norm === :reflecting
        Ψ[end] = zero(eltype(Ψ))
    elseif bc_norm === :dirichlet
        Π[end] = zero(eltype(Π))
    else
        throw(ArgumentError("Unsupported boundary condition `$bc_norm`. Use :absorbing, :reflecting, :dirichlet, or :none."))
    end

    return nothing
end

"""
    apply_wave_constraints!(Π, Ψ, ops; boundary_condition=:absorbing,
                            enforce_origin=true)

Backward-compatible convenience wrapper applying symmetry and state-projection BC.
The SAT-based wave solver path does not call this function.
"""
function apply_wave_constraints!(Π::AbstractVector,
                                 Ψ::AbstractVector,
                                 ops::WaveOperators;
                                 boundary_condition::Symbol = :absorbing,
                                 enforce_origin::Bool = true)
    n = length(ops.r)
    length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    length(Ψ) == n || throw(DimensionMismatch("`Ψ` length must match grid size $(n)."))

    apply_symmetry_state!(Π,
                          Ψ;
                          enforce_origin = enforce_origin,
                          has_origin_node = _has_origin_node(ops))
    apply_boundary_conditions!(Π, Ψ; boundary_condition = boundary_condition)
    return nothing
end
