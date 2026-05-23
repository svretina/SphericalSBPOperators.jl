"""
    wave_rhs!(dΠ, dΨ, Π, Ψ, ops)

Low-level semidiscrete first-order radial wave system on `[0, R]`:

`∂t Π = D*Ψ`,
`∂t Ψ = Geven*Π`.
"""
function wave_rhs!(dΠ::AbstractVector,
                   dΨ::AbstractVector,
                   Π::AbstractVector,
                   Ψ::AbstractVector,
                   ops::WaveRHSOperators)
    n = length(ops.r)
    length(dΠ) == n || throw(DimensionMismatch("`dΠ` length must match grid size $(n)."))
    length(dΨ) == n || throw(DimensionMismatch("`dΨ` length must match grid size $(n)."))
    length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    length(Ψ) == n || throw(DimensionMismatch("`Ψ` length must match grid size $(n)."))

    mul!(dΠ, ops.D, Ψ)
    mul!(dΨ, ops.Geven, Π)
    return nothing
end

@inline function _apply_cached_wave_rhs_constraints!(dΠ::AbstractVector,
                                                     dΨ::AbstractVector,
                                                     Π::AbstractVector,
                                                     Ψ::AbstractVector,
                                                     p::WaveODEParams)
    if p.enforce_origin
        @inbounds dΨ[1] = zero(eltype(dΨ))
    end

    bc = p.boundary_condition
    bc === :none && return nothing

    cache = p.boundary_cache
    @inbounds begin
        ΠN = Π[end]
        ΨN = Ψ[end]
        if bc === :absorbing
            w_in = ΠN + ΨN
            dΠ[end] -= cache.absorbing_pi * w_in
            dΨ[end] -= cache.absorbing_psi * w_in
        elseif bc === :reflecting
            dΠ[end] -= cache.reflecting_pi * ΨN
        elseif bc === :dirichlet
            dΨ[end] -= cache.dirichlet_psi * ΠN
        else
            throw(ArgumentError("Unsupported boundary condition `$bc`. Use :absorbing, :reflecting, :dirichlet, or :none."))
        end
    end

    return nothing
end

"""
    wave_system_ode!(dU, U, p, t)

SciML-compatible RHS for the radial wave system with SAT boundary conditions.

State layout (matrix form):
- `U[:,1] = Π`
- `U[:,2] = Ψ`
"""
function wave_system_ode!(dU::AbstractMatrix,
                          U::AbstractMatrix,
                          p::WaveODEParams,
                          t)
    _ = t
    n = length(p.ops.r)
    size(U, 1) == n || throw(DimensionMismatch("`U` first dimension must be $(n)."))
    size(dU, 1) == n || throw(DimensionMismatch("`dU` first dimension must be $(n)."))
    size(U, 2) == 2 || throw(DimensionMismatch("`U` must have exactly 2 columns (Π, Ψ)."))
    size(dU, 2) == 2 ||
        throw(DimensionMismatch("`dU` must have exactly 2 columns (dΠ, dΨ)."))

    Π = @view U[:, 1]
    Ψ = @view U[:, 2]
    dΠ = @view dU[:, 1]
    dΨ = @view dU[:, 2]

    wave_rhs!(dΠ, dΨ, Π, Ψ, p.ops)
    _apply_cached_wave_rhs_constraints!(dΠ, dΨ, Π, Ψ, p)

    return nothing
end

"""
    wave_system_ode_vec!(dU, U, p, t)

SciML-compatible RHS for the radial wave system in stacked vector form:

- `U[1:n] = Π`
- `U[n+1:2n] = Ψ`
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
    Ψ = @view U[(n + 1):(2 * n)]
    dΠ = @view dU[1:n]
    dΨ = @view dU[(n + 1):(2 * n)]

    wave_rhs!(dΠ, dΨ, Π, Ψ, p.ops)
    _apply_cached_wave_rhs_constraints!(dΠ, dΨ, Π, Ψ, p)

    return nothing
end

"""
    wave_energy(ops, Π, Ψ)

Discrete wave energy

`E = 0.5 * (Π' * S * Π + Ψ' * V * Ψ)`,
with the staggered case reducing to the shared `H` norm for both fields.
"""
function wave_energy(ops::WaveOperators, Π::AbstractVector, Ψ::AbstractVector)
    # n = length(ops.r)
    # length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    # length(Ψ) == n || throw(DimensionMismatch("`Ψ` length must match grid size $(n)."))

    S = _wave_scalar_mass(ops)
    V = _wave_vector_mass(ops)
    return 0.5 * (dot(Π, S * Π) + dot(Ψ, V * Ψ))
end
