"""
    wave_rhs!(dΠ, dΞ, Π, Ξ, ops)

Low-level semidiscrete first-order radial wave system on `[0, R]`:

`∂t Π = D*Ξ`,
`∂t Ξ = Geven*Π`.
"""
function wave_rhs!(
        dΠ::AbstractVector,
        dΞ::AbstractVector,
        Π::AbstractVector,
        Ξ::AbstractVector,
        ops::WaveOperators
    )
    n = length(ops.r)
    length(dΠ) == n || throw(DimensionMismatch("`dΠ` length must match grid size $(n)."))
    length(dΞ) == n || throw(DimensionMismatch("`dΞ` length must match grid size $(n)."))
    length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    length(Ξ) == n || throw(DimensionMismatch("`Ξ` length must match grid size $(n)."))

    mul!(dΠ, ops.D, Ξ)

    mul!(dΞ, ops.Geven, Π)
    return nothing
end

"""
    wave_system_ode!(dU, U, p, t)

SciML-compatible RHS for the radial wave system with SAT boundary conditions.

State layout (matrix form):
- `U[:,1] = Π`
- `U[:,2] = Ξ`
"""
function wave_system_ode!(
        dU::AbstractMatrix,
        U::AbstractMatrix,
        p::WaveODEParams,
        t
    )
    _ = t
    n = length(p.ops.r)
    size(U, 1) == n || throw(DimensionMismatch("`U` first dimension must be $(n)."))
    size(dU, 1) == n || throw(DimensionMismatch("`dU` first dimension must be $(n)."))
    size(U, 2) == 2 || throw(DimensionMismatch("`U` must have exactly 2 columns (Π, Ξ)."))
    size(dU, 2) == 2 ||
        throw(DimensionMismatch("`dU` must have exactly 2 columns (dΠ, dΞ)."))

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
function wave_system_ode_vec!(
        dU::AbstractVector,
        U::AbstractVector,
        p::WaveODEParams,
        t
    )
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
    wave_energy(ops, Π, Ξ)

Discrete wave energy

`E = 0.5 * (Π' * H * Π + Ξ' * H * Ξ)`.
"""
function wave_energy(ops::WaveOperators, Π::AbstractVector, Ξ::AbstractVector)
    n = length(ops.r)
    length(Π) == n || throw(DimensionMismatch("`Π` length must match grid size $(n)."))
    length(Ξ) == n || throw(DimensionMismatch("`Ξ` length must match grid size $(n)."))

    return 0.5 * (dot(Π, ops.H * Π) + dot(Ξ, ops.H * Ξ))
end
