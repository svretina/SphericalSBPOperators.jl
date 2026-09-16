"""
    wave_system_jac!(J, U, p, t)

Analytic Jacobian for `wave_system_ode_vec!` in stacked layout `[Π; Ψ]`.
"""
function wave_system_jac!(
        J::AbstractMatrix,
        U::AbstractVector,
        p::WaveODEParams,
        t
    )
    _ = U
    _ = t
    n = length(p.ops.r)
    size(J, 1) == 2 * n || throw(DimensionMismatch("Jacobian must have $(2 * n) rows."))
    size(J, 2) == 2 * n || throw(DimensionMismatch("Jacobian must have $(2 * n) cols."))

    fill!(J, zero(eltype(J)))

    I_D, J_D, V_D = findnz(p.ops.D)
    for k in eachindex(V_D)
        J[I_D[k], n + J_D[k]] = convert(eltype(J), V_D[k])
    end

    I_G, J_G, V_G = findnz(p.ops.Geven)
    for k in eachindex(V_G)
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

Sparse Jacobian nonzero pattern for `wave_system_jac!` in stacked layout `[Π; Ψ]`.
"""
function wave_system_jac_prototype(
        ops::WaveOperators;
        boundary_condition::Symbol = :absorbing
    )
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
