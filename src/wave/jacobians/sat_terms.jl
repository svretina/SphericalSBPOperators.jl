function _sat_boundary_jacobian_entries(
        ops::WaveOperators;
        bc::Symbol
    )
    bc_norm = _normalize_boundary_condition(bc)
    if bc_norm === :none
        return (
            apply = false,
            dPi_dPi = 0.0,
            dPi_dXi = 0.0,
            dXi_dPi = 0.0,
            dXi_dXi = 0.0,
        )
    end

    BNN = Float64(ops.B[end, end])
    SNN = Float64(_wave_scalar_mass(ops)[end, end])
    VNN = Float64(_wave_vector_mass(ops)[end, end])
    SNN == 0.0 &&
        throw(ArgumentError("`S[end,end]` must be nonzero for Π SAT Jacobian terms."))
    VNN == 0.0 &&
        throw(ArgumentError("`V[end,end]` must be nonzero for Ψ SAT Jacobian terms."))
    invSN = 1.0 / SNN
    invVN = 1.0 / VNN

    dPi_dPi = 0.0
    dPi_dXi = 0.0
    dXi_dPi = 0.0
    dXi_dXi = 0.0

    if bc_norm === :absorbing
        coeff_pi = -(BNN / 4.0) * invSN
        coeff_xi = -(BNN / 4.0) * invVN
        dPi_dPi += coeff_pi
        dPi_dXi += coeff_pi
        dXi_dPi += coeff_xi
        dXi_dXi += coeff_xi
    elseif bc_norm === :reflecting
        dPi_dXi += -(BNN * invSN)
    elseif bc_norm === :dirichlet
        dXi_dPi += -(BNN * invVN)
    else
        throw(ArgumentError("Unsupported boundary condition `$bc_norm`. Use :absorbing, :reflecting, :dirichlet, or :none."))
    end

    return (
        apply = true,
        dPi_dPi = dPi_dPi,
        dPi_dXi = dPi_dXi,
        dXi_dPi = dXi_dPi,
        dXi_dXi = dXi_dXi,
    )
end
