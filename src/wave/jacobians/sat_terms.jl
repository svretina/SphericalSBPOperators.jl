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
    HNN = Float64(ops.H[end, end])
    HNN == 0.0 &&
        throw(ArgumentError("`H[end,end]` must be nonzero for SAT Jacobian terms."))
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
        dXi_dXi = dXi_dXi,
    )
end
