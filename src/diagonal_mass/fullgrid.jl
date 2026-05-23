function _build_full_grid_objects(source;
                                  accuracy_order::Int,
                                  N::Int,
                                  R,
                                  mode)
    N > 0 || throw(ArgumentError("`N` must be positive."))
    accuracy_order > 0 || throw(ArgumentError("`accuracy_order` must be positive."))

    Nfull = 2 * N + 1
    Dfull = derivative_operator(source;
                                derivative_order = 1,
                                accuracy_order = accuracy_order,
                                xmin = -R,
                                xmax = R,
                                N = Nfull,
                                mode = mode)

    xfull = collect(grid(Dfull))
    M = length(xfull)
    Gfull = sparse(Matrix(Dfull))
    size(Gfull) == (M, M) ||
        throw(DimensionMismatch("Extracted derivative matrix has size $(size(Gfull)); expected ($M, $M)."))
    Hfull = sparse(mass_matrix(Dfull))
    snap_sparse!(Gfull)
    snap_sparse!(Hfull)
    return Dfull, xfull, Gfull, Hfull
end

function _boundary_closure_width_from_operator(Dfull)
    if !hasproperty(Dfull, :coefficients)
        return nothing
    end
    coeffs = getproperty(Dfull, :coefficients)
    if hasproperty(coeffs, :right_weights)
        rw = getproperty(coeffs, :right_weights)
        return length(rw)
    end
    if hasproperty(coeffs, :left_weights)
        # Fallback for non-standard operators where right weights are unavailable.
        lw = getproperty(coeffs, :left_weights)
        return length(lw)
    end
    return nothing
end
