"""
    _build_full_grid_objects(source; accuracy_order, N, R, mode,
                             include_origin=true, build_matrix=:probe)

Construct a Cartesian first-derivative SBP operator on `[-R, R]` and return
`(Dfull, xfull, Gfull, Hfull)`. `include_origin=true` creates `2N + 1` full
nodes; `false` creates the `2N` nodes required by the staggered construction.
The returned derivative and mass matrices are sparse and snapped to remove
roundoff-scale explicit entries.
"""
function _build_full_grid_objects(source;
                                  accuracy_order::Int,
                                  N::Int,
                                  R,
                                  mode,
                                  include_origin::Bool = true,
                                  build_matrix::Symbol = :probe)
    N > 0 || throw(ArgumentError("`N` must be positive."))
    accuracy_order > 0 || throw(ArgumentError("`accuracy_order` must be positive."))

    Nfull = include_origin ? 2 * N + 1 : 2 * N
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

"""Infer an SBP boundary-closure width from an operator's coefficient metadata.

Returns `nothing` when the operator does not expose compatible coefficient
metadata.
"""
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
