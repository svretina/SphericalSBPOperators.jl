function _probe_derivative_matrix(Dfull, M::Int)
    T = eltype(Dfull)
    Gdense = Matrix{T}(undef, M, M)
    e = fill(zero(T), M)
    @inbounds for j in 1:M
        e[j] = one(T)
        Gdense[:, j] = Dfull * e
        e[j] = zero(T)
    end
    return sparse(Gdense)
end

function _derivative_matrix(Dfull, M::Int, build_matrix::Symbol)
    build_matrix in (:probe, :matrix_if_square) ||
        throw(ArgumentError("`build_matrix` must be `:probe` or `:matrix_if_square`."))

    if build_matrix == :matrix_if_square
        maybe_dense = try
            Matrix(Dfull)
        catch
            nothing
        end
        if maybe_dense !== nothing && size(maybe_dense) == (M, M)
            return sparse(maybe_dense)
        end
    end

    return _probe_derivative_matrix(Dfull, M)
end

function _build_full_grid_objects(source;
                                  accuracy_order::Int,
                                  N::Int,
                                  R,
                                  mode,
                                  build_matrix::Symbol)
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
    Gfull = _derivative_matrix(Dfull, M, build_matrix)
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
