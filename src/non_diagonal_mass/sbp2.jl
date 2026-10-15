"""
    sbp2_vector_mass(s_diag)

Construct the diagonal vector norm for the second-order spherical SBP pair.
At second order the exact closure solve admits the common diagonal norm
`V = S`, so no off-diagonal vector-mass entries are required.
"""
function sbp2_vector_mass(s_diag::AbstractVector{T}) where {T <: Real}
    isempty(s_diag) && throw(ArgumentError("`s_diag` must be non-empty."))
    return spdiagm(0 => collect(s_diag))
end

"""Construct the second-order diagonal vector norm from a diagonal scalar norm."""
function sbp2_vector_mass(S::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
    return sbp2_vector_mass(_extract_diagonal(S))
end

"""
    sbp2_scalar_mass_gradient(source; accuracy_order=2, points=21, h=1,
                              N=points-1, R=h*(points-1), p=2, ...)

Build the folded Cartesian ingredients for the second-order diagonal-norm
closure. The construction uses the same exact-rational full-grid path as the
higher-order split-mass operators.
"""
function sbp2_scalar_mass_gradient(source;
                                   accuracy_order::Int = 2,
                                   points::Int = 21,
                                   h::Real = 1,
                                   N = points - 1,
                                   R = h * (points - 1),
                                   p::Int = 2,
                                   mode = SafeMode(),
                                   build_matrix::Symbol = :probe,
                                   atol = nothing)
    accuracy_order == 2 ||
        throw(ArgumentError("SBP2 requires `accuracy_order == 2`; got $accuracy_order."))
    return sbp6_scalar_mass_gradient(source;
                                     accuracy_order = accuracy_order,
                                     points = points,
                                     h = h,
                                     N = N,
                                     R = R,
                                     p = p,
                                     mode = mode,
                                     build_matrix = build_matrix,
                                     atol = atol)
end

"""
    sbp2_construct_divergence(S, V, Geven, r; p=2)

Construct the compatible second-order covariant divergence. This is a named
wrapper around the common diagonal-mass SBP construction.
"""
@inline sbp2_construct_divergence(args...; kwargs...) = sbp6_construct_divergence(args...; kwargs...)

"""
    sbp2_solve_accuracy_constraints(setup; exact_solve=true, verbose=true)

Solve the second-order spherical SBP closure over `Rational{BigInt}`. Both
norms are diagonal. Requiring the common norm `S = V` and exactness of
`D*r = (p+1)` gives the rational system

```math
((p+1)I + G^T\\operatorname{diag}(r))s = B r.
```

The final quadrature row is included explicitly; it is implied by the other
rows because the folded gradient annihilates constants. The solution has
strictly positive entries for the supported folded SBP grids.
"""
function sbp2_solve_accuracy_constraints(setup::NamedTuple;
                                         exact_solve::Bool = true,
                                         verbose::Bool = true)
    hasproperty(setup, :r) || throw(ArgumentError("`setup` must include `r`."))
    hasproperty(setup, :Geven) || throw(ArgumentError("`setup` must include `Geven`."))
    hasproperty(setup, :p) || throw(ArgumentError("`setup` must include `p`."))
    hasproperty(setup, :R) || throw(ArgumentError("`setup` must include `R`."))
    exact_solve ||
        throw(ArgumentError("Only exact rational solve is supported; set `exact_solve=true`."))

    p = Int(setup.p)
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    N = length(setup.r)
    N >= 2 || throw(ArgumentError("SBP2 requires at least two half-grid points."))
    size(setup.Geven) == (N, N) ||
        throw(DimensionMismatch("`Geven` must be $(N)x$(N)."))

    Tq = Rational{BigInt}
    r = Tq[_as_big_rational(ri) for ri in setup.r]
    I, J, values = findnz(sparse(setup.Geven))
    G = sparse(I, J, Tq[_as_big_rational(value) for value in values], N, N)
    R = _as_big_rational(setup.R)

    # D*r = (p+1) and D = S^{-1}(B - G'V), with S = V = diag(s).
    A = zeros(Tq, N + 1, N)
    b = zeros(Tq, N + 1)
    @inbounds for i in 1:N
        A[i, i] = p + 1
        for j in 1:N
            A[i, j] += G[j, i] * r[j]
        end
    end
    b[N] = r[end]^(p + 1)

    # Keep the volume quadrature condition in the exact system as a useful
    # consistency check, even though summing the divergence rows implies it.
    A[end, :] .= one(Tq)
    b[end] = R^(p + 1) / (p + 1)
    tags = vcat([(family = :divergence, degree = 1, row = i) for i in 1:N],
                [(family = :quadrature, degree = 0, row = 0)])

    sdiag = _solve_exact_linear_system(A, b)
    residual = _maxabs(abs.(A * sdiag .- b))
    all(si -> si > zero(Tq), sdiag) ||
        throw(ArgumentError("Second-order diagonal-norm solve did not produce a positive scalar norm."))

    S = spdiagm(0 => sdiag)
    V = sbp2_vector_mass(sdiag)
    div_data = sbp2_construct_divergence(S, V, G, r; p = p)
    D = div_data.D
    B = div_data.B

    rows = collect(1:N)
    err_r = _constraint_error(D, r, p, 1, rows)
    quadrature_target = R^(p + 1) / (p + 1)
    quadrature_error = Float64(abs(sum(sdiag) - quadrature_target))
    s_symmetric = _is_symmetric(S)
    s_positive_definite = _is_positive_definite(S)
    v_symmetric = _is_symmetric(V)
    v_positive_definite = _is_positive_definite(V)

    if verbose
        println("SBP2 solve mode: exact")
        println("  equations = ", size(A, 1), ", unknowns = ", size(A, 2),
                ", exact residual = ", residual)
        println("  constraint max errors: D*r = ", err_r,
                ", ones.S.ones = ", quadrature_error)
        println("  diagonal common norm: S == V = ", S == V,
                ", SPD = ", v_positive_definite)
    end

    return (mode = :exact,
            setup = setup,
            row_sets = (rows_r = rows,),
            A = A,
            b = b,
            tags = tags,
            x = sdiag,
            residual = residual,
            S = S,
            V = V,
            D = D,
            B = B,
            errors = (Dr_all_rows = err_r,
                      ones_S_ones = quadrature_error),
            spd = (S_symmetric = s_symmetric,
                   S_positive_definite = s_positive_definite,
                   V_symmetric = v_symmetric,
                   V_positive_definite = v_positive_definite,
                   common_diagonal_norm = S == V))
end

"""Source-based convenience overload for the exact second-order closure solve."""
function sbp2_solve_accuracy_constraints(source;
                                         accuracy_order::Int = 2,
                                         points::Int = 21,
                                         h::Real = 1,
                                         N = points - 1,
                                         R = h * (points - 1),
                                         p::Int = 2,
                                         mode = SafeMode(),
                                         build_matrix::Symbol = :probe,
                                         atol = nothing,
                                         exact_solve::Bool = true,
                                         verbose::Bool = true)
    setup = sbp2_scalar_mass_gradient(source;
                                      accuracy_order = accuracy_order,
                                      points = points,
                                      h = h,
                                      N = N,
                                      R = R,
                                      p = p,
                                      mode = mode,
                                      build_matrix = build_matrix,
                                      atol = atol)
    return sbp2_solve_accuracy_constraints(setup;
                                           exact_solve = exact_solve,
                                           verbose = verbose)
end

"""Construct second-order `(D, G, S, V, B)` operators with the exact diagonal closure."""
function sbp2_operators(source,
                        points::Integer;
                        h::Real = 1,
                        accuracy_order::Int = 2,
                        p::Int = 2,
                        mode = SafeMode(),
                        atol = nothing,
                        exact_solve::Bool = true,
                        verbose::Bool = false)
    solved = sbp2_solve_accuracy_constraints(source;
                                             accuracy_order = accuracy_order,
                                             points = Int(points),
                                             h = h,
                                             p = p,
                                             mode = mode,
                                             atol = atol,
                                             exact_solve = exact_solve,
                                             verbose = verbose)
    return (D = solved.D,
            G = solved.setup.Geven,
            S = solved.S,
            V = solved.V,
            B = solved.B)
end
