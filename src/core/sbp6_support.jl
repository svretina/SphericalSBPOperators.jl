# Shared exact-arithmetic support for the supported SBP6 construction.  These
# routines deliberately live in `core`: the publication implementation must
# not depend on experimental operator families.

"""Convert an exactly representable real input to `Rational{BigInt}`."""
@inline _as_big_rational(x::Rational{BigInt}) = x
@inline _as_big_rational(x::Integer) = big(x) // 1
@inline _as_big_rational(x::Rational{<:Integer}) = big(numerator(x)) // big(denominator(x))
function _as_big_rational(x::AbstractFloat)
    isfinite(x) || throw(ArgumentError("Cannot convert non-finite value `$x` to Rational{BigInt}."))
    rationalize(BigInt, x)
end
_as_big_rational(x::Real) = throw(ArgumentError("Cannot convert $(typeof(x)) to Rational{BigInt}."))

"""Extract the diagonal of `S`, rejecting non-negligible off-diagonal entries."""
function _extract_diagonal(S::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
    n, m = size(S)
    n == m || throw(DimensionMismatch("`S` must be square."))
    diagonal = fill(zero(T), n)
    max_offdiagonal = zero(T)
    I, J, values = findnz(S)
    @inbounds for k in eachindex(values)
        if I[k] == J[k]
            diagonal[I[k]] = values[k]
        else
            max_offdiagonal = max(max_offdiagonal, abs(values[k]))
        end
    end
    if T <: AbstractFloat
        tolerance = max(_resolve_atol(T, nothing), T(256) * eps(T))
        max_offdiagonal <= tolerance ||
            throw(ArgumentError("`S` must be diagonal (max off-diagonal magnitude = $max_offdiagonal)."))
    else
        max_offdiagonal == zero(T) || throw(ArgumentError("`S` must be exactly diagonal."))
    end
    diagonal
end

"""Return the maximum selected-row error in `D*r^degree = (p+degree)r^(degree-1)`."""
function _constraint_error(D::SparseMatrixCSC, r::AbstractVector, p::Int,
                           degree::Int, rows::Vector{Int})
    isempty(rows) && return 0.0
    exact = (p + degree) .* (r .^ (degree - 1))
    maximum(abs.(Float64.((D * (r .^ degree))[rows] .- exact[rows])))
end

"""Test sparse-matrix symmetry exactly or to `tol` for floating-point entries."""
function _is_symmetric(A::SparseMatrixCSC{T, Ti}; tol::Float64 = 1e-12) where {T <: Real, Ti <: Integer}
    T <: AbstractFloat ? norm(Matrix(A - transpose(A)), Inf) <= tol : Matrix(A) == transpose(Matrix(A))
end

"""Return whether square matrix `A` is positive definite by a Cholesky test."""
function _is_positive_definite(A::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
    size(A, 1) == size(A, 2) || return false
    try
        LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Matrix{Float64}(A)); check = true)
        true
    catch
        false
    end
end

"""
    sbp6_scalar_mass_gradient(source; accuracy_order=6, points=21, h=1,
                              N=points-1, R=h*(points-1), p=2, ...)

Construct the folded Cartesian ingredients used by the supported SBP6 closure:
the grid, even/odd gradients, Cartesian half mass, and diagonal scalar mass.
The radius is converted to exact rational form so the subsequent closure solve
can reproduce publication coefficients exactly.
"""
function sbp6_scalar_mass_gradient(source;
                                   accuracy_order::Int = 6,
                                   points::Int = 21,
                                   h::Real = 1,
                                   N = points - 1,
                                   R = h * (points - 1),
                                   p::Int = 2,
                                   mode = SafeMode(),
                                   build_matrix::Symbol = :probe,
                                   atol = nothing)
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))
    Nint = Int(N)
    Nint > 0 || throw(ArgumentError("`N` must be positive."))
    points > 1 || throw(ArgumentError("`points` must be > 1."))
    Rq = _as_big_rational(R)
    Dfull, xfull, Gfull, Hfull = _build_full_grid_objects(source;
                                                           accuracy_order = accuracy_order,
                                                           N = Nint, R = Rq, mode = mode,
                                                           build_matrix = build_matrix)
    T = eltype(xfull)
    atol_use = _resolve_atol(T, atol)
    r, Rop, Eeven, Eodd = _build_folding_operators(xfull; atol = atol_use)
    Geven = sparse(Rop * Gfull * Eeven)
    Godd = sparse(Rop * Gfull * Eodd)
    Hcart_half = sparse((convert(T, 1) / convert(T, 2)) * (transpose(Eeven) * Hfull * Eeven))
    S = sparse(Hcart_half * spdiagm(0 => r .^ p))
    (r = r, Geven = Geven, Godd = Godd, S = S, Sdiag = _extract_diagonal(S),
     Dfull = Dfull, xfull = xfull, Gfull = Gfull, Hfull = Hfull,
     Hcart_half = Hcart_half, Rop = Rop, Eeven = Eeven, Eodd = Eodd,
     p = p, accuracy_order = accuracy_order, R = Rq, mode = mode,
     atol = atol_use, build_matrix = build_matrix)
end

"""
    sbp6_construct_divergence(S, V, Geven, r; p=2)

Construct the covariant divergence `D = S⁻¹(B - Geven'V)` and the outer-boundary
matrix `B` with `B[end,end] = r[end]^p`. `S` must be diagonal and nonsingular.
"""
function sbp6_construct_divergence(S::SparseMatrixCSC{T, Ti}, V::SparseMatrixCSC{T, Ti},
                                   Geven::SparseMatrixCSC{T, Ti}, r::AbstractVector;
                                   p::Int = 2) where {T <: Real, Ti <: Integer}
    N = length(r)
    size(S) == (N, N) && size(V) == (N, N) && size(Geven) == (N, N) ||
        throw(DimensionMismatch("S, V, and Geven must be square matrices matching `r`."))
    Sdiag = _extract_diagonal(S)
    B = spzeros(T, N, N)
    B[end, end] = convert(T, r[end]^p)
    RHS = sparse(B - transpose(Geven) * V)
    I, J, values = findnz(RHS)
    Dvalues = similar(values)
    @inbounds for k in eachindex(values)
        Sdiag[I[k]] == zero(T) && throw(ArgumentError("Scalar mass is singular at row $(I[k])."))
        Dvalues[k] = values[k] / Sdiag[I[k]]
    end
    (D = sparse(I, J, Dvalues, N, N), B = B, RHS = RHS, Sdiag = Sdiag)
end
