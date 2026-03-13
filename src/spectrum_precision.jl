@inline _spectrum_as_big_rational(x::Rational{BigInt}) = x
@inline _spectrum_as_big_rational(x::Integer) = big(x) // 1
@inline _spectrum_as_big_rational(x::Rational{<:Integer}) = big(numerator(x)) // big(denominator(x))

function _spectrum_as_big_rational(x::AbstractFloat)
    isfinite(x) || throw(ArgumentError("Cannot convert non-finite floating-point value `$x` to Rational{BigInt}."))
    return rationalize(BigInt, BigFloat(x))
end

function _spectrum_as_big_rational(x::Real)
    throw(ArgumentError("Could not convert value of type $(typeof(x)) to Rational{BigInt}."))
end

function _spectrum_rationalize_matrix(A::AbstractMatrix{<:Real})
    m, n = size(A)
    Aq = Matrix{Rational{BigInt}}(undef, m, n)
    @inbounds for j in 1:n
        for i in 1:m
            Aq[i, j] = _spectrum_as_big_rational(A[i, j])
        end
    end
    return Aq
end

function _spectrum_float64x4_matrix(A::AbstractMatrix{<:Real})
    Aq = _spectrum_rationalize_matrix(A)
    return Matrix{Float64x4}(Aq)
end

function _high_precision_schur_values(A::AbstractMatrix{<:Real})
    Ahp = _spectrum_float64x4_matrix(A)
    return LinearAlgebra.schur(Ahp).values
end

function _high_precision_eigen(A::AbstractMatrix{<:Real})
    Ahp = _spectrum_float64x4_matrix(A)
    return LinearAlgebra.eigen(Ahp)
end
