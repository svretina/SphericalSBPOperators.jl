@inline function _default_atol(::Type{T}) where {T <: AbstractFloat}
    return T(1.0e-12)
end

@inline function _default_atol(::Type{T}) where {T <: Real}
    return zero(T)
end

function _resolve_atol(::Type{T}, atol) where {T <: Real}
    if atol === nothing
        return _default_atol(T)
    end
    if T <: AbstractFloat
        return T(atol)
    end
    if atol isa Integer || atol isa Rational
        return convert(T, atol)
    end
    throw(ArgumentError("For non-floating arithmetic, `atol` must be integer/rational or `nothing`."))
end

function _maxabs(values)
    m = zero(eltype(values))
    for v in values
        av = abs(v)
        if av > m
            m = av
        end
    end
    return m
end

function _maxabs_sparse(A::SparseMatrixCSC{T, Ti}) where {T <: Real, Ti <: Integer}
    vals = findnz(A)[3]
    isempty(vals) && return zero(T)
    return _maxabs(vals)
end
