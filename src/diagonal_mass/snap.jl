@inline function _default_atol(::Type{T}) where {T <: AbstractFloat}
    return T(1e-12)
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
    @inbounds for v in values
        av = abs(v)
        if av > m
            m = av
        end
    end
    return m
end

"""
    snap_sparse!(A; snap_factor=64.0)

Snap tiny floating-point entries in sparse matrices to exact zero and drop explicit zeros.

For `AbstractFloat` matrices:
- threshold: `snap_factor * eps(T) * max(maxabs(A.nzval), one(T))`
- entries with `abs(value) <= threshold` are set to `0`

For non-floating types (e.g. rationals), only `dropzeros!` is applied.
"""
function snap_sparse!(A::SparseMatrixCSC{T, Ti}; snap_factor::Float64 = 64.0) where {T <: Real, Ti <: Integer}
    if T <: AbstractFloat
        scale = _maxabs(A.nzval)
        threshold = T(snap_factor) * eps(T) * max(scale, one(T))
        @inbounds for k in eachindex(A.nzval)
            if abs(A.nzval[k]) <= threshold
                A.nzval[k] = zero(T)
            end
        end
    end
    dropzeros!(A)
    return A
end

