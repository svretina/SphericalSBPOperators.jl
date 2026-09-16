abstract type AbstractWaveOperatorKind end
struct WaveDOperator <: AbstractWaveOperatorKind end
struct WaveGevenOperator <: AbstractWaveOperatorKind end

const _WAVE_KERNEL_SUPPORTED_ORDERS = (4, 6, 8)

struct WaveMatrixKernel{KindT <: AbstractWaveOperatorKind,
                        Order,
                        OpsT,
                        AT <: AbstractMatrix,
                        T <: Real,
                        Ti <: Integer,
                        OffsetsT <: Tuple}
    parent::OpsT
    matrix::AT
    offsets::OffsetsT
                        band_coefficients::Matrix{T}
                        rowptr::Vector{Ti}
                        colind::Vector{Ti}
                        nzval::Vector{T}
                        interior_start::Int
                        interior_end::Int
end

Base.size(K::WaveMatrixKernel) = size(K.matrix)
Base.size(K::WaveMatrixKernel, d::Integer) = size(K.matrix, d)
Base.eltype(::Type{<:WaveMatrixKernel{KindT, Order, OpsT, AT, T}}) where {
                                                                           KindT,
                                                                           Order,
                                                                           OpsT,
                                                                           AT,
                                                                           T} = T
Base.eltype(K::WaveMatrixKernel) = eltype(typeof(K))

@inline _wave_kernel_offsets(::Val{4}) = (-2, -1, 1, 2)
@inline _wave_kernel_offsets(::Val{6}) = (-3, -2, -1, 1, 2, 3)
@inline _wave_kernel_offsets(::Val{8}) = (-4, -3, -2, -1, 1, 2, 3, 4)

@inline _wave_kernel_matrix(::WaveDOperator, ops) = ops.D
@inline _wave_kernel_matrix(::WaveGevenOperator, ops) = ops.Geven

function _wave_kernel_check_order(order::Int)
    order in _WAVE_KERNEL_SUPPORTED_ORDERS ||
        throw(ArgumentError("Wave matrix kernels currently support accuracy_order ∈ $(_WAVE_KERNEL_SUPPORTED_ORDERS); got $order."))
    return order
end

function _wave_kernel_rows(A::SparseMatrixCSC{TA, Ti},
                           ::Type{T}) where {TA <: Real, Ti <: Integer, T <: Real}
    n, m = size(A)
    row_counts = zeros(Ti, n)
    @inbounds for j in 1:m
        for ptr in A.colptr[j]:(A.colptr[j + 1] - 1)
            row_counts[A.rowval[ptr]] += one(Ti)
        end
    end

    rowptr = Vector{Ti}(undef, n + 1)
    rowptr[1] = one(Ti)
    @inbounds for i in 1:n
        rowptr[i + 1] = rowptr[i] + row_counts[i]
    end

    nnzA = length(A.nzval)
    colind = Vector{Ti}(undef, nnzA)
    nzval = Vector{T}(undef, nnzA)
    nextptr = copy(rowptr)
    @inbounds for j in 1:m
        for ptr in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[ptr]
            dest = nextptr[i]
            colind[dest] = convert(Ti, j)
            nzval[dest] = convert(T, A.nzval[ptr])
            nextptr[i] += one(Ti)
        end
    end

    return rowptr, colind, nzval
end

@inline function _wave_row_matches_offsets(i::Int,
                                           rowptr::AbstractVector{<:Integer},
                                           colind::AbstractVector{<:Integer},
                                           offsets::Tuple)
    first = Int(rowptr[i])
    last = Int(rowptr[i + 1]) - 1
    length(offsets) == last - first + 1 || return false
    @inbounds for k in eachindex(offsets)
        colind[first + k - 1] == i + offsets[k] || return false
    end
    return true
end

function _wave_kernel_interior_range(rowptr::AbstractVector{<:Integer},
                                     colind::AbstractVector{<:Integer},
                                     offsets::Tuple)
    n = length(rowptr) - 1
    ref = clamp(fld(n, 2), 1, n)
    _wave_row_matches_offsets(ref, rowptr, colind, offsets) ||
        throw(ArgumentError("Could not find expected wave-kernel band pattern at reference row $ref."))

    first = ref
    while first > 1 && _wave_row_matches_offsets(first - 1, rowptr, colind, offsets)
        first -= 1
    end

    last = ref
    while last < n && _wave_row_matches_offsets(last + 1, rowptr, colind, offsets)
        last += 1
    end

    return first, last
end

function _wave_kernel_band_coefficients(A::SparseMatrixCSC,
                                        rowptr::AbstractVector{Ti},
                                        colind::AbstractVector{Ti},
                                        nzval::AbstractVector{T},
                                        offsets::Tuple) where {T <: Real, Ti <: Integer}
    n = size(A, 1)
    coeffs = zeros(T, length(offsets), n)
    @inbounds for i in 1:n
        first = Int(rowptr[i])
        last = Int(rowptr[i + 1]) - 1
        for ptr in first:last
            rel = Int(colind[ptr]) - i
            for k in eachindex(offsets)
                if rel == offsets[k]
                    coeffs[k, i] = nzval[ptr]
                    break
                end
            end
        end
    end
    return coeffs
end

@inline function _default_wave_kernel_eltype(A::SparseMatrixCSC{T}) where {T <: AbstractFloat}
    return T
end

@inline function _default_wave_kernel_eltype(::SparseMatrixCSC)
    return Float64
end

function _wave_matrix_kernel(kind::KindT,
                             ops::Union{SphericalOperators,
                                        NonDiagonalMass.SphericalOperators};
                             accuracy_order::Integer = ops.accuracy_order,
                             target_eltype::Type{T} = _default_wave_kernel_eltype(_wave_kernel_matrix(kind, ops))) where {
                                                                                  KindT <:
                                                                                  AbstractWaveOperatorKind,
                                                                                  T <:
                                                                                  Real}
    order = _wave_kernel_check_order(Int(accuracy_order))
    A = _wave_kernel_matrix(kind, ops)
    rowptr, colind, nzval = _wave_kernel_rows(A, T)
    offsets = _wave_kernel_offsets(Val(order))
    interior_start, interior_end = _wave_kernel_interior_range(rowptr, colind, offsets)
    band_coefficients = _wave_kernel_band_coefficients(A, rowptr, colind, nzval, offsets)
    return WaveMatrixKernel{KindT, order, typeof(ops), typeof(A), T, eltype(rowptr),
                            typeof(offsets)}(ops,
                                             A,
                                             offsets,
                                             band_coefficients,
                                             rowptr,
                                             colind,
                                             nzval,
                                             interior_start,
                                             interior_end)
end

wave_D_kernel(ops::Union{SphericalOperators, NonDiagonalMass.SphericalOperators};
              kwargs...) = _wave_matrix_kernel(WaveDOperator(), ops; kwargs...)

wave_Geven_kernel(ops::Union{SphericalOperators, NonDiagonalMass.SphericalOperators};
                  kwargs...) = _wave_matrix_kernel(WaveGevenOperator(), ops; kwargs...)

function wave_kernel_operators(ops::Union{SphericalOperators,
                                          NonDiagonalMass.SphericalOperators};
                               target_eltype::Type{T} = Float64) where {T <: Real}
    return WaveKernelOperators(ops,
                               wave_D_kernel(ops; target_eltype = T),
                               wave_Geven_kernel(ops; target_eltype = T))
end

@inline function _wave_kernel_row_sum(K::WaveMatrixKernel, x::AbstractVector, i::Int)
    s = zero(promote_type(eltype(K), eltype(x)))
    @inbounds for ptr in Int(K.rowptr[i]):(Int(K.rowptr[i + 1]) - 1)
        s += K.nzval[ptr] * x[K.colind[ptr]]
    end
    return s
end

@inline function _wave_kernel_band_sum(K::WaveMatrixKernel{KindT, 4},
                                       x::AbstractVector,
                                       i::Int) where {KindT}
    c = K.band_coefficients
    @inbounds return c[1, i] * x[i - 2] + c[2, i] * x[i - 1] +
                      c[3, i] * x[i + 1] + c[4, i] * x[i + 2]
end

@inline function _wave_kernel_band_sum(K::WaveMatrixKernel{KindT, 6},
                                       x::AbstractVector,
                                       i::Int) where {KindT}
    c = K.band_coefficients
    @inbounds return c[1, i] * x[i - 3] + c[2, i] * x[i - 2] +
                      c[3, i] * x[i - 1] + c[4, i] * x[i + 1] +
                      c[5, i] * x[i + 2] + c[6, i] * x[i + 3]
end

@inline function _wave_kernel_band_sum(K::WaveMatrixKernel{KindT, 8},
                                       x::AbstractVector,
                                       i::Int) where {KindT}
    c = K.band_coefficients
    @inbounds return c[1, i] * x[i - 4] + c[2, i] * x[i - 3] +
                      c[3, i] * x[i - 2] + c[4, i] * x[i - 1] +
                      c[5, i] * x[i + 1] + c[6, i] * x[i + 2] +
                      c[7, i] * x[i + 3] + c[8, i] * x[i + 4]
end

@inline function _wave_kernel_boundary_mul!(y::AbstractVector,
                                            K::WaveMatrixKernel,
                                            x::AbstractVector,
                                            alpha::Number,
                                            beta::Number,
                                            first_i::Int,
                                            last_i::Int)
    first_i > last_i && return y
    if iszero(beta)
        @inbounds for i in first_i:last_i
            y[i] = alpha * _wave_kernel_row_sum(K, x, i)
        end
    else
        @inbounds for i in first_i:last_i
            y[i] = alpha * _wave_kernel_row_sum(K, x, i) + beta * y[i]
        end
    end
    return y
end

@inline function _wave_kernel_mul_serial!(y::AbstractVector,
                                          K::WaveMatrixKernel,
                                          x::AbstractVector,
                                          alpha::Number,
                                          beta::Number)
    _wave_kernel_boundary_mul!(y, K, x, alpha, beta, 1, K.interior_start - 1)
    if iszero(beta)
        @inbounds for i in K.interior_start:K.interior_end
            y[i] = alpha * _wave_kernel_band_sum(K, x, i)
        end
    else
        @inbounds for i in K.interior_start:K.interior_end
            y[i] = alpha * _wave_kernel_band_sum(K, x, i) + beta * y[i]
        end
    end
    _wave_kernel_boundary_mul!(y, K, x, alpha, beta, K.interior_end + 1, size(K, 1))
    return y
end

function LinearAlgebra.mul!(y::AbstractVector,
                            K::WaveMatrixKernel,
                            x::AbstractVector,
                            alpha::Number,
                            beta::Number)
    n, m = size(K)
    length(y) == n || throw(DimensionMismatch("destination has length $(length(y)); expected $n."))
    length(x) == m || throw(DimensionMismatch("source has length $(length(x)); expected $m."))
    return _wave_kernel_mul_serial!(y, K, x, alpha, beta)
end

function LinearAlgebra.mul!(y::AbstractVector, K::WaveMatrixKernel, x::AbstractVector)
    return mul!(y, K, x, one(eltype(K)), zero(eltype(K)))
end

function wave_kernel_mul!(y::AbstractVector,
                          K::WaveMatrixKernel,
                          x::AbstractVector,
                          alpha::Number,
                          beta::Number)
    return mul!(y, K, x, alpha, beta)
end

function wave_kernel_mul!(y::AbstractVector, K::WaveMatrixKernel, x::AbstractVector)
    return mul!(y, K, x)
end

function wave_D_mul!(y::AbstractVector,
                     K::WaveMatrixKernel{WaveDOperator},
                     x::AbstractVector)
    return wave_kernel_mul!(y, K, x)
end

function wave_Geven_mul!(y::AbstractVector,
                         K::WaveMatrixKernel{WaveGevenOperator},
                         x::AbstractVector)
    return wave_kernel_mul!(y, K, x)
end

function wave_D_mul!(y::AbstractVector,
                     K::WaveMatrixKernel{WaveDOperator},
                     x::AbstractVector,
                     alpha::Number,
                     beta::Number)
    return wave_kernel_mul!(y, K, x, alpha, beta)
end

function wave_Geven_mul!(y::AbstractVector,
                         K::WaveMatrixKernel{WaveGevenOperator},
                         x::AbstractVector,
                         alpha::Number,
                         beta::Number)
    return wave_kernel_mul!(y, K, x, alpha, beta)
end
