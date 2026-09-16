"""Return a uniform grid spacing after verifying that all successive spacings agree."""
function _uniform_spacing(r::Vector{T}; atol::T) where {T <: AbstractFloat}
    length(r) >= 2 || throw(ArgumentError("At least two grid points are required."))
    Δr = r[2] - r[1]
    Δr > zero(T) || throw(ArgumentError("Grid spacing must be strictly positive."))
    tolerance = max(atol, T(512) * eps(T) * max(one(T), abs(Δr)))
    all(abs(r[i] - r[i - 1] - Δr) <= tolerance for i in 3:length(r)) ||
        throw(ArgumentError("Non-uniform half-grid detected."))
    Δr
end

function _uniform_spacing(r::Vector{T}; atol::T) where {T <: Real}
    length(r) >= 2 || throw(ArgumentError("At least two grid points are required."))
    Δr = r[2] - r[1]
    Δr > zero(T) || throw(ArgumentError("Grid spacing must be strictly positive."))
    all(r[i] - r[i - 1] == Δr for i in 3:length(r)) ||
        throw(ArgumentError("Non-uniform half-grid detected."))
    Δr
end

"""Choose the default output type for rescaling to radius `R`."""
_default_scale_eltype(R) = R isa AbstractFloat ? typeof(R) : Rational{BigInt}

"""Convert and scale sparse matrix `A`, then remove roundoff-scale entries."""
function _scale_sparse_matrix(A::SparseMatrixCSC{Ta, Ti}, factor, ::Type{Tb};
                              snap_factor::Float64) where {Ta <: Real, Ti <: Integer, Tb <: Real}
    I, J, values = findnz(A)
    scaled = Tb[convert(Tb, value) * convert(Tb, factor) for value in values]
    result = sparse(I, J, scaled, size(A)...)
    snap_sparse!(result; snap_factor = snap_factor)
end

"""Measure consecutive leading and trailing rows that differ from an interior stencil."""
function _closure_diagnostics(G::SparseMatrixCSC)
    n = size(G, 1)
    n == 0 && return (closure_width_right = 0, closure_width_left = 0)
    reference = clamp(fld(n, 2), 1, n)
    row_pattern(i) = findall(!iszero, Array(view(G, i, :))) .- i
    pattern = row_pattern(reference)
    left = 0
    for i in 1:n
        row_pattern(i) == pattern && break
        left += 1
    end
    right = 0
    for i in n:-1:1
        row_pattern(i) == pattern && break
        right += 1
    end
    (closure_width_right = right, closure_width_left = left)
end

"""
    _solve_exact_linear_system(A, b)

Solve `A*x = b` exactly over `Rational{BigInt}` using reduced row echelon form.
For a consistent underdetermined system, free variables are set to zero. An
inconsistent system raises `ArgumentError`.
"""
function _solve_exact_linear_system(A::Matrix{Rational{BigInt}}, b::Vector{Rational{BigInt}})
    size(A, 1) == length(b) || throw(DimensionMismatch("Incompatible linear-system dimensions."))
    M = hcat(copy(A), copy(b))
    pivot_columns = Int[]
    pivot_row = 1
    for col in axes(A, 2)
        pivot = findfirst(r -> M[r, col] != 0 // 1, pivot_row:size(M, 1))
        isnothing(pivot) && continue
        pivot = pivot_row - 1 + pivot
        M[pivot_row, :], M[pivot, :] = M[pivot, :], M[pivot_row, :]
        M[pivot_row, :] ./= M[pivot_row, col]
        for row in axes(M, 1)
            row == pivot_row && continue
            M[row, :] .-= M[row, col] .* M[pivot_row, :]
        end
        push!(pivot_columns, col)
        pivot_row += 1
        pivot_row > size(M, 1) && break
    end
    for row in axes(M, 1)
        all(M[row, col] == 0 // 1 for col in axes(A, 2)) && M[row, end] != 0 // 1 &&
            throw(ArgumentError("Exact constraint system is inconsistent."))
    end
    solution = zeros(Rational{BigInt}, size(A, 2))
    for (row, col) in enumerate(pivot_columns)
        solution[col] = M[row, end]
    end
    solution
end
