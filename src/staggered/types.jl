"""
    SphericalOperators{T, Ti}

Container for staggered-grid folded spherical operators on `[0, R]`.

Compared to the collocated construction, `r[1] > 0` and all diagonal entries of `H`
are nonzero, so all rows of `D` are derived directly from the SBP relation.
"""
struct SphericalOperators{T <: Real, Ti <: Integer}
    r::Vector{T}
    H::SparseMatrixCSC{T, Ti}
    B::SparseMatrixCSC{T, Ti}
    Geven::SparseMatrixCSC{T, Ti}
    Godd::SparseMatrixCSC{T, Ti}
    D::SparseMatrixCSC{T, Ti}
    closure_width::Int
    accuracy_order::Int
    p::Int
    R::T
    source::Any
    mode::Any
    atol::T
    snap_factor::Float64
    build_matrix::Symbol
    M_full::Int
    Nh::Int
end
