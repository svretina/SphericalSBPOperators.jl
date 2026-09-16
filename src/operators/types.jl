abstract type AbstractSphericalOperators end

"""
    DiagonalMassSphericalOperators{T, Ti}

Container of folded half-grid operators for the diagonal-mass spherical-symmetry SBP
construction on `[0, R]`.
"""
struct DiagonalMassSphericalOperators{T <: Real, Ti <: Integer} <:
       AbstractSphericalOperators
    r::Vector{T}
    S::SparseMatrixCSC{T, Ti}
    V::SparseMatrixCSC{T, Ti}
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
    M_full::Int
    Nh::Int
end

"""
    NonDiagonalMassSphericalOperators{T, Ti}

Container of folded half-grid operators for the non-diagonal-mass spherical-symmetry
SBP construction on `[0, R]`.
"""
struct NonDiagonalMassSphericalOperators{T <: Real, Ti <: Integer} <:
       AbstractSphericalOperators
    r::Vector{T}
    H::SparseMatrixCSC{T, Ti}
    S::SparseMatrixCSC{T, Ti}
    V::SparseMatrixCSC{T, Ti}
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
    M_full::Int
    Nh::Int
end

"""
    StaggeredSphericalOperators{T, Ti}

Container of folded half-grid operators for the staggered spherical-symmetry SBP
construction on `[0, R]`.
"""
struct StaggeredSphericalOperators{T <: Real, Ti <: Integer} <: AbstractSphericalOperators
    r::Vector{T}
    H::SparseMatrixCSC{T, Ti}
    B::SparseMatrixCSC{T, Ti}
    Geven::SparseMatrixCSC{T, Ti}
    Godd::SparseMatrixCSC{T, Ti}
    D::SparseMatrixCSC{T, Ti}
    divergence_method::Symbol
    closure_width::Int
    accuracy_order::Int
    p::Int
    R::T
    source::Any
    mode::Any
    atol::T
    snap_factor::Float64
    M_full::Int
    Nh::Int
end

"""Return the scalar-field SBP mass matrix for an operator set."""
scalar_mass(ops::DiagonalMassSphericalOperators) = ops.S
scalar_mass(ops::NonDiagonalMassSphericalOperators) = ops.S
scalar_mass(ops::StaggeredSphericalOperators) = ops.H

"""Return the radial-flux SBP mass matrix for an operator set."""
vector_mass(ops::DiagonalMassSphericalOperators) = ops.V
vector_mass(ops::NonDiagonalMassSphericalOperators) = ops.V
vector_mass(ops::StaggeredSphericalOperators) = ops.H

"""Whether the half grid includes the origin."""
has_origin_node(ops::AbstractSphericalOperators) = first(ops.r) == zero(eltype(ops.r))
