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

function Base.getproperty(ops::DiagonalMassSphericalOperators, name::Symbol)
    if name === :H
        return getfield(ops, :S)
    end
    return getfield(ops, name)
end

function Base.propertynames(::DiagonalMassSphericalOperators, private::Bool = false)
    base = (:r,
            :S,
            :V,
            :B,
            :Geven,
            :Godd,
            :D,
            :closure_width,
            :accuracy_order,
            :p,
            :R,
            :source,
            :mode,
            :atol,
            :snap_factor,
            :M_full,
            :Nh)
    return private ? base : (base..., :H)
end

function Base.getproperty(ops::NonDiagonalMassSphericalOperators, name::Symbol)
    if name === :G
        return getfield(ops, :Geven)
    end
    return getfield(ops, name)
end

function Base.propertynames(::NonDiagonalMassSphericalOperators, private::Bool = false)
    base = (:r,
            :H,
            :S,
            :V,
            :B,
            :Geven,
            :Godd,
            :D,
            :closure_width,
            :accuracy_order,
            :p,
            :R,
            :source,
            :mode,
            :atol,
            :snap_factor,
            :M_full,
            :Nh)
    return private ? base : (base..., :G)
end
