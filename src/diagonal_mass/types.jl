"""
    SphericalOperators{T, Ti}

Container of folded half-grid operators for spherical-symmetry SBP discretizations.

Fields:
- `r`: half-grid coordinates on `[0, R]`
- `S`: scalar (even-field) metric-weighted mass matrix
- `V`: vector (odd-field) metric-weighted mass matrix
- `B`: boundary operator with only outer boundary contribution
- `Geven`: folded gradient for even fields
- `Godd`: folded derivative for odd fields
- `D`: folded divergence for odd fluxes
- `closure_width`: estimated right-boundary closure width
- metadata (`accuracy_order`, `p`, `R`, `source`, `mode`, `atol`, `snap_factor`,
  `build_matrix`, `M_full`, `Nh`)
"""
struct SphericalOperators{T <: Real, Ti <: Integer}
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
    source
    mode
    atol::T
    snap_factor::Float64
    build_matrix::Symbol
    M_full::Int
    Nh::Int
end

function Base.getproperty(ops::SphericalOperators, name::Symbol)
    if name === :H
        return getfield(ops, :S)
    end
    return getfield(ops, name)
end

function Base.propertynames(::SphericalOperators, private::Bool = false)
    base = (
            :r,
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
            :build_matrix,
            :M_full,
            :Nh,
           )
    return private ? base : (base..., :H)
end
