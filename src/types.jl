"""
    SphericalOperators{T, Ti}

Container of folded half-grid operators for spherical-symmetry SBP discretizations.

Fields:
- `r`: half-grid coordinates on `[0, R]`
- `H`: metric-weighted mass matrix
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
    H::SparseMatrixCSC{T, Ti}
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

