module SphericalSBPOperators

using LinearAlgebra: dot, norm
import LinearAlgebra
using SparseArrays: SparseMatrixCSC, dropzeros!, findnz, sparse, spdiagm, spzeros
using SummationByPartsOperators:
                                 SafeMode,
                                 FastMode,
                                 MattssonNordström2004,
                                 derivative_operator,
                                 grid,
                                 mass_matrix
export spherical_operators
export AbstractSphericalOperators
export NonDiagonalMassSphericalOperators, StaggeredSphericalOperators
export scalar_mass, vector_mass, has_origin_node
export apply_even_gradient, apply_odd_derivative, apply_divergence
export validate_staggered, diagnose_staggered, interpret_diagnostics_staggered

include("operators/types.jl")
const SphericalOperators = DiagonalMassSphericalOperators
include("core/snap.jl")
include("core/fullgrid.jl")
include("core/folding.jl")
include("core/operator_utils.jl")
include("core/sbp6_support.jl")
include("non_diagonal_mass/sbp4.jl")
include("non_diagonal_mass/sbp6.jl")
include("non_diagonal_mass/NonDiagonalMass.jl")
include("staggered/Staggered.jl")
include("operators/api.jl")

"""
    SphericalSBPOperators

Build spherical-symmetry SBP operators on `[0, R]` by folding Cartesian SBP operators
constructed on the mirrored grid `[-R, R]`.

Parity conventions at the origin:
- scalar-like fields are even under reflection;
- radial flux-like fields are odd under reflection, so `u(0) = 0`.

Accordingly:
- `Geven` maps even fields to odd derivatives;
- `D` maps odd radial fluxes to even divergence values.

The metric-weighted SBP masses are
```math
S = H_{\\mathrm{cart,half}}\\,\\mathrm{diag}(r^p), \\qquad
V = H_{\\mathrm{cart,half}}\\,\\mathrm{diag}(r^p),
```
and the discrete SBP relation is
```math
S D + G^T V = B, \\quad B = \\mathrm{diag}(0,\\dots,0,R^p).
```

For `p > 0`, `S[1,1] = 0` (and likewise `V[1,1] = 0`) at `r = 0`, so SBP does not constrain the origin row of
`D`. This row is fixed using the removable-singularity condition for odd fluxes:
```math
(Du)(0) = (p+1)u'(0),
```
implemented as
```math
D[1,:] = (p+1)\\,G_{\\mathrm{odd}}[1,:].
```

Rational arithmetic is supported through type inference from inputs (e.g. `R = 1//1`),
matching `SummationByPartsOperators`.
"""
SphericalSBPOperators

end
