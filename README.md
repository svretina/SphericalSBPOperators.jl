# SphericalSBPOperators.jl

[![CI](https://github.com/svretina/SphericalSBPOperators.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/svretina/SphericalSBPOperators.jl/actions/workflows/CI.yml)
[![Documentation build](https://github.com/svretina/SphericalSBPOperators.jl/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/svretina/SphericalSBPOperators.jl/actions/workflows/docs.yml)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://svretina.github.io/SphericalSBPOperators.jl/)
[![Julia: 1.11+](https://img.shields.io/badge/Julia-1.11%2B-9558B2.svg?logo=julia)](https://julialang.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20400455.svg)](https://doi.org/10.5281/zenodo.20400455)

High-order, energy-stable summation-by-parts (SBP) finite-difference operators
for radial problems with an (r^{-p}) coordinate singularity. The package
constructs collocated operators with a node at the origin and staggered
operators that straddle it.

It is intended for first-order hyperbolic systems, wave propagation, and other
problems requiring compatible radial gradient and divergence operators.

## Features

- Collocated non-diagonal-norm SBP operators of accuracy order 4 and 6.
- Staggered SBP comparison operators of accuracy order 4 and 6.
- Exact rational construction for published SBP4/SBP6 coefficients.
- A single public constructor, `spherical_operators`.
- Sparse matrices, stable operator container types, and type-stable operator
  application for a fixed numeric input type.
- Support for general integer metric powers `p`; `p=2` is spherical symmetry
  and `p=1` is cylindrical symmetry.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/svretina/SphericalSBPOperators.jl")
```

The package builds on
[SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl);
choose a Cartesian SBP source from that package.

## Quick start

```julia
using SphericalSBPOperators
using SummationByPartsOperators: MattssonNordström2004

source = MattssonNordström2004()

# Collocated paper SBP6 on r ∈ [0, 1]. N is the number of subintervals.
ops = spherical_operators(source;
    accuracy_order = 6,
    N = 64,
    R = 1.0,
    p = 2,
    grid = :collocated,
)

u = ops.r .^ 3                 # an odd radial flux
div_u = apply_divergence(ops, u)
exact = 5 .* ops.r .^ 2        # (∂r + 2/r) r³
```

`ops` contains the grid `r`, scalar and vector mass matrices, the boundary
matrix, and compatible gradient/divergence matrices. Prefer the stable generic
accessors when writing downstream code:

```julia
scalar_mass(ops)     # S
vector_mass(ops)     # V
has_origin_node(ops) # true for collocated, false for staggered
```

## Theory in brief

For regularized radial variables, the divergence has the form

```math
\mathcal D_p u = \frac{1}{r^p}\partial_r(r^p u)
                 = \partial_r u + \frac{p}{r}u.
```

The apparent singularity at `r = 0` is a coordinate effect. Regular scalar
variables are even under reflection through the origin; radial-flux variables
are odd. The package folds a Cartesian SBP operator on `[-R, R]` using these
parities and constructs matrices satisfying the discrete integration-by-parts
identity

```math
S D + G^T V = B, \qquad B = \operatorname{diag}(0,\ldots,0,R^p).
```

Here `G` is the even-to-odd gradient, `D` is the odd-to-even covariant
divergence, and `S` and `V` are the scalar and vector mass matrices. Thus the
semidiscrete energy changes only through the outer boundary. For the supported
collocated operators, `S` is diagonal and `V` has a small symmetric
off-diagonal closure near the origin; this permits high-order accuracy together
with the SBP identity.

See the [documentation site](docs/src/index.md) for the derivation, parity
conventions, and the relation to the publication.

## Inspecting the origin closure

Use a deliberately small grid to inspect only the rows touched by the origin
closure. This is useful for teaching, debugging, and reproducing paper tables;
it is not a recommended production resolution.

```julia
using LinearAlgebra: norm

small = spherical_operators(source;
    accuracy_order = 4, N = 16, R = 16//1, p = 2, grid = :collocated)

small.r[1:7]
Matrix(small.Geven[1:7, 1:7])
Matrix(small.D[1:7, 1:7])
Matrix(vector_mass(small)[1:7, 1:7])

# Exact rational arithmetic makes the SBP residual exactly zero.
norm(Matrix(scalar_mass(small) * small.D + small.Geven' * vector_mass(small) - small.B))
```

The first grid point is exactly zero. `Geven` differentiates even scalar data,
whereas `D` accepts odd flux data. Do not apply `D` to arbitrary values at the
origin: an odd flux must satisfy `u[1] == 0` on a collocated grid.

## Collocated and staggered grids

```julia
# Node at r = 0; N means subintervals and yields N + 1 nodes.
collocated = spherical_operators(source;
    accuracy_order = 4, N = 32, R = 1.0, p = 2, grid = :collocated)

# No origin node; N means staggered half-grid nodes.
staggered = spherical_operators(source;
    accuracy_order = 4, N = 32, R = 1.0, p = 2, grid = :staggered)
```

Choose `:collocated` when values at the origin are part of the state and its
regularity is represented by parity. Choose `:staggered` when a grid that avoids
the origin is more natural for the surrounding discretization. Both supported
families satisfy their intended SBP relation; `method = :naive` in staggered
mode is available only as a comparison construction and is not an SBP method.

## Numeric types and performance

The construction follows Julia’s numeric types:

```julia
float_ops = spherical_operators(source;
    accuracy_order = 4, N = 32, R = 1.0, p = 2)

exact_ops = spherical_operators(source;
    accuracy_order = 4, N = 32, R = 1//1, p = 2)
```

`float_ops` stores `Float64` sparse matrices; `exact_ops` stores
`Rational{BigInt}` coefficients and is appropriate for reproducibility checks,
not large production runs. Operator containers encode their matrix element and
index types. Matrix application is inference-stable for a fixed operator and
input vector type:

```julia
using Test
x = ones(length(float_ops.r))
@inferred apply_even_gradient(float_ops, x) # Vector{Float64}
```

Construct operators once and reuse them. Exact coefficient construction is
deliberately more expensive than floating-point construction.

## Scope and companion packages

This package deliberately contains operator construction, scaling, and
validation only. The following companion packages are staged for extraction and
will be linked here when released:

- `SphericalSBPWave.jl` — semidiscrete wave systems, boundary conditions, and
  ODE integration.
- `SphericalSBPPlots.jl` — Makie-based visualization and publication figures.
- `SphericalSBPAnalysis.jl` — convergence, spectrum, and post-processing tools.

Their current handoff source is in [companion-packages](companion-packages/README.md).
Experimental operator prototypes remain in `src/experimental/` and are not part
of the loaded package or public API.

## Reproducibility and citation

The construction parameters and exact coefficient path for the publication are
recorded in [PAPER_OPERATOR_REPRODUCTION.md](PAPER_OPERATOR_REPRODUCTION.md).
Please cite the associated publication and software record when using this
package in research.

## Documentation

Build the local documentation site with:

```julia
using Pkg
Pkg.activate("docs")
Pkg.develop(path = ".")
Pkg.instantiate()
include("docs/make.jl")
```

The generated HTML entry point is `docs/build/index.html`.
