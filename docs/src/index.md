# SphericalSBPOperators.jl

`SphericalSBPOperators.jl` provides high-order summation-by-parts finite
difference operators for radial domains with a coordinate singularity at the
origin. It supports collocated grids containing the origin and staggered grids
that avoid it.

```@contents
Pages = ["theory.md", "examples.md", "api.md"]
Depth = 2
```

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/<organisation>/SphericalSBPOperators.jl")
```

## The public constructor

```@example start
using SphericalSBPOperators
using SummationByPartsOperators: MattssonNordström2004

source = MattssonNordström2004()
ops = spherical_operators(source;
    accuracy_order = 6,
    N = 64,
    R = 1.0,
    p = 2,
    grid = :collocated,
)
```

The package’s stable surface is deliberately small: construct an operator set
with `spherical_operators`, use the generic mass/origin accessors, and apply
the compatible derivative operators. The low-level coefficient construction is
kept internal so that downstream applications need not depend on a particular
closure representation. See the [API reference](@ref) for function signatures
and contracts.

The publication reproduction parameters and exact rational-coefficient path
are documented in the repository’s `PAPER_OPERATOR_REPRODUCTION.md`.
