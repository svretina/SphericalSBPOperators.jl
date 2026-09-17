# Examples

## Construct a collocated paper operator

```@example collocated
using SphericalSBPOperators
using SummationByPartsOperators: MattssonNordström2004

source = MattssonNordström2004()
ops = spherical_operators(source;
    accuracy_order = 4, N = 32, R = 1.0, p = 2, grid = :collocated)

(nodes = length(ops.r), origin = ops.r[1], outer_radius = ops.r[end],
 has_origin = has_origin_node(ops))
```

Apply the compatible divergence to an odd flux. The input must vanish at the
origin, which `u = r^3` does automatically.

```@example collocated
u = ops.r .^ 3
numerical = apply_divergence(ops, u)
exact = 5 .* ops.r .^ 2
maximum(abs.(numerical - exact))
```

The largest error includes the outer boundary closure. In the interior and
near-origin closure, the construction’s exact moment conditions are the more
meaningful diagnostic.

## Inspect a small origin closure

Use rational inputs and a small grid when inspecting coefficients. This mirrors
the paper’s coefficient presentation while keeping the displayed blocks small.

```@example closure
using SphericalSBPOperators
using SummationByPartsOperators: MattssonNordström2004
source = MattssonNordström2004()
small = spherical_operators(source;
    accuracy_order = 4, N = 16, R = 16//1, p = 2, grid = :collocated)

small.r[1:7]
```

```@example closure
Matrix(small.Geven[1:7, 1:7])
```

```@example closure
Matrix(vector_mass(small)[1:7, 1:7])
```

The first row/column corresponds to (r=0). Only a small leading block of the
vector mass is non-diagonal. The exact SBP identity can be checked directly:

```@example closure
using LinearAlgebra: norm
residual = scalar_mass(small) * small.D + small.Geven' * vector_mass(small) - small.B
norm(Matrix(residual))
```

## Construct SBP6 with publication coefficients

```@example sbp6
using SphericalSBPOperators
using SummationByPartsOperators: MattssonNordström2004
source = MattssonNordström2004()
paper_sbp6 = spherical_operators(source;
    accuracy_order = 6, N = 30, R = 30//1, p = 2, grid = :collocated)

(s11 = paper_sbp6.S[1, 1], v23 = paper_sbp6.V[2, 3])
```

This parameter set follows the exact rational operator reported in the paper.
See `PAPER_OPERATOR_REPRODUCTION.md` for the full reproducibility map.

## Staggered operators

```@example staggered
using SphericalSBPOperators
using SummationByPartsOperators: MattssonNordström2004
source = MattssonNordström2004()
staggered = spherical_operators(source;
    accuracy_order = 4, N = 32, R = 1.0, p = 2, grid = :staggered)

(first_node = staggered.r[1], has_origin = has_origin_node(staggered),
 mass_type = typeof(scalar_mass(staggered)))
```

Here `N` is the number of staggered half-grid nodes, unlike collocated mode
where `N` is the number of subintervals.

## Numeric types and inference

Use a floating radius for standard numerical work and a rational radius for
exact coefficient inspection:

```@example types
using SphericalSBPOperators
using SummationByPartsOperators: MattssonNordström2004
source = MattssonNordström2004()
float_ops = spherical_operators(source;
    accuracy_order = 4, N = 20, R = 1.0, p = 2)
exact_ops = spherical_operators(source;
    accuracy_order = 4, N = 20, R = 1//1, p = 2)

(float_eltype = eltype(float_ops.D), exact_eltype = eltype(exact_ops.D))
```

The sparse operator matrices retain their numeric element type. In a test or
performance audit, inspect operator application with Julia’s inference tools:

```julia
using Test
x = ones(length(float_ops.r))
@inferred apply_even_gradient(float_ops, x)
@code_warntype apply_even_gradient(float_ops, x)
```

For a fixed operator and vector element type, the `apply_*` calls are
type-stable. Construction is normally performed once and reused.
