# API reference

## Construction

```@docs
spherical_operators
```

`source` is a Cartesian SBP source from `SummationByPartsOperators.jl`.
`accuracy_order`, `N`, and `R` are required. `grid` is either `:collocated`
(default) or `:staggered`.

For collocated operators, `N` is the number of subintervals, hence the grid has
`N + 1` nodes. For staggered operators, `N` is the number of positive half-grid
nodes.

## Operator data

All operator sets expose `r`, `B`, `Geven`, `Godd`, `D`, `accuracy_order`, `p`,
and `R`. The generic accessors provide the correct masses for either family.

```@docs
AbstractSphericalOperators
NonDiagonalMassSphericalOperators
StaggeredSphericalOperators
scalar_mass
vector_mass
has_origin_node
```

## Application

```@docs
apply_even_gradient
apply_odd_derivative
apply_divergence
```

`apply_even_gradient` expects an even scalar field. `apply_odd_derivative` and
`apply_divergence` expect an odd radial-flux field; for collocated grids, its
first entry must be zero.

## Staggered validation

```@docs
validate_staggered
diagnose_staggered
interpret_diagnostics_staggered
```

These are available for the supported staggered construction. Exact
coefficient-level validation for collocated SBP4/SBP6 is maintained in the test
suite and the paper reproduction note.
