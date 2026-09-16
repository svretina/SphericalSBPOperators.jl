"""
    spherical_operators(source; accuracy_order, N, R, p=2, grid=:collocated, kwargs...)

Construct a spherical SBP operator set on `[0, R]` from a
`SummationByPartsOperators` source.

`grid` selects the publication-supported discretization:

- `:collocated` (default): the non-diagonal-mass SBP4/SBP6 operators with a
  node at the origin;
- `:staggered`: the staggered SBP operators used for the paper comparison.

`accuracy_order`, `N`, and `R` are required. In collocated mode, `N` denotes
subintervals, so the grid has `N + 1` nodes. In staggered mode, it is forwarded
as the staggered half-grid node count. Extra keywords are forwarded to the
selected constructor, e.g. `method=:naive` for the staggered comparison operator.
"""
function spherical_operators(source;
                             accuracy_order::Integer,
                             N::Integer,
                             R,
                             p::Integer = 2,
                             grid::Symbol = :collocated,
                             kwargs...)
    if grid === :collocated
        return non_diagonal_spherical_operators(source;
                                                 accuracy_order = Int(accuracy_order),
                                                 N = Int(N),
                                                 R = R,
                                                 p = Int(p),
                                                 kwargs...)
    elseif grid === :staggered
        return staggered_spherical_operators(source;
                                              accuracy_order = Int(accuracy_order),
                                              N = Int(N),
                                              R = R,
                                              p = Int(p),
                                              kwargs...)
    end

    throw(ArgumentError("`grid` must be :collocated or :staggered; got `$grid`."))
end

@inline staggered_spherical_operators(args...; kwargs...) = Staggered.spherical_operators(args...;
                                                                                          kwargs...)

@inline validate_staggered(args...; kwargs...) = Staggered.validate(args...; kwargs...)

@inline diagnose_staggered(args...; kwargs...) = Staggered.diagnose(args...; kwargs...)

@inline interpret_diagnostics_staggered(args...; kwargs...) = Staggered.interpret_diagnostics(args...;
                                                                                              kwargs...)

@inline non_diagonal_spherical_operators(args...; kwargs...) = NonDiagonalMass.spherical_operators(args...;
                                                                                                   kwargs...)

@inline apply_even_gradient(ops::Staggered.SphericalOperators, phi) = Staggered.apply_even_gradient(ops,
                                                                                                    phi)

@inline apply_odd_derivative(ops::Staggered.SphericalOperators, u) = Staggered.apply_odd_derivative(ops,
                                                                                                    u)

@inline apply_divergence(ops::Staggered.SphericalOperators, u) = Staggered.apply_divergence(ops,
                                                                                            u)

@inline apply_even_gradient(ops::NonDiagonalMass.SphericalOperators, phi) = ops.Geven * phi
@inline apply_odd_derivative(ops::NonDiagonalMass.SphericalOperators, u) = ops.Godd * u
@inline apply_divergence(ops::NonDiagonalMass.SphericalOperators, u) = ops.D * u
