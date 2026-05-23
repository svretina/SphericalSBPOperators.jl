@inline diagonal_spherical_operators(args...; kwargs...) = spherical_operators(args...; kwargs...)

@inline staggered_spherical_operators(args...; kwargs...) = Staggered.spherical_operators(args...;
                                                                                          kwargs...)

@inline validate_staggered(args...; kwargs...) = Staggered.validate(args...; kwargs...)

@inline diagnose_staggered(args...; kwargs...) = Staggered.diagnose(args...; kwargs...)

@inline interpret_diagnostics_staggered(args...; kwargs...) = Staggered.interpret_diagnostics(args...;
                                                                                              kwargs...)

@inline non_diagonal_spherical_operators(args...; kwargs...) = NonDiagonalMass.spherical_operators(args...;
                                                                                                   kwargs...)

@inline non_diagonal_exp_spherical_operators(args...; kwargs...) = NonDiagonalMass.sbp6_exp_spherical_operators(args...;
                                                                                                                 kwargs...)

@inline mixed_order_diagonal_spherical_operators(args...; kwargs...) = MixedOrderDiagonalMass.spherical_operators(args...;
                                                                                                                  kwargs...)

@inline diagonal_exp_spherical_operators(args...; kwargs...) = DiagonalExp.spherical_operators(args...;
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
