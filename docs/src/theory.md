# Theory

## Radial divergence and regularity

For a radial flux (u), the metric-weighted divergence is

```math
\mathcal D_p u = \frac{1}{r^p}\frac{d}{dr}(r^p u)
                = \frac{d u}{dr} + \frac{p}{r}u.
```

The parameter (p) describes the radial volume element. In particular,
(p=2) is the standard three-dimensional spherical case and (p=1) is the
cylindrical case. The singular-looking term is regular on physically smooth
fields: scalar variables are even under reflection at the origin, while radial
flux variables are odd and vanish there.

For example, (u(r)=r^3) is odd and

```math
\mathcal D_2(r^3) = 5r^2.
```

## Summation by parts

The continuous weighted integration-by-parts identity motivates matrices
(G), (D), (S), (V), and (B) satisfying

```math
S D + G^T V = B,
\qquad B = \operatorname{diag}(0,\ldots,0,R^p).
```

`G` maps even scalar data to odd derivative data; `D` maps odd flux data to
even divergence data. `S` and `V` define the scalar and vector discrete energy
inner products. This identity transfers the continuum energy argument to the
semi-discrete problem: only the outer boundary contributes to the energy rate.

## Collocated construction

The collocated construction begins with a Cartesian SBP derivative on
([-R,R]). Even and odd extension/folding maps restrict it to ([0,R]),
thereby enforcing regularity by symmetry rather than treating the origin as a
physical boundary.

The supported SBP4 and SBP6 families use a diagonal scalar mass `S` and a
symmetric vector mass `V` containing a small off-diagonal block near the
origin. The divergence is then obtained from the SBP identity,

```math
D = S^{-1}(B-G^T V).
```

The closure coefficients are solved with exact rational arithmetic. In addition
to the SBP relation, the construction imposes exact radial-divergence moments
and the correct weighted volume quadrature. The SBP6 default is the
publication’s left-origin closure.

## Staggered construction

Staggered grids use positive half-grid points and therefore contain no origin
node. They are useful for comparisons and for applications whose variables are
already naturally staggered. `grid = :staggered` selects the compatible SBP
construction. The optional `method = :naive` is provided for comparison only;
it is not an SBP discretization and should not be used when an energy estimate
is required.

## Accuracy

The package accepts formal accuracy orders 4 and 6 for both supported public
families. Since (mathcal D_p) differentiates (r^p u), polynomial exactness
of the divergence differs from that of a Cartesian first derivative. Formal
convergence order nevertheless remains the requested order for smooth,
regular fields.
