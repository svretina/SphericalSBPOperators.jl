# Porting the exact SBP2 spherical operator

This note specifies the second-order collocated spherical SBP construction in
`src/non_diagonal_mass/sbp2.jl`. It is intended to be sufficient for an agent
to implement the same operator family in another library without copying this
repository's API or data structures.

The result is a first-derivative/weighted-divergence pair on `r ∈ [0, R]` for
the radial weight `r^p`, normally `p = 2`:

```math
S D + G^T V = B,
\qquad B = \operatorname{diag}(0, \ldots, 0, R^p).
```

At second order, both norms can be diagonal. The useful solution is stronger:
they can be chosen to be the **same positive diagonal norm**, `S = V`.

## Inputs and conventions

Use a Cartesian second-order diagonal-norm SBP first derivative on the
mirrored interval `[-R, R]`, with a node at zero. Construct it with exact
rational coefficients when possible. Let its derivative be `Gfull` and let
the full grid be symmetric about zero.

Fold this Cartesian operator onto the non-negative half grid:

1. Let `r = [0, h, 2h, ..., R]` have `m = N + 1` nodes.
2. Form the usual even extension `Eeven`, odd extension `Eodd`, and half-grid
   restriction `Rop`.
3. Define

   ```math
   G    = Rop\,G_{full}\,E_{even},
   \qquad
   G_{odd} = Rop\,G_{full}\,E_{odd}.
   ```

`G` maps an even scalar field to its odd radial derivative. `Godd` is kept for
the origin-regularity check below. Index `0` in this document is the origin;
translate it to index `1` in one-based languages.

The construction requires the folded derivative to annihilate constants:
`G * 1 = 0`. If the source library creates tiny floating-point entries, remove
only roundoff-scale entries before converting to rationals; do not solve this
closure in floating point.

## Exact diagonal closure solve

An ansatz with independent diagonal vectors `s` and `v` is underdetermined by
the basic divergence constraint. The common-norm choice,

```math
S = V = \operatorname{diag}(s),
```

removes that ambiguity and produces a unique, positive rational solution.

Require exact divergence of the regular linear radial flux:

```math
D r = (p + 1)\mathbf{1}.
```

Substituting `D = S^{-1}(B - G^T V)` and the common diagonal norm gives the
linear system

```math
\left((p + 1)I + G^T\operatorname{diag}(r)\right)s = B r.
```

In component form, for half-grid indices `i,j = 0, ..., m-1`, assemble

```text
A[i,j] = (p + 1) * delta(i,j) + G[j,i] * r[j]
b[i]   = 0
b[m-1] = R^(p+1)
```

because `(B*r)[m-1] = R^p * R`. Append the exact volume-quadrature row

```text
A[m,:] = 1
b[m]   = R^(p+1) / (p+1)
```

and solve the overdetermined system exactly over `Rational{BigInt}` (or the
other library's unbounded rational type). The last row is redundant when
`G*1 = 0`, but retaining it catches construction or indexing errors.

Use an exact RREF/Gaussian-elimination solver that accepts a consistent
overdetermined system. For the second-order folded stencil this returns zero
residual and positive `s[i]`. Reject the candidate if either property fails.

## Constructing the operators

With the solved vector `s`, form

```text
S = diagonal(s)
V = diagonal(s)                 # deliberately equal to S
B = zero(m,m); B[m-1,m-1] = R^p
D = diagonal(1 ./ s) * (B - transpose(G) * V)
```

Build all of these quantities in exact arithmetic. Do not start from the
standard metric-lumped mass `Hhalf * diagonal(r^p)` and only modify a small
boundary block: its interior entries do not make `D*r` exact for the
second-order centered stencil. The exact solve determines the whole common
diagonal norm.

The origin entry of `S` is positive. This is intentional: unlike a direct
metric-lumped `r^p` mass, it makes `S` invertible and lets the SBP identity
determine the origin divergence row.

## Reference pseudocode

```text
function build_sbp2_spherical(Gfull, xfull, p, R):
    r, Rop, Eeven, Eodd = folding_maps_with_origin(xfull)
    G    = exact_sparse(Rop * Gfull * Eeven)
    Godd = exact_sparse(Rop * Gfull * Eodd)
    m    = length(r)

    A = zero_rational_matrix(m + 1, m)
    b = zero_rational_vector(m + 1)
    for i in 0:m-1:
        for j in 0:m-1:
            A[i,j] = ((p + 1) if i == j else 0) + G[j,i] * r[j]
    b[m-1] = R^(p + 1)
    A[m,:] = 1
    b[m]   = R^(p + 1) / (p + 1)

    s = solve_exact_consistent_system(A, b)
    assert A*s == b
    assert every_entry_is_positive(s)

    S = diagonal(s)
    V = diagonal(s)
    B = zero_rational_matrix(m, m)
    B[m-1,m-1] = R^p
    D = diagonal(1/s) * (B - transpose(G) * V)
    return r, G, Godd, D, S, V, B
```

## Required checks

Run these checks as exact equalities for a rational build, at least for
`p = 2` and a comfortably resolved grid such as `N = 20`:

```text
S == V
S and V are diagonal
all(diagonal(S) > 0)
S*D + transpose(G)*V == B
D*r == (p+1)*ones(m)
sum(diagonal(S)) == R^(p+1)/(p+1)
D[origin,:] == (p+1)*Godd[origin,:]
```

Also verify numerical refinement on a physical interval, for example `R = 1`.
For `p = 2`, use the even scalar `phi(r) = cos(r)` and regular odd flux
`u(r) = r exp(-r^2)`. Away from the outer closure, check second-order
convergence of

```math
G\phi + \sin(r),
\qquad
Du - (3 - 2r^2)\exp(-r^2).
```

## Scaling from a canonical grid

For clean rational arithmetic, it is convenient to build on the canonical
grid with `h = 1` and `R_can = N`, then scale to a requested physical radius.
With `lambda = R_target / R_can`, apply

```text
r, H       *= lambda
S, V       *= lambda^(p+1)
B          *= lambda^p
G, Godd, D *= lambda^(-1)
```

The SBP identity and all polynomial identities are preserved by this scaling.
If the other library instead constructs directly at rational `R`, the scaling
step is unnecessary.

## Integration points in this repository

The portable implementation is [src/non_diagonal_mass/sbp2.jl](src/non_diagonal_mass/sbp2.jl).
The dispatch addition is in [src/non_diagonal_mass/construct.jl](src/non_diagonal_mass/construct.jl),
and executable acceptance tests are in [test/runtests.jl](test/runtests.jl).
