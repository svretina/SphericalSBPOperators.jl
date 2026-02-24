# SphericalSBPOperators

`SphericalSBPOperators.jl` builds spherical-symmetry SBP operators on `[0, R]` by
folding Cartesian SBP operators from a mirrored grid `[-R, R]`.

## Theory summary

At the origin, parity is a symmetry constraint, not a boundary condition:
- scalar-like fields `ϕ` are even (`ϕ(-r) = ϕ(r)`);
- radial flux-like fields `u` are odd (`u(-r) = -u(r)`, so `u(0)=0`).

The operators enforce:
- `Geven`: even -> odd derivative;
- `D`: odd -> even spherical divergence.

For metric power `p` (`p=2` spherical, `p=1` cylindrical):
```math
\mathrm{Div}_p(u)=\frac{1}{r^p}\frac{d}{dr}\left(r^p u\right).
```

The metric-weighted mass uses:
```math
H = H_{\mathrm{cart,half}}\,\mathrm{diag}(r^p), \quad
H_{\mathrm{cart,half}} = \frac12 E_{\mathrm{even}}^T H_{\mathrm{full}} E_{\mathrm{even}}.
```

The discrete SBP identity is:
```math
H D + G^T H = B, \quad B=\mathrm{diag}(0,\dots,0,R^p).
```

`B[end,end]=R^p` because only the outer boundary contributes; `r=0` is not a boundary.

For `p>0`, `H[1,1]=0`, so SBP does not determine the origin row of `D`.
The implementation fixes that row with the removable-singularity limit:
```math
D[1,:] = (p+1)\,G_{\mathrm{odd}}[1,:].
```

## Installation

```julia
using Pkg
Pkg.activate("path/to/SphericalSBPOperators.jl")
Pkg.instantiate()
```

## Usage

```julia
using SphericalSBPOperators
using SummationByPartsOperators: MattssonNordström2004

source = MattssonNordström2004()
ops = spherical_operators(source;
    accuracy_order = 4,
    N = 64,
    R = 1.0,
    p = 2,
    build_matrix = :probe,
)

report = validate(ops; verbose = true)
diag = diagnose(ops, report; Ktest = 8, verbose = true)
```

## Exact rational mode

Numeric type is inferred from inputs similarly to `SummationByPartsOperators`.
If `R` is rational, operators are built with rational arithmetic.

```julia
ops_exact = spherical_operators(source;
    accuracy_order = 4,
    N = 16,
    R = 1//1,
    p = 2,
)
```

In rational mode, exact identities (such as `sbp_no_origin`) can be exactly zero.

## Diagnostics

`diagnose(ops, report)` runs grid/folding/operator-consistency checks and localized
polynomial error analysis, then returns a structured `NamedTuple` plus an interpreted
conclusion list.

## Wave SAT boundary conditions

The radial first-order wave system uses:

```math
\partial_t \Pi = D\Xi,\qquad \partial_t \Xi = G_{\mathrm{even}}\Pi
```

with discrete energy

```math
E = \tfrac12\left(\Pi^T H \Pi + \Xi^T H \Xi\right).
```

Using `H D + G^T H = B`, interior terms collapse to an outer-boundary flux:

```math
\frac{dE}{dt} = \Pi^T B \Xi = B_{NN}\,\Pi_N\Xi_N.
```

At `r=R`, characteristic variables are

```math
w_{\mathrm{in}} = \Pi + \Xi,\qquad w_{\mathrm{out}} = \Pi - \Xi.
```

SAT penalties are added only at node `N`:

```math
\dot\Pi_N \mathrel{+}= -\sigma_\Pi H_{NN}^{-1}\rho,\qquad
\dot\Xi_N \mathrel{+}= -\sigma_\Xi H_{NN}^{-1}\rho.
```

- `bc=:absorbing`: `\rho=w_{\mathrm{in}}`, `\sigma_\Pi=B_{NN}/2`, `\sigma_\Xi=B_{NN}/2` (dissipative).
- `bc=:reflecting`: `\rho=w_{\mathrm{in}}-w_{\mathrm{out}}=2\Xi` (equivalent to `\Xi(R)=0`), `\sigma_\Pi=B_{NN}/2`, `\sigma_\Xi=0` (energy-conserving).

Origin symmetry is enforced separately:
- state constraint: `\Xi(0)=0`,
- RHS constraint: `\dot\Xi(0)=0`.

This is required because for `p>0`, `H[1,1]=0`, so the origin DOF is invisible to the energy norm unless parity is explicitly enforced.

## Initial Data Notes

Coupling is immediate in the first-order system:

```math
\Xi_t = G_{\mathrm{even}}\Pi.
```

So even when `\Xi_0 = 0`, any spatial variation in `\Pi_0` gives nonzero `\Xi_t(0)`,
and `\Xi` appears right away. This is expected and not itself an energy bug.

`solve_wave_ode` supports:
- `initial_data_mode=:auto`: default behavior (`\Xi_0 = G\phi_0` if `\Xi_0` is omitted),
- `initial_data_mode=:potential`: require `\phi_0` and build `\Xi_0 = G\phi_0`,
- characteristic profiles via `w_in0`, `w_out0` with
  `\Pi_0 = (w_{\mathrm{in},0}+w_{\mathrm{out},0})/2`,
  `\Xi_0 = (w_{\mathrm{in},0}-w_{\mathrm{out},0})/2`.

Use `check_potential_consistency(ops, \Pi, \Xi)` to reconstruct a best-fit `\hat\phi`
from `\Xi \approx G\hat\phi`, report residuals, and report `max|G\Pi|` as the expected
instantaneous source of `\Xi` growth.
