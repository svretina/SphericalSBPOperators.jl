# Published non-diagonal operators: reproduction and cleanup inventory

This note traces the rational coefficients displayed in
`/home/svretina/PhD/mypapers/Spherical-SBP-Operators-Paper/main.tex` to their
implementation paths.  It deliberately distinguishes the published operators
from later experimental operator families in this repository.

## Published parameter sets

The displayed rational matrices in the paper are the collocated,
**non-diagonal-mass** spherical operators based on
`SummationByPartsOperators.MattssonNordstrom2004()`.

| Paper subsection | Order | `p` | `R` | `h` | Nodes on `[0,R]` | Published origin accuracy constraints |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4th-order non-diagonal norm | 4 | 2 | 30 | 1 | 31 | `D*r = 3` through row `N-4`; `D*r^3 = 5r^2` on rows 1:5; exact volume |
| 6th-order non-diagonal norm | 6 | 2 | 30 | 1 | 31 | `D*r = 3` everywhere; `D*r^3 = 5r^2` except the right closure; `D*r^5 = 7r^4` on rows 1:3; exact volume |

`N` in the public unified API means *subintervals*, so use `N = 30`; the
result has `N + 1 = 31` nodes.  Use `R = 30//1` and
`target_eltype = Rational{BigInt}` to retain exact rational output.

## Component inventory

Keep these as the minimal construction path for the paper's coefficients.

| Role | File | Required names |
| --- | --- | --- |
| Public package surface and module wiring | `src/SphericalSBPOperators.jl` | `non_diagonal_spherical_operators`, `sbp4_operators`, `sbp6_operators`; includes for `sbp4.jl`, `sbp6.jl`, and `NonDiagonalMass.jl` |
| Unified physical-grid wrapper and scaling | `src/non_diagonal_mass/construct.jl` | `NonDiagonalMass.spherical_operators`, `_resolve_non_diagonal_points`, `_resolve_non_diagonal_grid`, `_build_non_diagonal_ops`, `scale_spherical_operators` |
| Fourth-order published construction | `src/non_diagonal_mass/sbp4.jl` | `sbp4_scalar_mass_gradient`, `sbp4_v_offdiag_pairs`, `sbp4_vector_mass`, `sbp4_solve_accuracy_constraints`, `sbp4_construct_divergence`, `sbp4_operators` |
| Sixth-order published construction | `src/non_diagonal_mass/sbp6.jl` | `sbp6_v_offdiag_pairs`, `sbp6_vector_mass`, `sbp6_solve_accuracy_constraints`, `sbp6_construct_divergence`, `sbp6_operators` |
| Cartesian source and exact linear algebra dependency | `Project.toml` | `SummationByPartsOperators` (with `MattssonNordström2004`, `SafeMode`) |
| Regression-test location | `test/runtests.jl` | `Non-Diagonal Unified API`; `Experimental SBP6 without outer boundary closure help`; add the missing SBP4 coefficient assertions |

`src/experimental/diagonal_mass/**`, `src/staggered/**`,
`src/experimental/diagonal_exp/**`,
`src/experimental/diagonal_mass_mixed_order/**`,
`src/experimental/non_diagonal_mass/sbp6_experimental.jl`, and
`src/experimental/non_diagonal_mass/sbp8.jl` are not needed to construct the rational
matrices printed in the paper.

## Reproduction commands

The fourth-order path is the current public construction path:

```julia
using SphericalSBPOperators
using SummationByPartsOperators: MattssonNordström2004, SafeMode

source = MattssonNordström2004()
ops4 = non_diagonal_spherical_operators(source;
    accuracy_order = 4, N = 30, R = 30//1, p = 2,
    mode = SafeMode(), target_eltype = Rational{BigInt})
```

The published rational entries to assert include
`ops4.S[1,1] == 3714185//6311896`,
`ops4.V[2,3] == 8002//46411`, and
`ops4.V[4,5] == -1132080//5522909`.

The sixth-order paper route is the normal non-diagonal constructor; the critical
setting is `outer_boundary_closure_help = false` (the default):

```julia
ops6 = non_diagonal_spherical_operators(source;
    accuracy_order = 6, N = 30, R = 30//1, p = 2, mode = SafeMode(),
    target_eltype = Rational{BigInt})
```

This implementation pins precisely the paper's SBP6 structure:

- `D*r^5 = 7r^4` on rows `1:3` (`first_rows_r5 = 3`);
- exactly seven origin off-diagonal pairs: `(2,3)`, `(2,4)`, `(3,4)`, `(3,5)`,
  `(4,5)`, `(5,6)`, `(6,7)`;
- no additional right-boundary entries (`outer_boundary_closure_help = false`).

`test/runtests.jl` has an exact regression test for this parameter set.
It asserts the paper's displayed scalar-norm and vector-norm rationals,
including `V[2,3] == 4996740529431//6413875155127` and
`V[6,7] == -70992217935//12827750310254`.

The similarly named standard `src/experimental/non_diagonal_mass/sbp6_experimental.jl` is **not** the
paper operator: it defaults to five `r^5` rows and includes two additional
off-diagonal pairs, `(4,6)` and `(5,7)`.  It belongs in the experimental or
archive track unless it has another supported use case.

## Cleanup recommendation

Make publication reproduction a small, tested core before moving code:

1. Add a `test/paper_operator_coefficients.jl` regression test for both orders
   at the exact parameter sets above.  Assert every displayed near-origin
   `S`/`V` coefficient and exact `S*D + G' * V == B`.
2. Promote the current SBP6 paper path to an explicit constructor, e.g.
   `paper_non_diagonal_spherical_operators(order=6)`, or rename
   historical `sbp6_exp_*` helper names to `sbp6_*`, retaining compatibility
   aliases temporarily.
3. Keep the modules in the inventory table plus a minimal construction script
   (adapt `scripts/construct_operators.jl` to one `:paper_non_diagonal`
   configuration).
4. Move candidates only after the regression passes: the standard alternate
   `sbp6`, SBP8 search/archive material, diagonal experiments, alternate
   mass families, plotting pipelines, spectrum-table scripts, and wave
   simulation machinery can all live under `archive/` or a separate research
   repository.  Preserve each move with `git mv` and a short archive README
   stating its status and last known command.

This sequence keeps a reviewable, publication-faithful core and prevents later
prototype defaults from changing the operator advertised by the paper.
