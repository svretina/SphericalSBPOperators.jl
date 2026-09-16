# Experimental operator families

These implementations are retained for research reproducibility but are not the
operator constructions displayed in the accompanying paper.

- `diagonal_mass/`: diagonal-mass construction, validation, diagnostics, and
  legacy wave helpers.
- `diagonal_exp/`: wider near-origin diagonal-mass repair.
- `diagonal_mass_mixed_order/`: mixed-order diagonal-mass construction.
- `non_diagonal_mass/sbp6_experimental.jl`: alternate sixth-order split-mass closure.  It
  has five `r^5` origin constraints and two additional off-diagonal couplings;
  it does not reproduce the published SBP6 rationals.
- `non_diagonal_mass/sbp8.jl`: eighth-order search construction.

The shared Cartesian-grid, folding, and sparse-snapping utilities are in
`src/core/`, because the publication SBP4/SBP6 and staggered implementations
depend on them.

The publication SBP6 implementation is `src/non_diagonal_mass/sbp6.jl`. Its
exported `sbp6_exp_*` helper names are retained temporarily for compatibility;
with `outer_boundary_closure_help=false`, they reproduce the paper coefficients.
