# MixedOrderDiagonalMass

This folder implements a diagonal-mass spherical SBP variant that follows the
same construction path as `src/diagonal_mass/`, with one deliberate change:

- `Geven` is built from a Cartesian SBP first-derivative operator of requested
  accuracy order `s`;
- `Godd` is built from a Cartesian SBP first-derivative operator of order
  `s + 2`.

The motivation is to keep the standard diagonal-mass SBP structure for the
even-field gradient operator while upgrading the odd-derivative information used
in the removable-singularity (`l'Hospital`) origin row for the divergence.

The construction still aims to satisfy the spherical SBP relation

`S * D + transpose(Geven) * V = B`

in exactly the same sense as the standard diagonal-mass operators:

- `S`, `V`, and `B` are built from the `s`-order Cartesian mass path;
- `Geven` is folded from the `s`-order Cartesian derivative and then repaired
  with the same coupled near-origin row procedure as the standard
  diagonal-mass construction;
- `D[i, :]` for `i >= 2` is derived from the SBP relation using that repaired
  `Geven`;
- the free origin row `D[1, :]` is set from the higher-order odd derivative via
  `D[1, :] = (p + 1) * Godd[1, :]`.

Because `S[1,1] = 0` for `p > 0`, replacing the origin row in this way does not
break the spherical SBP identity away from the origin row, and it preserves the
same API style as the existing operator families.
