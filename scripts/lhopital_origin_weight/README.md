# L'Hopital Origin-Weight Probe

This experiment analyzes a raw (unrepaired) folded spherical SBP construction with:

- even scalar parity,
- odd vector parity,
- origin closure imposed by a discrete L'Hopital rule,
- diagonal split masses.

The script computes a candidate origin mass weight from

`w1 = -(G[j,1] * M[j,j]) / D[1,j]`

and checks whether a single positive constant `w1` is consistent across `j`.

## Run

From repository root:

```bash
julia --project=. scripts/lhopital_origin_weight/run_lhopital_origin_weight.jl
```

Help:

```bash
julia --project=. scripts/lhopital_origin_weight/run_lhopital_origin_weight.jl --help
```

## Notes

- The experiment intentionally does **not** apply gradient first-column repair.
- For spherical 3D runs use `--p 2` (default), so the origin row closure is `D[1,:] = 3 * Godd[1,:]`.
