# Collocated vs Staggered SBP Comparison (6th Order, Rational{BigInt})

Generated: 2026-02-28 20:39:04

## Requested Setup
- accuracy order: 6
- arithmetic: `Rational{BigInt}` (SafeMode)
- target spacing: `Δr = 1`
- target points: `81` half-grid points in each operator set

## Grid Definitions
- collocated: `r = 0,1,...,80` (`Nh=81`, `N=80`, `R=80`)
- staggered: `r = 1/2,3/2,...,161/2` (`Nh=81`, `N=81`, `R=161/2`)

### Spacing/size checks
- collocated Nh: 81, min Δr: 1.0, max Δr: 1.0
- staggered Nh: 81, min Δr: 1.0, max Δr: 1.0

## SBP Relation Error (max abs)
Residual: `R = H*D + Geven' * H - B`

| set | full | first row | safe interior rows | boundary closure rows |
|---|---:|---:|---:|---:|
| collocated | 0.0 | 0.0 (`r=0`) | 0.0 | 0.0 |
| staggered | 0.0 | 0.0 (`r=h/2`) | 0.0 | 0.0 |

## Accuracy-Condition Errors by Moment
Region definitions:
- first row: collocated at `r=0`, staggered at `r=h/2`
- interior: max error over `idx_safe`
- boundary closure: max error over right closure rows

### Even-gradient moments: `phi=r^k`, `Geven*phi` vs `d(phi)/dr`
| k | collocated first row | collocated interior | collocated boundary | staggered first row | staggered interior | staggered boundary |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 4 | 0.0 | 0.0 | 16.22943563 | 0.0 | 0.0 | 16.22943563 |
| 6 | 0.0 | 0.0 | 1.45666738e6 | 0.0 | 0.0 | 1.475558615e6 |

### Odd-divergence moments: `u=r^k`, `D*u` vs `(p+k)r^(k-1)` (p=2)
| k | collocated first row | collocated interior | collocated boundary | staggered first row | staggered interior | staggered boundary |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3 | 0.0 | 0.0 | 1.031688295 | 0.0 | 0.0 | 1.025171841 |

## Quadrature Errors by Moment
For each `k`, compare numerical `dot(r^k, H*1)` with exact `∫_0^R r^(k+p)dr`.

| k | collocated abs error | staggered abs error |
|---:|---:|---:|
| 0 | 0.0 | 0.0 |
| 1 | 0.008333333333 | 0.007291666667 |
| 2 | 0.0 | 0.0 |
| 3 | 0.03888888889 | 0.04670138889 |
| 4 | 28.11309524 | 28.24166667 |
| 5 | 9864.619444 | 9963.231977 |
| 6 | 2.505417028e6 | 2.545072581e6 |

## Additional Notes
- First-row diagnostics are reported for both operators.
- This corresponds to `r=0` for collocated and `r=h/2` for staggered.
- Both constructions are exact-arithmetic (`Rational{BigInt}`), with reported values converted to Float64 for readability.
