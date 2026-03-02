# Staggered SBP Operator Report (6th Order)

Generated: 2026-02-28 19:57:25

## Configuration
- branch: feature/staggered-spherical-sbp
- source: MattssonNordström2004
- accuracy_order: 6
- N: 64
- R: 1.0
- p: 2
- build_matrix: `matrix_if_square`

## Grid and Mass
- Nh: 64
- M_full: 128
- r_min: 0.007874015748
- r_max: 1.0
- has origin node: false
- min(H_ii): 9.763799055e-7
- max(H_ii): 0.02121175457
- all H_ii > 0: true

## SBP and Closure
- closure_width_left: 3
- closure_width_right: 6
- safe_count: 55
- sbp_full: 4.440892099e-16
- sbp_no_origin (same as full for staggered): 4.440892099e-16

## Gradient Errors (Even Monomials)
| degree | max_error | max_error_safe | max_error_near_boundary |
|---:|---:|---:|---:|
| 0 | 2.8421709430000003e-14 | 1.554312234e-15 | 2.8421709430000003e-14 |
| 2 | 4.2632564150000004e-14 | 1.065814104e-14 | 4.2632564150000004e-14 |
| 4 | 6.338437931e-5 | 7.549516567e-15 | 6.338437931e-5 |
| 6 | 0.0008732072308 | 8.437694987e-15 | 0.0008732072308 |

## Divergence Errors (Odd Monomials)
| degree | max_error | max_error_safe | max_error_near_boundary |
|---:|---:|---:|---:|
| 1 | 1.5099033130000002e-14 | 1.110223025e-14 | 1.5099033130000002e-14 |
| 3 | 0.0003237735502 | 1.5543122340000002e-14 | 0.0003237735502 |

## Quadrature Errors
| monomial degree k | abs_error for ∫ r^(k+p) dr |
|---:|---:|
| 0 | 5.5511151229999994e-17 |
| 1 | 4.484685123e-10 |
| 2 | 2.775557562e-17 |
| 3 | 7.123190926e-13 |
| 4 | 5.733774566e-12 |
| 5 | 2.595951232e-11 |
| 6 | 8.380554584e-11 |

## Diagnose Summary
- grid_ok: true
- gradient global safe max: 1.065814104e-14
- divergence global safe max: 1.5543122340000002e-14
- sbp max_no_origin: 4.440892099e-16

### Interpretation
- No structural inconsistency detected; staggered operators satisfy expected SBP and interior-accuracy checks.
