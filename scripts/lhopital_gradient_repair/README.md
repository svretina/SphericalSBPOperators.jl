# L'Hopital Gradient Repair (Canonical `Δr=1`)

Standalone experiment for exact-rational (`Rational{BigInt}`) construction of a spherical folded SBP block with:

- canonical collocated grid spacing fixed to `Δr = 1`,
- L'Hopital divergence closure at the pole,
- user-selectable positive origin mass weight `w1` (or inferred from `j=2` raw ratio),
- boundary first-column gradient repair,
- coupled exact solve enforcing both:
  - gradient even-monomial constraints up to `accuracy_order`,
  - divergence odd-monomial constraints up to `accuracy_order - p` on repaired rows,
  - plus `q=1` divergence constraints on the solved near-origin row block `J_solve`,
- global divergence rederivation from exact SBP identity,
- optional physical scaling helper.

Raw stencil policy in `raw_pattern_expand_right` mode:
- start from a minimum width that covers the repaired rows and coupled constraints, then
- expand the fixed stencil band rightward until an exact coupled solution exists (no adaptive cascade row-updates during an attempt).

Repair-row selection follows `src/construct.jl`:
- repair rows are the rows with nonzero first-column leakage in raw folded `Geven`.

## Files

- `lhopital_gradient_repair_core.jl`: exact construction, repair, checks, scaling helper
- `run_lhopital_gradient_repair.jl`: CLI runner

## Canonical Build

User-facing input is the number of points on `[0,R]`, denoted `Nhalf` in the script
for backwards compatibility. If `Nhalf = n`, then the canonical internal index is
`r = 0:1:(n-1)` and the mirrored Cartesian grid is `x = -(n-1):1:(n-1)`.

Main constructor:

- `build_lhopital_repaired_canonical(cfg)`

with config:

- `accuracy_order::Int`
- `Nhalf::Int`
- `spatial_dim::Int` (default `3`, so `p=d-1`)
- `source` (default `MattssonNordström2004()`)
- `mode` (default package mode)
- `w1_mode::Symbol` in `(:user, :from_j2_raw)`
- `w1_user::Union{Nothing,Rational{BigInt}}`
- `stencil_mode::Symbol` in `(:raw_pattern_expand_right, :fixed_cols)`
- `custom_stencil_cols::Union{Nothing,Vector{Int}}`

## Physical Scaling Helper

- `scale_repaired_operators(canonical_report; R_phys)`

Uses `ρ = R_phys / (Nhalf-1)` and applies:

- `r -> ρ r`
- `G, D -> (1/ρ)`
- `M -> ρ^(p+1)`
- `B -> ρ^p`

Includes SBP residual check after scaling.

## CLI Usage

From repository root:

```bash
julia --project=. scripts/lhopital_gradient_repair/run_lhopital_gradient_repair.jl \
  --accuracy-order 6 --nhalf 41 --w1-from-j2
```

User `w1` mode:

```bash
julia --project=. scripts/lhopital_gradient_repair/run_lhopital_gradient_repair.jl \
  --accuracy-order 6 --nhalf 10 --w1-num 1 --w1-den 3
```

Fixed stencil mode:

```bash
julia --project=. scripts/lhopital_gradient_repair/run_lhopital_gradient_repair.jl \
  --accuracy-order 6 --nhalf 10 --w1-from-j2 \
  --stencil-mode fixed_cols --stencil-cols 2,3,4,5
```

With physical scaling output:

```bash
julia --project=. scripts/lhopital_gradient_repair/run_lhopital_gradient_repair.jl \
  --accuracy-order 6 --nhalf 10 --w1-from-j2 --R-phys 20
```

Help:

```bash
julia --project=. scripts/lhopital_gradient_repair/run_lhopital_gradient_repair.jl --help
```
