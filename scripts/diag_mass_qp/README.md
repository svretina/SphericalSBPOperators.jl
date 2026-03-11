# Diagonal-M QP Infeasibility Probe

This folder contains a standalone experiment for optimizing diagonal mass entries
`m_i` on `x = -R:1:R` while enforcing hard constraints and surfacing infeasibility
diagnostics when the system is inconsistent.

## Files

- `run_diag_mass_qp.jl`: CLI entrypoint.
- `diag_mass_qp_core.jl`: model construction, solve, verification, diagnostics.

## Constraint Set

- Diagonal positivity: `m_i >= m_lb` where default
  `m_lb = ∫_0^Δr r^2 dr = Δr^3/3` (next-cell volume from `r=0`, with `Δr=1` => `1/3`).
  Override with `--epsilon`.
- Symmetry: `m[c-k] = m[c+k]`.
- Optional anti-stiffness: `m[c] = m[c+1]` (disabled by default).
- Quadrature default is capped by `d/2` (`--quadrature-cap` to override):
  - default quadrature cap: `min(2d-p-1, d/2)`.
- Quadrature exactness through degree `0:quad_max_degree`
  (default `quad_max_degree = min(2d-p-1, d/2)`).
- Divergence constraints on odd monomials with boundary-lowered order:
  - Interior rows: odd `k <= d-p` (or CLI override).
  - Boundary rows: odd `k <= db-p` with `db` inferred from Cartesian boundary-row
    exactness (fallback `db=3` when `d=6` if inference fails).
  - Constraints are imposed via the SBP-translated linear form in `m`.

## Run

From repository root:

```bash
julia --project=. scripts/diag_mass_qp/run_diag_mass_qp.jl
```

Help:

```bash
julia --project=. scripts/diag_mass_qp/run_diag_mass_qp.jl --help
```

## Scenarios

1. Default run (expected infeasible or high violation):

```bash
julia --project=. scripts/diag_mass_qp/run_diag_mass_qp.jl
```

2. Debug relaxation run:

```bash
julia --project=. scripts/diag_mass_qp/run_diag_mass_qp.jl \
  --disable-anti \
  --quad-max-degree 2 \
  --div-max-odd 1 \
  --boundary-div-max-odd 1
```

With default lower bound `m_lb=1/3`, this case can still be near-infeasible
in some settings; to test a lighter lower bound:

```bash
julia --project=. scripts/diag_mass_qp/run_diag_mass_qp.jl \
  --disable-anti \
  --quad-max-degree 2 \
  --div-max-odd 1 \
  --boundary-div-max-odd 1 \
  --epsilon 1e-12
```

3. Anti-stiffness isolation run:

```bash
julia --project=. scripts/diag_mass_qp/run_diag_mass_qp.jl \
  --quad-max-degree 2 \
  --div-max-odd 1
```

When the solve is not feasible, the script prints:

- `rank(Aeq)` and `rank([Aeq|beq])`,
- singular-value conditioning,
- least-squares residual norms,
- per-family residual summaries (`symmetry`, `anti`, `quadrature`, `divergence`).

Note: Ipopt cannot directly optimize an overdetermined equality system, so the
script solves on an independent equality-row subset and always checks/report the
full hard system (`Aeq`, `beq`) afterward.
