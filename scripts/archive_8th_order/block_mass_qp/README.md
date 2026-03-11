# Block-Mass QP Sweep

This experiment keeps the mass-matrix diagonal close to `|r|^p` and adds a
near-origin symmetric off-diagonal block with increasing radius to add DOFs.

For each radius, it solves:

- objective: diagonal fit to `|r|^p` plus off-diagonal regularization,
- constraints: reflection symmetry, quadrature, and SBP-translated divergence,
- then reports rank checks and full residuals.

The sweep stops at the first radius whose full residual satisfies `solve_tol`
(or continues with `--no-stop`).

## Run

```bash
julia --project=. scripts/block_mass_qp/run_block_mass_qp.jl
```

Example (61 points, `d=4`, `p=2`, divergence order fixed to 1):

```bash
julia --project=. scripts/block_mass_qp/run_block_mass_qp.jl \
  --R 30 \
  --d 4 \
  --p 2 \
  --div-max-odd 1 \
  --boundary-div-max-odd 1 \
  --quad-max-degree 3 \
  --block-radius-start 0 \
  --block-radius-max 8
```
