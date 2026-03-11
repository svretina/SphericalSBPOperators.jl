# SBP8 Resume Checkpoint (2026-03-05)

## Project Path
- `/home/svretina/Codes/SphericalSBPOperators.jl`

## User Goal
- Find a `V` structure (with diagonal `S`) for the SBP8 spherical operator such that:
- accuracy constraints are exact (rational solve),
- `L = DIV * G` has purely real spectrum,
- and all eigenvalues are non-positive (no positive real part).

## Non-Negotiable Constraints Used
- Source: `Mattsson2017(:central)`.
- Folded size: `31 x 31` (via `points=31`).
- Keep `s1 = 11/20` fixed.
- Keep `v1 = 1` fixed.
- Keep tail scaling:
- `S[24:31] = (S_cart * r^2)[24:31]`.
- `Vdiag[24] = (S_cart * r^2)[24]`.
- Accuracy rows:
- `(DIV.r - 3) == 0` on all rows.
- `(DIV.r^3 - 5 r^2) == 0` on rows `1:23`.
- `(DIV.r^5 - 7 r^4) == 0` on rows `1:23`.
- `(DIV.r^7 - 9 r^6) == 0` on rows `1:fr7`.
- Quadrature exact.

## Core Architecture Used in Latest Scans
- `V` off-diagonal pairs from `sbp8_v_offdiag_pairs(N; boundary_rows=br, boundary_bandwidth=bw)`.
- `diag_free_indices = 1:24`.
- Base zero pairs enforced in all structural scans:
- `(1,2), (1,3), (1,4), (13,16), (14,16), (15,16)`.

## Current Best Results
- Best `max_real` with `max_imag=0` found so far:
- `br=19`, `bw=7`, `fr7=10`, extra zeros `(14,17),(15,17),(16,17)`.
- `rho(L)=34.36856247554086`
- `max_real(L)=6.105970805947682e-8`
- `max_abs_imag(L)=0.0`
- Exact errors all `0.0`.

- Lower `rho` but worse positive part:
- `br=19`, `bw=7`, `fr7=12`, extra zeros `(14,17),(15,17),(16,17)`.
- `rho(L)=28.37197120395912`
- `max_real(L)=6.318434796250822e-7`
- `max_abs_imag(L)=0.0`

- Closest baseline architecture:
- `br=17`, `bw=7`, `fr7=11`, base-only.
- `rho(L)=34.760771262222086`
- `max_real(L)=1.621658227501359e-7`
- `max_abs_imag(L)=0.0`

## Saved Reproducible Checkpoint Runs
- Directory:
- `/home/svretina/Codes/SphericalSBPOperators.jl/data/sbp8_m17_checkpoint_runs`
- Main index:
- `/home/svretina/Codes/SphericalSBPOperators.jl/data/sbp8_m17_checkpoint_runs/README.md`
- Reproduce all 3 runs:
- `julia --project=. data/sbp8_m17_checkpoint_runs/reproduce_runs.jl`

## Important Scan Files
- Main fixed-`s1` structural scan:
- `/home/svretina/Codes/SphericalSBPOperators.jl/data/sbp8_m17_fixed_s1_struct_scan.tsv`
- Expanded `br=20..24` structural scan:
- `/home/svretina/Codes/SphericalSBPOperators.jl/data/sbp8_struct_expand_scan.tsv`
- Run-B free-DOF 2D scan (`a40`, `a51`):
- `/home/svretina/Codes/SphericalSBPOperators.jl/data/sbp8_runB_free2_scan.tsv`
- Strict `br=16` local tuning scan (`a41`, `a42`):
- `/home/svretina/Codes/SphericalSBPOperators.jl/data/sbp8_br16_a41_a42_tune_scan.tsv`
- Strict `br=16` local tuning summary:
- `/home/svretina/Codes/SphericalSBPOperators.jl/data/sbp8_br16_a41_a42_tune_summary.txt`
- Repro script for strict `br=16` local tuning:
- `/home/svretina/Codes/SphericalSBPOperators.jl/data/sbp8_br16_tune_a41_a42.jl`

## What Was Ruled Out
- Under run-B (`br=19,bw=7,fr7=10,k1417_1517_1617`), only 2 free DOFs remain:
- `a40 = V[14,15]`
- `a51 = V[18,19]`
- 2D exact scan showed best point is baseline `a40=0, a51=0`; no better `max_real` found.

- `run_A` (`br=17,fr7=11`) and `run_C` (`br=19,fr7=12`) are fully determined (no free DOFs) with current fixed constraints.

- Expanding `br` to `20..24` did not improve beyond the same best floor `max_real=6.105970805947682e-8` for the same effective structure (`fr7=10`, `k1417_1517_1617`).

## Last Completed Step (2026-03-06)
- Rebuilt and ran the `br=16` tuning script as:
- `julia --startup-file=no --project=. data/sbp8_br16_tune_a41_a42.jl`
- Strict target used during scan:
- `L = D*G` purely real and strictly negative (`max_abs_imag <= 1e-12`, `max_real <= -1e-12`),
- `S` SPD (`min(Sdiag)>0`),
- `V` SPD (boundary SPD and full-matrix SPD checks),
- exact rational accuracy constraints.
- Scan domain:
- Stage-1 coarse: `a41,a42` offsets in `[-1e-4, 1e-4]` with step `1e-5` around baseline.
- Stage-2 fine: `a41,a42` offsets in `[-1e-5, 1e-5]` with step `1e-6` around stage-1 best.
- Total evaluated points: `882`.
- Feasible exact solves: `2` (only baseline point repeated at coarse/fine center).
- Strict target hits: `0`.
- Best available (non-strict) `br=16` local point:
- `a41 = 3.706581614818424e-7`, `a42 = -7.225789076047524e-7`
- `rho(L)=35.01936236050898`
- `max_real(L)=5.171005654962693e-12`
- `max_abs_imag(L)=1.034908732090478e-7`
- `V_boundary_PD=true`, `V_full_PD=true`

## Exact Next Steps to Resume
1. Keep strict objective as hard filter:
- exact constraints, `S`/`V` SPD, `max_abs_imag(L) <= 1e-12`, `max_real(L) <= -1e-12`.

2. Move to next structure family (run-B relaxation):
- Relax one high-order zero constraint at a time from run-B around tail pairs `(17,18)` and `(18,19)`,
- keep fixed: `s1=11/20`, `v1=1`, tail scaling constraints, exact quadrature/accuracy.

3. If no strict hit in relaxed run-B family:
- switch to nullspace-parameterized search with more free DOFs (`>2`) by changing `br/bw/extra-zero-set`,
- optimize in order: strict feasibility first, then smaller `rho`.

## Notes for Tomorrow
- Ask to continue from this file:
- `/home/svretina/Codes/SphericalSBPOperators.jl/data/sbp8_resume_checkpoint_2026-03-05.md`
- All critical outputs are already written in `data/` and checkpoint subfolder above.
