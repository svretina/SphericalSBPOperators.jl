# H11=0 vs H11=w1>0: Detailed Findings on Pole Closure, Gradient Repair, and Divergence Consistency

## 1) Executive summary

This note summarizes what works, what fails, and why, when comparing:

- the old construction in `src/construct.jl` (degenerate pole mass, `H11 = 0`), and
- the new positive-origin-mass construction in `scripts/lhopital_gradient_repair` (`H11 = w1 > 0`).

Main conclusion:

1. The old method is internally consistent because `H11=0` makes the pole SBP row degenerate, so one can set `Geven[j,1]=0` for `j>=2` and still overwrite `D[1,:]` with LHopital closure.
2. With `H11=w1>0`, the pole row is no longer degenerate. Then `D[1,j]` and `G[j,1]` are algebraically coupled by SBP for every `j>=2`.
3. This creates a real compatibility constraint: if `D1[j]` is fixed by LHopital and nonzero, then `G[j,1]` cannot be zero.
4. Local exact coupled repairs are feasible, but strict global exactness with all desired constraints appears infeasible for the tested setup (`acc=6`, `p=2`, 41 points).

---

## 2) Problem setup and notation

Grid and operators on folded half domain `r in [0,R]`:

- `G` (even-to-odd gradient),
- `D` (odd-to-even divergence),
- `H` (diagonal metric mass),
- `B` (outer boundary operator).

Discrete SBP identity:

`H*D + G^T*H = B`

LHopital pole closure for odd flux in `d=p+1` dimensions:

`D[1,:] = (p+1)*Godd[1,:]`.

For all tests in this report:

- `p=2` (3D spherical),
- Cartesian base accuracy order `acc=6`,
- half-grid points on `[0,R]`: 41.

---

## 3) Why old method works with H11=0

### 3.1 What old method imposes (from `src/construct.jl`)

Relevant implementation points:

- It detects rows with nonzero first column leakage and repairs those rows:
  - `_rows_with_nonzero_first_column` in [construct.jl](/home/svretina/Codes/SphericalSBPOperators.jl/src/construct.jl:81)
- It solves coupled constraints for those rows (gradient + divergence moments):
  - `_solve_coupled_geven_block` in [construct.jl](/home/svretina/Codes/SphericalSBPOperators.jl/src/construct.jl:373)
- It explicitly sets `Geven[row,1] = 0` for repaired rows:
  - [construct.jl:584](/home/svretina/Codes/SphericalSBPOperators.jl/src/construct.jl:584)
- It derives `D[i>=2,:]` from SBP and then overwrites pole row with LHopital:
  - `_set_divergence_rows!` [construct.jl:32](/home/svretina/Codes/SphericalSBPOperators.jl/src/construct.jl:32)
  - `_set_origin_row!` [construct.jl:50](/home/svretina/Codes/SphericalSBPOperators.jl/src/construct.jl:50)
- Package docs explicitly note pole degeneracy for `p>0`:
  - [SphericalSBPOperators.jl:77](/home/svretina/Codes/SphericalSBPOperators.jl/src/SphericalSBPOperators.jl:77)

### 3.2 Algebraic reason

At entry `(1,j)` (`j>=2`) of SBP:

`H11*D1j + (G^T H)1j = B1j = 0`.

With diagonal `H`, `(G^T H)1j = G[j,1]*Hjj`.

If `H11=0`, this reduces to:

`G[j,1]*Hjj = 0`.

Since `Hjj>0` for `j>=2`, it enforces `G[j,1]=0`, but imposes **no condition on `D1j`**.

Therefore, one can set `G[j,1]=0` and independently set pole row `D[1,:]` to LHopital. This is exactly why the old method can be both stable and clean locally.

### 3.3 What old method used for acc=6

Direct introspection of old repair routine (`N=40`, equivalent to 41 half-grid points) gave:

- `rows_to_solve = [2,3,4]`
- requested divergence odd degrees: `[1,3]`
- used divergence odd degrees: `[1,3]`
- stencil columns: `[2,3,4,5,6,7]`
- equations/unknowns: `18/18` (exact square solve)

So the old method did not need to drop divergence moments in this case.

---

## 4) What changes when H11 is replaced by w1>0

In the new experiment, we set positive pole mass:

`H11 = w1 > 0`.

Pole SBP coupling becomes:

`w1*D1j + G[j,1]*Hjj = 0` for `j>=2`.

If `D1j` is prescribed by LHopital and nonzero, then

`G[j,1] = -(w1/Hjj)*D1j`.

So `G[j,1]=0` is no longer admissible unless `D1j=0`.

This is the central structural difference.

---

## 5) What I tried for H11=w1>0 and what happened

All below used `acc=6`, `p=2`, 41 points, `w1` from j=2 raw formula (`w1=1/6`, positive).

### 5.1 Method A: local closed-block exact coupled solve (feasible)

Current implementation in `scripts/lhopital_gradient_repair`:

- Keep LHopital row fixed: `D1 = (p+1)*Godd[1,:]`.
- Set first-column compatibility on leakage rows `J_aff=[2,3,4]`:
  `G[j,1] = -(w1*D1[j])/Hjj`.
- Solve coupled gradient+divergence constraints exactly on a near-origin block
  `J_solve=[2,3,4,5,6,7]`.
- Auto-selected stencil for this case: columns `[2..11]`.

Result:

- SBP residual exactly zero.
- Pole row exactly retained.
- Constraints exact on solved rows.
- But unconstrained rows still show q=1 divergence leakage:
  - row 8: `-180/833`
  - row 9: `735/2176` (max)
  - row 10: `-4900/26163`
  - row 11: `294/8075`

Interpretation: local exactness is achieved, but transpose coupling spreads effects into nearby unconstrained rows.

### 5.2 Trying to also constrain the touched interior rows (partial or failed)

I tried variants to force q=1 exactness beyond repair rows:

1. Add q=1 constraints on repaired rows plus selected interior rows touched by stencil band.
   - This can fix row 6, but then max leakage moved outward (example: row 9).
2. Expand constrained set dynamically with touched columns/rows.
   - This triggers cascade behavior (constraints propagate outward).
3. Enforce q=1 on rows union stencil columns.
   - Became algebraically infeasible (no exact rational solve) for tested widths.

Reason: each added divergence-row constraint introduces transposed dependencies that require new unknown rows/columns, which in turn induce more coupled constraints.

### 5.3 Method B: global coupled exact solve (infeasible)

I attempted global exact coupling over nearly all non-origin rows:

- rows `2..40` (and also variant with `2..41`),
- columns `2..41`,
- even gradient moments + odd divergence moments.

Outcome:

- Exact linear system had no exact solution (`inconsistent`).

### 5.4 Global LSQ fallback (feasible numerically, poor exactness)

Using least squares on global coupled equations:

- Pole row match stayed exact by construction,
- SBP residual numerically tiny (~`1e-14`),
- but moment errors became large at higher degree / outer boundary closure.

Example from run:

- LSQ residual max in coupled equations: `13.272...`
- divergence q=1 max abs error: `1.18e-3` (row 41)
- divergence q=3 max abs error: `3.43` (row 41)
- gradient q=6 max abs error: `2.2e5` (row 41)

Interpretation: numerical compromise is possible, but not the exact high-order behavior we want.

---

## 6) Why these failures happen (root cause)

### 6.1 Nondegenerate pole row removes the old decoupling

With `H11=0`, pole row is degenerate and allows independent LHopital overwrite.
With `H11=w1>0`, pole row directly couples `D1j` and `G[j,1]`.

### 6.2 Transpose coupling creates row/column dependency graph

Divergence constraints involve `G^T*H`, so fixing/solving a row block in `G` changes divergence equations at rows corresponding to affected columns. This is why errors can appear just outside the repaired row block.

### 6.3 Exact global compatibility appears overconstrained for this stencil family

For tested operator family/closure, demanding all target moments exactly with positive `w1` and fixed LHopital row appears algebraically inconsistent globally.

---

## 7) Comparative conclusion

### Old method (`H11=0`)

Pros:

- Exact local coupled repair with compact stencil,
- clean first-column zeroing,
- no pole-row coupling conflict.

Cons:

- `H` is semidefinite at origin (`H11=0`), not strictly positive definite.

### New method (`H11=w1>0`)

Pros:

- positive mass at origin,
- physically meaningful nondegenerate pole quadrature weight,
- exact local coupled block feasible.

Cons:

- introduces unavoidable SBP coupling `w1*D1j + G[j,1]Hjj = 0`,
- strict global exactness appears infeasible under current constraints,
- local exact repairs can leave unconstrained-neighbor leakage unless block is expanded (which tends toward cascade/infeasibility).

---

## 8) Practical recommendation

If strict positive definiteness at origin is non-negotiable (`w1>0`), then the consistent strategy is:

1. Use a clearly defined local exact block with coupled constraints,
2. report residual leakage outside that block explicitly,
3. optionally use optimization (weighted LSQ/QP) to minimize global leakage while preserving exact priority constraints.

If exact global algebraic closure with compact stencils is prioritized over positive-definite origin mass, the old `H11=0` structure is naturally compatible with the `G[:,1]=0` cleanup.

---

## 9) Files involved

- Old construction:
  - [construct.jl](/home/svretina/Codes/SphericalSBPOperators.jl/src/construct.jl)
  - [SphericalSBPOperators.jl](/home/svretina/Codes/SphericalSBPOperators.jl/src/SphericalSBPOperators.jl)
- New experiments:
  - [lhopital_gradient_repair_core.jl](/home/svretina/Codes/SphericalSBPOperators.jl/scripts/lhopital_gradient_repair/lhopital_gradient_repair_core.jl)
  - [run_lhopital_gradient_repair.jl](/home/svretina/Codes/SphericalSBPOperators.jl/scripts/lhopital_gradient_repair/run_lhopital_gradient_repair.jl)
