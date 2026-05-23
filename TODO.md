# 
- Test the new operators from Mathematica
- use cfl=1/2 in non diagonal operators
- scaled errors from analytic solution with scalings
- use the :regular_left_moving_spherical_gaussian initial data.
- make new simulations.
- Show coefficients for an example solve






## Code

- [ ] Simplify wave solver structure
- [ ] Unify operator API
- [ ] Write Analytic solution for wave propagation on sphere.
- [ ] Unify current plotting (plots, Plots etc)
- [ ] Spectrum plots module for SSBP
- [ ] Introduce common types for SSBP operators.

## Other simplifications
- [ ] right boundary closure from SummationByPartsOperators.jl instead of calculating it.
- [ ] what does validate.jl actually do?
- [ ] in types.jl :

```
function Base.getproperty(ops::SphericalOperators, name::Symbol)
    if name === :H
        return getfield(ops, :S)
    end
    return getfield(ops, name)
end
```
this looks fishy.

- [ ] construct.jl : why exact linear solve dont we use a LinearAlgebra solver or native julia function?


## Plots

- [ ] Convergence plots (self)
- [ ] Energy stability plots (Radiative and Reflective)
- [ ] Staggered Div vs SSBP Div on f=c,r,r^3 etc.
- [ ] Simulation snapshots of wave evolution.
- [ ] Make simulation snapshots into html with MakieBake.jl

## Future
- [ ] Move wave solver into separate package, e.g. SSBPWaveSolver.jl
- [ ] Move plots into separate package, e.g. SSBPWavePlots.jl
- [ ] Move spectrum plots into separate package, e.g. SSBPOperatorSpectrum.jl
