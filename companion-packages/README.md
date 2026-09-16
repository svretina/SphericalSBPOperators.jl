# Companion package staging area

This directory preserves code intentionally removed from the core operator
package. It is not loaded, tested, or dependency-managed by
`SphericalSBPOperators.jl`.

- `SphericalSBPWave/`: ODE integration, initial/boundary data, and wave tests.
- `SphericalSBPPlots/`: Makie themes, plotting utilities, and legacy plotting code.
- `SphericalSBPAnalysis/`: convergence and spectrum-analysis utilities.

Each future companion should become an independent Julia package with its own
`Project.toml`, module entry point, tests, and a dependency on
`SphericalSBPOperators`. Replace former parent-module access to private helpers
with the public operator API and `scalar_mass`, `vector_mass`, and
`has_origin_node`.
