# Run this script from the repository environment, for example:
#   julia --project=. scripts/construct_operators.jl
#
# Load the package source directly from this checkout.
# This keeps the script convenient while we are developing inside the repository.
include(joinpath(@__DIR__, "..", "src", "SphericalSBPOperators.jl"))

using SummationByPartsOperators: MattssonNordström2004, SafeMode

"""
    construct_operators(; kwargs...)

Build folded spherical SBP operators through the unified construction API.

This script is intentionally organized around a small set of clearly named parameters:

- grid parameters:
  - `N`: number of subintervals on `[0, R]`; the folded grid has `N + 1` nodes
  - `R`: physical outer radius of the domain
  - `p`: metric power (`0` Cartesian, `1` cylindrical, `2` spherical)
- accuracy parameter:
  - `accuracy_order`: design order requested from the constructor
- family selector:
  - `family = :diagonal`, `:staggered`, `:non_diagonal`, or `:all`
- SBP source and arithmetic mode:
  - `source`: full-grid SBP family used underneath
  - `mode`: arithmetic mode forwarded to the constructors

The routine returns a dictionary keyed by construction family.
"""

# Default parameters for a clean first run.
# Edit these values in one place when you want to change the example setup.
const DEFAULT_CONSTRUCTION_CONFIG = (
                                     # Which operator family to build.
                                     # Use :all to build diagonal, staggered, and non-diagonal operators together.
                                     family = :all,

                                     # Full-grid SBP source family used by all constructors.
                                     source = MattssonNordström2004(),

                                     # Requested design accuracy order.
                                     # The current unified non-diagonal wrapper supports 4 and 6, but not 8.
                                     accuracy_order = 4,

                                     # Number of subintervals on [0, R].
                                     # Diagonal and non-diagonal constructors both include the boundary node.
                                     N = 32,

                                     # Physical outer radius of the computational domain [0, R].
                                     R = 1.0,

                                     # Metric power in r^p.
                                     # Use p = 2 for spherical symmetry.
                                     p = 2,

                                     # SBP arithmetic/extraction mode from SummationByPartsOperators.
                                     mode = SafeMode(),

                                     # Print a short summary for each constructed operator.
                                     verbose = true)

function _normalized_families(family::Symbol)
    family === :all && return (:diagonal, :staggered, :non_diagonal)
    family in (:diagonal, :staggered, :non_diagonal) ||
        throw(ArgumentError("`family` must be :diagonal, :staggered, :non_diagonal, or :all."))
    return (family,)
end

function _construct_operator(family::Symbol, config)
    if family === :diagonal
        return SphericalSBPOperators.diagonal_spherical_operators(config.source;
                                                         accuracy_order = config.accuracy_order,
                                                         N = config.N,
                                                         R = config.R,
                                                         p = config.p,
                                                         mode = config.mode)
    elseif family === :staggered
        return SphericalSBPOperators.staggered_spherical_operators(config.source;
                                                                   accuracy_order = config.accuracy_order,
                                                                   N = config.N,
                                                                   R = config.R,
                                                                   p = config.p,
                                                                   mode = config.mode)
    end

    return SphericalSBPOperators.non_diagonal_spherical_operators(config.source;
                                                                  accuracy_order = config.accuracy_order,
                                                                  N = config.N,
                                                                  R = config.R,
                                                                  p = config.p,
                                                                  mode = config.mode)
end

function _validate_operator(family::Symbol, ops, accuracy_order::Int)
    if family === :diagonal
        return SphericalSBPOperators.validate(ops;
                                              max_monomial_degree = accuracy_order,
                                              verbose = false)
    elseif family === :staggered
        return SphericalSBPOperators.validate_staggered(ops;
                                                        max_monomial_degree = accuracy_order,
                                                        verbose = false)
    end

    # There is not yet a non-diagonal validation report with the same public API.
    return nothing
end

function _print_summary(family::Symbol, ops, report)
    println("[$family]")
    println("  type = ", typeof(ops))
    println("  Nh = ", ops.Nh, ", M_full = ", ops.M_full, ", closure_width = ",
            ops.closure_width)
    println("  accuracy_order = ", ops.accuracy_order, ", p = ", ops.p, ", R = ", ops.R)
    println("  matrix sizes: Geven ", size(ops.Geven), ", D ", size(ops.D))

    if report === nothing
        println("  validation = not available through a unified public report yet")
    else
        println("  sbp_no_origin = ", report.sbp.sbp_no_origin)
    end
end

function construct_operators(;
                             family::Symbol = DEFAULT_CONSTRUCTION_CONFIG.family,
                             source = DEFAULT_CONSTRUCTION_CONFIG.source,
                             accuracy_order::Int = DEFAULT_CONSTRUCTION_CONFIG.accuracy_order,
                             N::Int = DEFAULT_CONSTRUCTION_CONFIG.N,
                             R = DEFAULT_CONSTRUCTION_CONFIG.R,
                             p::Int = DEFAULT_CONSTRUCTION_CONFIG.p,
                             mode = DEFAULT_CONSTRUCTION_CONFIG.mode,
                             verbose::Bool = DEFAULT_CONSTRUCTION_CONFIG.verbose)
    config = (;
              family = family,
              source = source,
              accuracy_order = accuracy_order,
              N = N,
              R = R,
              p = p,
              mode = mode,
              verbose = verbose,)

    results = Dict{Symbol, NamedTuple}()
    for family_name in _normalized_families(config.family)
        ops = _construct_operator(family_name, config)
        report = _validate_operator(family_name, ops, config.accuracy_order)
        results[family_name] = (ops = ops, report = report)

        if config.verbose
            _print_summary(family_name, ops, report)
        end
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = construct_operators()

    println()
    println("Constructed families: ", join(string.(collect(keys(results))), ", "))
end
