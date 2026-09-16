using Documenter
using SphericalSBPOperators

makedocs(
    modules = [SphericalSBPOperators],
    sitename = "SphericalSBPOperators.jl",
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Examples" => "examples.md",
        "API reference" => "api.md",
    ],
)

# Add `deploydocs` here once the repository's canonical GitHub URL and release
# branch are established.
