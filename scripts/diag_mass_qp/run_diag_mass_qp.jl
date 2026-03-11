#!/usr/bin/env julia

include(joinpath(@__DIR__, "diag_mass_qp_core.jl"))

function main(args::Vector{String})
    cfg = parse_cli_args(args)
    cfg === nothing && return
    run_diag_mass_qp(cfg)
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main(ARGS)
    catch err
        println(stderr, "Error: ", err)
        println(stderr, "")
        print_cli_help(stderr)
        rethrow(err)
    end
end
