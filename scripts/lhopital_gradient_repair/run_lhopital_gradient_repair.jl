#!/usr/bin/env julia

include(joinpath(@__DIR__, "lhopital_gradient_repair_core.jl"))

function main(args::Vector{String})
    parsed = parse_lhopital_gradient_repair_cli_args(args)
    parsed === nothing && return
    run_lhopital_gradient_repair(parsed.cfg; R_phys = parsed.R_phys)
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main(ARGS)
    catch err
        println(stderr, "Error: ", err)
        println(stderr)
        print_lhopital_gradient_repair_cli_help(stderr)
        rethrow(err)
    end
end
