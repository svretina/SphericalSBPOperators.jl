#!/usr/bin/env julia

include(joinpath(@__DIR__, "lhopital_origin_weight_core.jl"))

function main(args::Vector{String})
    cfg = parse_lhopital_cli_args(args)
    cfg === nothing && return
    run_lhopital_origin_weight(cfg)
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main(ARGS)
    catch err
        println(stderr, "Error: ", err)
        println(stderr, "")
        print_lhopital_cli_help(stderr)
        rethrow(err)
    end
end
