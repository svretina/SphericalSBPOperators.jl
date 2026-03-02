using CDDLib
using Ipopt
using JuMP:
    GenericModel,
    MOI,
    @constraint,
    @objective,
    @variable,
    optimize!,
    set_start_value,
    set_optimizer_attribute,
    termination_status,
    value
using LinearAlgebra: Symmetric, cholesky, issymmetric

export construct_split_mass_matrices
export construct_split_mass_matrices_optimization
export construct_split_mass_matrices_cddlib
export split_mass_constraint_report

include("split_mass.jl")
include("dispatch.jl")
