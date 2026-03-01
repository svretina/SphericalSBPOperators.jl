"""
    WaveEvolutionResult

Container for the discrete wave evolution on the folded half grid.
"""
struct WaveEvolutionResult{T <: Real}
    t::Vector{T}
    Π::Matrix{T}
    Ξ::Matrix{T}
    energy::Vector{T}
    r::Vector{T}
    dt::T
    nsteps::Int
    boundary_condition::Symbol
    initial_data_check::Any
end

# Backward-compatible aliases.
function Base.getproperty(sol::WaveEvolutionResult, name::Symbol)
    if name === :pi
        return getfield(sol, :Π)
    elseif name === :xi
        return getfield(sol, :Ξ)
    end
    return getfield(sol, name)
end

function Base.propertynames(::WaveEvolutionResult, private::Bool = false)
    base = (:t, :Π, :Ξ, :energy, :r, :dt, :nsteps, :boundary_condition, :initial_data_check)
    return private ? (base..., :pi, :xi) : (base..., :pi, :xi)
end

const WaveOperators = Union{SphericalOperators, Staggered.SphericalOperators}

@inline _has_origin_node(::SphericalOperators) = true
@inline _has_origin_node(::Staggered.SphericalOperators) = false

@inline function _resolve_enforce_origin(ops::WaveOperators, enforce_origin::Bool)
    return enforce_origin && _has_origin_node(ops)
end

"""
    WaveODEParams(ops; boundary_condition=:absorbing, enforce_origin=true)

Parameters for `wave_system_ode!` in SciML `f!(du,u,p,t)` style.
"""
struct WaveODEParams{OpsT}
    ops::OpsT
    boundary_condition::Symbol
    enforce_origin::Bool
end

function WaveODEParams(
        ops::WaveOperators;
        boundary_condition::Symbol = :absorbing,
        enforce_origin::Bool = true
    )
    return WaveODEParams{typeof(ops)}(
        ops,
        _normalize_boundary_condition(boundary_condition),
        _resolve_enforce_origin(ops, enforce_origin)
    )
end
