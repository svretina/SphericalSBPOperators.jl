"""
    WaveEvolutionResult

Container for the discrete wave evolution on the folded half grid.
"""
struct WaveEvolutionResult{T <: Real}
    t::Vector{T}
    Π::Matrix{T}
    Ψ::Matrix{T}
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
    elseif name === :psi
        return getfield(sol, :Ψ)
    end
    return getfield(sol, name)
end

function Base.propertynames(::WaveEvolutionResult, private::Bool = false)
    base = (:t, :Π, :Ψ, :energy, :r, :dt, :nsteps, :boundary_condition, :initial_data_check)
    return private ? (base..., :pi, :psi) : (base..., :pi, :psi)
end

const WaveOperators = Union{SphericalOperators,
                            Staggered.SphericalOperators,
                            NonDiagonalMass.SphericalOperators}

struct WaveKernelOperators{OpsT, DK, GK}
    parent::OpsT
    D::DK
    Geven::GK
end

function Base.getproperty(ops::WaveKernelOperators, name::Symbol)
    if name === :parent || name === :D || name === :Geven
        return getfield(ops, name)
    end
    return getproperty(getfield(ops, :parent), name)
end

function Base.propertynames(ops::WaveKernelOperators, private::Bool = false)
    names = (:parent, :D, :Geven, propertynames(getfield(ops, :parent), private)...)
    return private ? names : Base.filter(!=(:parent), names)
end

const WaveRHSOperators = Union{WaveOperators, WaveKernelOperators}

@inline _has_origin_node(::SphericalOperators) = true
@inline _has_origin_node(::Staggered.SphericalOperators) = false
@inline _has_origin_node(::NonDiagonalMass.SphericalOperators) = true
@inline _has_origin_node(ops::WaveKernelOperators) = _has_origin_node(ops.parent)

@inline _wave_scalar_mass(ops::SphericalOperators) = ops.S
@inline _wave_scalar_mass(ops::Staggered.SphericalOperators) = ops.H
@inline _wave_scalar_mass(ops::NonDiagonalMass.SphericalOperators) = ops.S
@inline _wave_scalar_mass(ops::WaveKernelOperators) = _wave_scalar_mass(ops.parent)

@inline _wave_vector_mass(ops::SphericalOperators) = ops.V
@inline _wave_vector_mass(ops::Staggered.SphericalOperators) = ops.H
@inline _wave_vector_mass(ops::NonDiagonalMass.SphericalOperators) = ops.V
@inline _wave_vector_mass(ops::WaveKernelOperators) = _wave_vector_mass(ops.parent)

@inline function _resolve_enforce_origin(ops::WaveRHSOperators, enforce_origin::Bool)
    return enforce_origin && _has_origin_node(ops)
end

struct WaveBoundaryCache{T}
    absorbing_pi::T
    absorbing_psi::T
    reflecting_pi::T
    dirichlet_psi::T
end

function _wave_boundary_cache(ops::WaveRHSOperators)
    S = _wave_scalar_mass(ops)
    V = _wave_vector_mass(ops)
    B = ops.B
    T = promote_type(eltype(ops.r), eltype(S), eltype(V), eltype(B))
    BNN = convert(T, B[end, end])
    invSN = one(T) / convert(T, S[end, end])
    invVN = one(T) / convert(T, V[end, end])
    return WaveBoundaryCache{T}(BNN * invSN / convert(T, 4),
                                BNN * invVN / convert(T, 4),
                                BNN * invSN,
                                BNN * invVN)
end

"""
    WaveODEParams(ops; boundary_condition=:absorbing, enforce_origin=true)

Parameters for `wave_system_ode!` in SciML `f!(du,u,p,t)` style.
"""
struct WaveODEParams{OpsT, CacheT}
    ops::OpsT
    boundary_condition::Symbol
    enforce_origin::Bool
    boundary_cache::CacheT
end

function WaveODEParams(ops::WaveRHSOperators;
                       boundary_condition::Symbol = :absorbing,
                       enforce_origin::Bool = true)
    bc_norm = _normalize_boundary_condition(boundary_condition)
    enforce_origin_eff = _resolve_enforce_origin(ops, enforce_origin)
    boundary_cache = _wave_boundary_cache(ops)
    return WaveODEParams{typeof(ops), typeof(boundary_cache)}(ops,
                                                              bc_norm,
                                                              enforce_origin_eff,
                                                              boundary_cache)
end
