"""
    apply_symmetry_state!(Π, Ψ; enforce_origin=true)

Mutating symmetry constraints on the state variables.

- `Π` is even at the origin (no direct point constraint needed here).
- `Ψ` is odd at the origin, so `Ψ(0)=0` is imposed by setting `Ψ[1]=0`.
"""
function apply_symmetry_state!(Π::AbstractVector,
                               Ψ::AbstractVector;
                               enforce_origin::Bool = true,
                               has_origin_node::Bool = true)
    n = length(Π)
    length(Ψ) == n || throw(DimensionMismatch("`Π` and `Ψ` must have matching lengths."))

    if enforce_origin && has_origin_node && n >= 1
        Ψ[1] = zero(eltype(Ψ))
    end

    return nothing
end

"""
    apply_symmetry_rhs!(dΠ, dΨ; enforce_origin=true)

Mutating symmetry constraints on the RHS variables.

For stability with metric-weighted mass (`H[1,1]=0` for `p>0`), enforce invariance
of the odd subspace at the origin by imposing `dΨ(0)=0`.
"""
function apply_symmetry_rhs!(dΠ::AbstractVector,
                             dΨ::AbstractVector;
                             enforce_origin::Bool = true,
                             has_origin_node::Bool = true)
    n = length(dΠ)
    length(dΨ) == n || throw(DimensionMismatch("`dΠ` and `dΨ` must have matching lengths."))

    if enforce_origin && has_origin_node && n >= 1
        dΨ[1] = zero(eltype(dΨ))
    end

    return nothing
end

"""
    initialize_wave_state!(Π, Ψ; enforce_origin=true)

Apply required symmetry constraints to initial state data.
"""
function initialize_wave_state!(Π::AbstractVector,
                                Ψ::AbstractVector;
                                enforce_origin::Bool = true,
                                has_origin_node::Bool = true)
    apply_symmetry_state!(Π,
                          Ψ;
                          enforce_origin = enforce_origin,
                          has_origin_node = has_origin_node)
    return nothing
end

"""
    apply_symmetry_constraints!(Π, Ψ; enforce_origin=true)

Backward-compatible alias for `apply_symmetry_state!`.
"""
function apply_symmetry_constraints!(Π::AbstractVector,
                                     Ψ::AbstractVector;
                                     enforce_origin::Bool = true,
                                     has_origin_node::Bool = true)
    apply_symmetry_state!(Π,
                          Ψ;
                          enforce_origin = enforce_origin,
                          has_origin_node = has_origin_node)
    return nothing
end

function _default_wave_check_tol(::Type{T}) where {T <: AbstractFloat}
    return T(100) * eps(T)
end

function _default_wave_check_tol(::Type{T}) where {T <: Real}
    return zero(T)
end

"""
    check_wave_data_consistency(Π, Ψ; boundary_condition=:absorbing,
                                enforce_origin=true, require_boundary=false,
                                tol=nothing)

Check whether a wave-state pair `(Π, Ψ)` is consistent with imposed parity and, when
requested, boundary-condition residual tolerance.

For SAT evolution, `require_boundary=false` is generally appropriate for initial data,
since boundary conditions are imposed weakly through the RHS, not by state projection.
"""
function check_wave_data_consistency(Π::AbstractVector,
                                     Ψ::AbstractVector;
                                     boundary_condition::Symbol = :absorbing,
                                     enforce_origin::Bool = true,
                                     has_origin_node::Bool = true,
                                     require_boundary::Bool = false,
                                     tol = nothing)
    n = length(Π)
    length(Ψ) == n || throw(DimensionMismatch("`Π` and `Ψ` must have matching lengths."))
    n > 0 || throw(ArgumentError("State vectors must be non-empty."))

    bc_norm = _normalize_boundary_condition(boundary_condition)

    T = promote_type(eltype(Π), eltype(Ψ))
    tolT = tol === nothing ? _default_wave_check_tol(T) : convert(T, tol)

    max_abs_pi = maximum(abs.(Π))
    max_abs_pi_interior = n > 1 ? maximum(abs.(view(Π, 1:(n - 1)))) : max_abs_pi
    max_abs_xi = maximum(abs.(Ψ))

    enforce_origin_effective = enforce_origin && has_origin_node
    origin_residual = enforce_origin_effective ? abs(Ψ[1]) : zero(T)
    origin_ok = !enforce_origin_effective || origin_residual <= tolT

    boundary_residual = abs(boundary_characteristic_residual(Π, Ψ; bc = bc_norm))
    boundary_ok = bc_norm === :none || boundary_residual <= tolT

    finite_ok = all(isfinite, Π) && all(isfinite, Ψ)
    consistent = finite_ok && origin_ok && (!require_boundary || boundary_ok)

    return (consistent = consistent,
            finite_ok = finite_ok,
            origin_ok = origin_ok,
            boundary_ok = boundary_ok,
            require_boundary = require_boundary,
            origin_residual = origin_residual,
            boundary_residual = boundary_residual,
            max_abs_pi = max_abs_pi,
            max_abs_pi_interior = max_abs_pi_interior,
            max_abs_xi = max_abs_xi,
            boundary_condition = bc_norm,
            enforce_origin = enforce_origin_effective,
            has_origin_node = has_origin_node,
            tol = tolT)
end
