function _assert_origin(r::Vector{T}, atol::T) where {T <: Real}
    isempty(r) && throw(ArgumentError("Half-grid is empty; could not locate nonnegative nodes."))
    if T <: AbstractFloat
        abs(r[1]) <= atol ||
            throw(ArgumentError("Could not identify origin node: |r[1]| = $(abs(r[1])) > atol = $atol."))
    else
        r[1] == zero(T) ||
            throw(ArgumentError("For exact arithmetic, first half-grid node must be exactly zero."))
    end
    return nothing
end

function _build_half_lookup(r::Vector{T}, atol::T) where {T <: AbstractFloat}
    scale = max(_maxabs(r), one(T))
    pair_tol = max(atol, T(64) * eps(T) * scale)
    key_to_index = Dict{Int, Int}()
    for (j, rj) in enumerate(r)
        key_to_index[round(Int, rj / pair_tol)] = j
    end
    return (key_to_index = key_to_index, pair_tol = pair_tol)
end

function _build_half_lookup(r::Vector{T}, ::T) where {T <: Real}
    return Dict{T, Int}(rj => j for (j, rj) in enumerate(r))
end

function _lookup_half_index(absx::T, lookup::NamedTuple, r::Vector{T}) where {T <: AbstractFloat}
    key = round(Int, absx / lookup.pair_tol)
    for dk in -2:2
        j = get(lookup.key_to_index, key + dk, 0)
        if j != 0 && abs(r[j] - absx) <= lookup.pair_tol
            return j
        end
    end
    throw(ArgumentError("Could not pair mirrored grid point |x| = $absx with any half-grid node."))
end

function _lookup_half_index(absx::T, lookup::Dict{T, Int}, ::Vector{T}) where {T <: Real}
    j = get(lookup, absx, 0)
    j != 0 && return j
    throw(ArgumentError("Could not pair mirrored grid point |x| = $absx with any half-grid node."))
end

function _build_folding_operators(xfull::Vector{T}; atol::T) where {T <: Real}
    M = length(xfull)
    half_indices = sort(
                        [i for i in eachindex(xfull) if xfull[i] >= -atol];
                        by = i -> xfull[i]
                       )
    isempty(half_indices) && throw(ArgumentError("No half-grid nodes found with x >= -atol."))

    r = xfull[half_indices]
    _assert_origin(r, atol)

    Nh = length(r)
    rowR = collect(1:Nh)
    valR = fill(one(T), Nh)
    Rop = sparse(rowR, half_indices, valR, Nh, M)

    lookup = _build_half_lookup(r, atol)
    row_even = Int[]
    col_even = Int[]
    val_even = T[]
    row_odd = Int[]
    col_odd = Int[]
    val_odd = T[]

    for i in eachindex(xfull)
        x = xfull[i]
        absx = abs(x)
        j = _lookup_half_index(absx, lookup, r)

        push!(row_even, i)
        push!(col_even, j)
        push!(val_even, one(T))

        sign = ifelse(x > zero(T), one(T), ifelse(x < zero(T), -one(T), zero(T)))
        if sign != zero(T)
            push!(row_odd, i)
            push!(col_odd, j)
            push!(val_odd, sign)
        end
    end

    Eeven = sparse(row_even, col_even, val_even, M, Nh)
    Eodd = sparse(row_odd, col_odd, val_odd, M, Nh)
    snap_sparse!(Rop)
    snap_sparse!(Eeven)
    snap_sparse!(Eodd)

    return r, Rop, Eeven, Eodd
end

