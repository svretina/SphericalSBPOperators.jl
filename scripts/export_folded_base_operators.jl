using DelimitedFiles: writedlm
using InteractiveUtils: subtypes
using Printf: @printf, @sprintf
using SparseArrays: findnz, sparse, spdiagm

using SphericalSBPOperators
using SummationByPartsOperators

const _SYMBOL_SOURCE_FALLBACKS = (:central, :upwind, :standard, :convergent)
const _SKIP_SOURCE_TYPES = Set([SourceOfCoefficientsCombination])

function _construct_source_instance(T::DataType)
    if hasmethod(T, Tuple{})
        return (ok = true, source = T(), reason = "")
    end

    if hasmethod(T, Tuple{Symbol})
        for sym in _SYMBOL_SOURCE_FALLBACKS
            try
                return (ok = true, source = T(sym), reason = "")
            catch
            end
        end
        return (ok = false, source = nothing,
            reason = "no compatible Symbol constructor argument")
    end

    return (ok = false, source = nothing, reason = "no zero-arg/Symbol constructor")
end

function _source_dict()
    types_all = sort!(collect(subtypes(SourceOfCoefficients)); by = t -> string(t))
    out = Dict{String, Any}()
    for T in types_all
        if isabstracttype(T) || !isconcretetype(T) || (T in _SKIP_SOURCE_TYPES)
            continue
        end
        built = _construct_source_instance(T)
        if built.ok
            out[string(nameof(typeof(built.source)))] = built.source
        end
    end
    return out
end

function _resolve_source(name::String)
    srcs = _source_dict()
    haskey(srcs, name) && return srcs[name], collect(keys(srcs))
    return nothing, collect(keys(srcs))
end

function _write_sparse_triplet(path::AbstractString, A)
    I, J, V = findnz(A)
    rows = hcat(I, J, V)
    writedlm(path, rows, ',')
    return length(I)
end

function _parse_real_arg(s::AbstractString)
    occursin("//", s) && begin
        parts = split(s, "//")
        length(parts) == 2 || throw(ArgumentError("Invalid rational format `$s`; use `a//b`."))
        num = parse(Int, strip(parts[1]))
        den = parse(Int, strip(parts[2]))
        den == 0 && throw(ArgumentError("Rational denominator cannot be zero in `$s`."))
        return num // den
    end
    return parse(Float64, s)
end

@inline function _mma_number(x::Rational)
    den = denominator(x)
    den == 1 && return string(numerator(x))
    return string(numerator(x), "/", den)
end

@inline _mma_number(x::Integer) = string(x)

function _mma_number(x::Real)
    xf = float(x)
    if isnan(xf)
        return "Indeterminate"
    elseif isinf(xf)
        return signbit(xf) ? "-Infinity" : "Infinity"
    end

    s = lowercase(@sprintf("%.17e", xf))
    parts = split(s, 'e')
    length(parts) == 2 || return s
    base = parts[1]
    expn = parse(Int, parts[2])
    return string(base, "*^", expn)
end

function _mma_vector(v::AbstractVector{<:Real})
    return "{" * join(_mma_number.(v), ", ") * "}"
end

function _mma_matrix(A::AbstractMatrix{<:Real})
    n = size(A, 1)
    rows = Vector{String}(undef, n)
    @inbounds for i in 1:n
        rows[i] = "{" * join(_mma_number.(A[i, :]), ", ") * "}"
    end
    return "{" * join(rows, ",\n ") * "}"
end

function _build_mma_block(r_vec, G_even_dense, G_odd_dense, H_dense;
        D_dense::Union{Nothing, AbstractMatrix{<:Real}} = nothing)
    blocks = String[
        "(* Mathematica-ready folded operators *)",
        string("r = ", _mma_vector(r_vec), ";"),
        string("GEven = ", _mma_matrix(G_even_dense), ";"),
        string("GOdd = ", _mma_matrix(G_odd_dense), ";"),
        string("HcartDiagRp = ", _mma_matrix(H_dense), ";"),
    ]
    if D_dense !== nothing
        push!(blocks, string("DIV = ", _mma_matrix(D_dense), ";"))
    end
    return join(blocks, "\n\n")
end

function _raw_folded_base_operators(source;
        accuracy_order::Int,
        N::Int,
        R::Real,
        p::Int,
        mode = SafeMode(),
        build_matrix::Symbol = :probe,
        snap_factor::Float64 = 64.0,
        return_canonical::Bool = false)
    R_canonical = big(Int(N)) // 1
    Dfull, xfull, Gfull, Hfull = SphericalSBPOperators._build_full_grid_objects(
        source;
        accuracy_order = Int(accuracy_order),
        N = Int(N),
        R = R_canonical,
        mode = mode,
        build_matrix = build_matrix)

    T = eltype(xfull)
    atol_construct = if isdefined(SphericalSBPOperators, Symbol("_resolve_atol"))
        SphericalSBPOperators._resolve_atol(T, nothing)
    elseif T <: AbstractFloat
        max(T(512) * eps(T), T(1e-14))
    else
        zero(T)
    end

    r, Rop, Eeven, Eodd = SphericalSBPOperators._build_folding_operators(xfull; atol = atol_construct)
    Geven_raw = sparse(Rop * Gfull * Eeven)
    Godd_raw = sparse(Rop * Gfull * Eodd)
    snap_sparse!(Geven_raw; snap_factor = snap_factor)
    snap_sparse!(Godd_raw; snap_factor = snap_factor)

    half_factor = convert(T, 1) / convert(T, 2)
    Hcart_half = sparse(half_factor * (transpose(Eeven) * Hfull * Eeven))
    metric = spdiagm(0 => r .^ p)
    S_seed = sparse(Hcart_half * metric)
    snap_sparse!(S_seed; snap_factor = snap_factor)
    V_seed = copy(S_seed)

    if return_canonical
        return (r = r, Geven = Geven_raw, Godd = Godd_raw, S = S_seed, V = V_seed)
    end

    Tout = promote_type(eltype(r), typeof(R))
    scale_ratio = convert(Tout, R) / convert(Tout, N)

    r_scaled = Tout.(r) .* scale_ratio
    Geven_scaled = sparse((one(Tout) / scale_ratio) .* Tout.(Geven_raw))
    Godd_scaled = sparse((one(Tout) / scale_ratio) .* Tout.(Godd_raw))
    S_scaled = sparse((scale_ratio^(p + 1)) .* Tout.(S_seed))
    V_scaled = sparse((scale_ratio^(p + 1)) .* Tout.(V_seed))

    snap_sparse!(Geven_scaled; snap_factor = snap_factor)
    snap_sparse!(Godd_scaled; snap_factor = snap_factor)
    snap_sparse!(S_scaled; snap_factor = snap_factor)
    snap_sparse!(V_scaled; snap_factor = snap_factor)

    return (r = r_scaled, Geven = Geven_scaled, Godd = Godd_scaled, S = S_scaled, V = V_scaled)
end

"""
    print_folded_base_operators_mma(; kwargs...)

Build diagonal folded operators and print Mathematica-ready assignments directly
to stdout, for copy/paste into a REPL/notebook.
"""
function print_folded_base_operators_mma(;
        source_name::String = "MattssonNordström2004",
        accuracy_order::Int = 6,
        N::Int = 31,
        R::Real = 1.0,
        p::Int = 2,
        return_canonical::Bool = false,
        raw_folded::Bool = true)
    src, available = _resolve_source(source_name)
    src === nothing &&
        throw(ArgumentError(
            "Unknown source `$source_name`.\nAvailable source labels:\n  " *
            join(sort!(available), "\n  ")
        ))

    ops = if raw_folded
        _raw_folded_base_operators(src;
            accuracy_order = accuracy_order,
            N = N,
            R = R,
            p = p,
            mode = SafeMode(),
            build_matrix = :probe,
            return_canonical = return_canonical)
    else
        spherical_operators(src;
            accuracy_order = accuracy_order,
            N = N,
            R = R,
            p = p,
            mode = SafeMode(),
            return_canonical = return_canonical)
    end

    D_dense = raw_folded ? nothing : Matrix(ops.D)
    block = _build_mma_block(
        collect(ops.r),
        Matrix(ops.Geven),
        Matrix(ops.Godd),
        Matrix(ops.S);
        D_dense = D_dense)
    println(block)
    return block
end

"""
    export_folded_base_operators(; kwargs...)

Build diagonal-mass folded spherical operators and export:
- `G_even` (`ops.Geven`)
- `G_odd` (`ops.Godd`)
- `H_cartesian_half * diag(r^p)` (this is `ops.S` in the diagonal construction)

Also exports `r` and sparse-triplet versions for easy Mathematica import.
"""
function export_folded_base_operators(;
        source_name::String = "MattssonNordström2004",
        accuracy_order::Int = 6,
        N::Int = 31,
        R::Real = 1.0,
        p::Int = 2,
        outdir::String = joinpath("papers", "mathematica_folded_ops"),
        return_canonical::Bool = false,
        print_mma_to_stdout::Bool = true,
        raw_folded::Bool = true)
    N > 0 || throw(ArgumentError("`N` must be positive."))
    accuracy_order > 0 || throw(ArgumentError("`accuracy_order` must be positive."))
    p >= 0 || throw(ArgumentError("`p` must be nonnegative."))

    src, available = _resolve_source(source_name)
    src === nothing &&
        throw(ArgumentError(
            "Unknown source `$source_name`.\nAvailable source labels:\n  " *
            join(sort!(available), "\n  ")
        ))

    ops = if raw_folded
        _raw_folded_base_operators(src;
            accuracy_order = accuracy_order,
            N = N,
            R = R,
            p = p,
            mode = SafeMode(),
            build_matrix = :probe,
            return_canonical = return_canonical)
    else
        spherical_operators(src;
            accuracy_order = accuracy_order,
            N = N,
            R = R,
            p = p,
            mode = SafeMode(),
            return_canonical = return_canonical)
    end

    # In the diagonal construction, this is exactly H_cartesian_half * diag(r^p).
    Hcart_times_metric = ops.S

    mkpath(outdir)
    suffix = raw_folded ? "raw" : "repaired"
    base = "folded_ops_$(source_name)_ord$(accuracy_order)_N$(N)_p$(p)_R$(R)_$(suffix)"

    r_path = joinpath(outdir, base * "_r.csv")
    g_even_path = joinpath(outdir, base * "_G_even_dense.csv")
    g_odd_path = joinpath(outdir, base * "_G_odd_dense.csv")
    h_path = joinpath(outdir, base * "_Hcart_times_diag_rp_dense.csv")
    d_path = joinpath(outdir, base * "_DIV_dense.csv")

    g_even_triplet_path = joinpath(outdir, base * "_G_even_triplet.csv")
    g_odd_triplet_path = joinpath(outdir, base * "_G_odd_triplet.csv")
    h_triplet_path = joinpath(outdir, base * "_Hcart_times_diag_rp_triplet.csv")
    d_triplet_path = joinpath(outdir, base * "_DIV_triplet.csv")
    mma_path = joinpath(outdir, base * "_mathematica.wl")

    meta_path = joinpath(outdir, base * "_metadata.txt")

    r_vec = collect(ops.r)
    G_even_dense = Matrix(ops.Geven)
    G_odd_dense = Matrix(ops.Godd)
    H_dense = Matrix(Hcart_times_metric)
    D_dense = raw_folded ? nothing : Matrix(ops.D)

    writedlm(r_path, r_vec, ',')
    writedlm(g_even_path, G_even_dense, ',')
    writedlm(g_odd_path, G_odd_dense, ',')
    writedlm(h_path, H_dense, ',')
    if D_dense !== nothing
        writedlm(d_path, D_dense, ',')
    end

    nnz_ge = _write_sparse_triplet(g_even_triplet_path, ops.Geven)
    nnz_go = _write_sparse_triplet(g_odd_triplet_path, ops.Godd)
    nnz_h = _write_sparse_triplet(h_triplet_path, Hcart_times_metric)
    nnz_d = if D_dense === nothing
        0
    else
        _write_sparse_triplet(d_triplet_path, ops.D)
    end

    maxabs_sv = maximum(abs.(Matrix(ops.S - ops.V)))

    mma_block = _build_mma_block(
        r_vec,
        G_even_dense,
        G_odd_dense,
        H_dense;
        D_dense = D_dense)

    open(mma_path, "w") do io
        println(io, mma_block)
    end

    open(meta_path, "w") do io
        println(io, "source_name=", source_name)
        println(io, "accuracy_order=", accuracy_order)
        println(io, "N=", N)
        println(io, "R=", R)
        println(io, "p=", p)
        println(io, "return_canonical=", return_canonical)
        println(io, "mode=SafeMode()")
        println(io, "raw_folded=", raw_folded)
        println(io, "note=H_cartesian_half*diag(r^p) is exported as ops.S in diagonal construction")
        println(io, "maxabs(S-V)=", maxabs_sv)
        println(io, "nnz(G_even)=", nnz_ge)
        println(io, "nnz(G_odd)=", nnz_go)
        println(io, "nnz(H_cart_times_diag_rp)=", nnz_h)
        println(io, "nnz(DIV)=", nnz_d)
    end

    @printf("Export complete.\n")
    @printf("  r: %s\n", r_path)
    @printf("  G_even (dense): %s\n", g_even_path)
    @printf("  G_odd  (dense): %s\n", g_odd_path)
    @printf("  H_cart*diag(r^p) (dense): %s\n", h_path)
    if D_dense !== nothing
        @printf("  DIV (dense): %s\n", d_path)
    end
    @printf("  G_even (triplet): %s\n", g_even_triplet_path)
    @printf("  G_odd  (triplet): %s\n", g_odd_triplet_path)
    @printf("  H_cart*diag(r^p) (triplet): %s\n", h_triplet_path)
    if D_dense !== nothing
        @printf("  DIV (triplet): %s\n", d_triplet_path)
    end
    @printf("  Mathematica block (.wl): %s\n", mma_path)
    @printf("  metadata: %s\n", meta_path)
    @printf("  maxabs(S-V): %.6e\n", float(maxabs_sv))

    if print_mma_to_stdout
        println("\n--- Mathematica Copy Block ---")
        println(mma_block)
        println("--- End Mathematica Copy Block ---")
    end

    return (;
        r_path,
        g_even_path,
        g_odd_path,
        h_path,
        g_even_triplet_path,
        g_odd_triplet_path,
        h_triplet_path,
        d_path,
        d_triplet_path,
        mma_path,
        meta_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 31
    accuracy_order = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6
    p = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2
    R = length(ARGS) >= 4 ? _parse_real_arg(ARGS[4]) : 1.0
    source_name = length(ARGS) >= 5 ? ARGS[5] : "MattssonNordström2004"
    outdir = length(ARGS) >= 6 ? ARGS[6] : joinpath("papers", "mathematica_folded_ops")
    return_canonical = length(ARGS) >= 7 ? parse(Bool, ARGS[7]) : false
    print_mma_to_stdout = length(ARGS) >= 8 ? parse(Bool, ARGS[8]) : true
    print_only = length(ARGS) >= 9 ? parse(Bool, ARGS[9]) : false
    raw_folded = length(ARGS) >= 10 ? parse(Bool, ARGS[10]) : true

    if print_only
        print_folded_base_operators_mma(;
            source_name = source_name,
            accuracy_order = accuracy_order,
            N = N,
            R = R,
            p = p,
            return_canonical = return_canonical,
            raw_folded = raw_folded)
    else
        export_folded_base_operators(;
            source_name = source_name,
            accuracy_order = accuracy_order,
            N = N,
            R = R,
            p = p,
            outdir = outdir,
            return_canonical = return_canonical,
            print_mma_to_stdout = print_mma_to_stdout,
            raw_folded = raw_folded)
    end
end
