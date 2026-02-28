using LinearAlgebra: diag, norm
using Printf: @printf
using SummationByPartsOperators: MattssonNordström2004, derivative_operator, grid, mass_matrix

Base.@kwdef struct LHopitalOriginConfig
    R::Float64 = 10.0
    d::Int = 6
    p::Int = 2
    rational_tol::Float64 = 1e-12
    nonzero_tol::Float64 = 1e-14
end

function _canonical_full_grid(R::Float64)
    Rint = round(Int, R)
    abs(R - Rint) <= 1e-12 ||
        throw(ArgumentError("R must be integer-valued for a collocated symmetric grid; got R=$R."))
    Nfull = 2 * Rint + 1
    return Float64(Rint), Nfull
end

function _derivative_matrix_dense(Dfull, M::Int)
    maybe = try
        Matrix(Dfull)
    catch
        nothing
    end
    if maybe !== nothing && size(maybe) == (M, M)
        return Matrix{Float64}(maybe)
    end

    G = zeros(Float64, M, M)
    e = zeros(Float64, M)
    @inbounds for j in 1:M
        e[j] = 1.0
        G[:, j] = Dfull * e
        e[j] = 0.0
    end
    return G
end

function _build_folding_maps(xfull::Vector{Float64}; atol::Float64 = 1e-12)
    half_indices = sort([i for i in eachindex(xfull) if xfull[i] >= -atol]; by = i -> xfull[i])
    isempty(half_indices) && throw(ArgumentError("No nonnegative half-grid nodes found."))

    r = xfull[half_indices]
    abs(r[1]) <= atol || throw(ArgumentError("First half-grid node is not the origin: r[1]=$(r[1])."))

    Nh = length(r)
    M = length(xfull)

    Rop = zeros(Float64, Nh, M)
    @inbounds for (i, idx) in enumerate(half_indices)
        Rop[i, idx] = 1.0
    end

    Eeven = zeros(Float64, M, Nh)
    Eodd = zeros(Float64, M, Nh)
    @inbounds for i in 1:M
        absx = abs(xfull[i])
        j = findfirst(rj -> abs(rj - absx) <= atol, r)
        j === nothing && throw(ArgumentError("Could not pair |x|=$absx with half-grid nodes."))

        Eeven[i, j] = 1.0
        if xfull[i] > atol
            Eodd[i, j] = 1.0
        elseif xfull[i] < -atol
            Eodd[i, j] = -1.0
        end
    end

    return r, Rop, Eeven, Eodd
end

function _ratstr(x::Float64; tol::Float64)
    abs(x) <= tol && return "0"
    q = rationalize(BigInt, x; tol = tol)
    return string(numerator(q), "//", denominator(q))
end

function _build_split_spherical_operators(cfg::LHopitalOriginConfig)
    cfg.d > 0 || throw(ArgumentError("d must be positive."))
    cfg.p >= 0 || throw(ArgumentError("p must be non-negative."))
    cfg.rational_tol > 0 || throw(ArgumentError("rational_tol must be positive."))
    cfg.nonzero_tol > 0 || throw(ArgumentError("nonzero_tol must be positive."))

    R, Nfull = _canonical_full_grid(cfg.R)
    source = MattssonNordström2004()
    Dfull = derivative_operator(source;
                                derivative_order = 1,
                                accuracy_order = cfg.d,
                                xmin = -R,
                                xmax = R,
                                N = Nfull)

    xfull = collect(grid(Dfull))
    M = length(xfull)
    Gcart = _derivative_matrix_dense(Dfull, M)
    Hcart = Matrix{Float64}(mass_matrix(Dfull))

    r, Rop, Eeven, Eodd = _build_folding_maps(xfull)
    Nh = length(r)

    # Raw folded gradient for even fields: no repair is applied.
    G = Rop * Gcart * Eeven
    # Folded odd derivative used for L'Hopital closure.
    Godd = Rop * Gcart * Eodd

    # Folded Cartesian half-mass and metric weighting (p=2 for spherical by default).
    Hhalf = 0.5 * transpose(Eeven) * Hcart * Eeven
    Mdiag = diag(Hhalf) .* (r .^ cfg.p)
    Mdiag[1] = 0.0

    # Divergence: interior uses dr + p/r; origin row uses L'Hopital replacement.
    D = copy(Godd)
    @inbounds for i in 2:Nh
        D[i, i] += cfg.p / r[i]
    end
    D[1, :] .= (cfg.p + 1) .* Godd[1, :]

    return (
        cfg = cfg,
        R = R,
        xfull = xfull,
        r = r,
        Gcart = Gcart,
        Hcart = Hcart,
        Rop = Rop,
        Eeven = Eeven,
        Eodd = Eodd,
        G = G,
        Godd = Godd,
        D = D,
        Mdiag = Mdiag,
    )
end

function _w1_ratios(report; js::Vector{Int} = Int[])
    G = report.G
    D = report.D
    Mdiag = report.Mdiag
    Nh = length(report.r)
    tol = report.cfg.nonzero_tol

    indices = isempty(js) ? collect(2:Nh) : copy(js)
    rows = NamedTuple[]
    for j in indices
        2 <= j <= Nh || continue
        g = G[j, 1]
        d = D[1, j]
        m = Mdiag[j]
        if abs(g) <= tol || abs(d) <= tol
            push!(rows, (j = j, usable = false, G = g, D = d, M = m, w1 = NaN))
        else
            w1 = -(g * m) / d
            push!(rows, (j = j, usable = true, G = g, D = d, M = m, w1 = w1))
        end
    end
    return rows
end

function run_lhopital_origin_weight(cfg::LHopitalOriginConfig = LHopitalOriginConfig())
    report = _build_split_spherical_operators(cfg)
    Nh = length(report.r)

    println("L'Hopital-origin diagonal-mass probe")
    @printf("  config: R=%.1f, d=%d, p=%d\n", report.R, cfg.d, cfg.p)
    println("  half-grid size Nh = ", Nh)
    println("  parity: scalar even, vector odd (with xi(0)=0)")
    println("  gradient repair: disabled (raw folded G is used)")

    println("\nStep 1: Folded operators and interior mass")
    println("  Raw folded gradient G = R * D_cart * E_even")
    println("  Raw folded odd derivative G_odd = R * D_cart * E_odd")
    println("  Interior mass M_jj = (1/2 * E_even^T H_cart E_even)_jj * r_j^p, with M_11 = 0")

    println("\nStep 2: L'Hopital closure for D")
    println("  Interior rows i>=2: D = G_odd + diag(p/r)")
    println("  Origin row i=1:     D_1j = (p+1) * (G_odd)_1j")

    println("\nRequested coefficients (value and rationalized fraction)")
    println("  First row of D (D_1j):")
    @inbounds for j in 1:Nh
        v = report.D[1, j]
        @printf("    j=%2d  D_1j=% .16e   (%s)\n", j, v, _ratstr(v; tol = cfg.rational_tol))
    end

    println("  First column of G (G_j1):")
    @inbounds for j in 1:Nh
        v = report.G[j, 1]
        @printf("    j=%2d  G_j1=% .16e   (%s)\n", j, v, _ratstr(v; tol = cfg.rational_tol))
    end

    println("  Interior mass weights M_jj (j>=2):")
    @inbounds for j in 2:Nh
        v = report.Mdiag[j]
        @printf("    j=%2d  M_jj=% .16e   (%s)\n", j, v, _ratstr(v; tol = cfg.rational_tol))
    end

    println("\nStep 4: Candidate origin weight w1 from each j")
    requested_j = collect(2:min(4, Nh))
    rows_req = _w1_ratios(report; js = requested_j)
    rows_all = _w1_ratios(report)

    println("  Requested checks (j=2,3,4 when available):")
    for row in rows_req
        if row.usable
            @printf("    j=%2d  w1=-(G_j1*M_jj)/D_1j = %.16e   (%s)\n",
                    row.j, row.w1, _ratstr(row.w1; tol = cfg.rational_tol))
        else
            @printf("    j=%2d  ratio unusable (|G_j1|=%.3e, |D_1j|=%.3e)\n",
                    row.j, abs(row.G), abs(row.D))
        end
    end

    usable = [row for row in rows_all if row.usable]
    if isempty(usable)
        println("  No usable j found with nonzero G_j1 and D_1j.")
        return (report = report, requested_rows = rows_req, all_rows = rows_all, consistent = false, positive = false)
    end

    wvals = [row.w1 for row in usable]
    wref = wvals[1]
    spread = maximum(abs.(wvals .- wref))
    consistent = spread <= 1e-10 * max(1.0, abs(wref))
    positive = all(w -> w > 0.0, wvals)

    @printf("  Across all usable j>=2: count=%d, min(w1)=%.16e, max(w1)=%.16e, spread=%.3e\n",
            length(wvals), minimum(wvals), maximum(wvals), spread)
    println("  Consistent constant w1 across j? ", consistent)
    println("  Positive w1 across j?            ", positive)

    if consistent && positive
        println("  Conclusion: raw folded scheme supports a constant positive origin mass weight.")
    else
        println("  Conclusion: raw folded scheme does NOT provide a single positive constant origin mass weight.")
    end

    return (
        report = report,
        requested_rows = rows_req,
        all_rows = rows_all,
        consistent = consistent,
        positive = positive,
        spread = spread,
    )
end

function print_lhopital_cli_help(io::IO = stdout)
    println(io, "Usage: julia scripts/lhopital_origin_weight/run_lhopital_origin_weight.jl [options]")
    println(io, "")
    println(io, "Options:")
    println(io, "  --R <float>             Domain half-width (must be integer-valued). Default: 10")
    println(io, "  --d <int>               Cartesian SBP derivative accuracy order. Default: 6")
    println(io, "  --p <int>               Geometry power in div: d/dr + p/r. Default: 2")
    println(io, "  --rational-tol <float>  Tolerance for rationalized printout. Default: 1e-12")
    println(io, "  --nonzero-tol <float>   Tolerance for declaring coefficients nonzero. Default: 1e-14")
    println(io, "  -h, --help              Show this help text")
end

function parse_lhopital_cli_args(args::Vector{String})
    cfg = LHopitalOriginConfig()

    R = cfg.R
    d = cfg.d
    p = cfg.p
    rational_tol = cfg.rational_tol
    nonzero_tol = cfg.nonzero_tol

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "-h" || arg == "--help"
            print_lhopital_cli_help()
            return nothing
        end

        i < length(args) || throw(ArgumentError("Missing value for argument $arg."))
        val = args[i + 1]

        if arg == "--R"
            R = parse(Float64, val)
        elseif arg == "--d"
            d = parse(Int, val)
        elseif arg == "--p"
            p = parse(Int, val)
        elseif arg == "--rational-tol"
            rational_tol = parse(Float64, val)
        elseif arg == "--nonzero-tol"
            nonzero_tol = parse(Float64, val)
        else
            throw(ArgumentError("Unknown argument `$arg`. Use --help for usage."))
        end
        i += 2
    end

    return LHopitalOriginConfig(
        R = R,
        d = d,
        p = p,
        rational_tol = rational_tol,
        nonzero_tol = nonzero_tol,
    )
end
