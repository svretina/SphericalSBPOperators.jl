using InteractiveUtils: subtypes
using GenericSchur
using LinearAlgebra: schur
using MultiFloats: Float64x4
using Printf: @printf, @sprintf

using SphericalSBPOperators
using SummationByPartsOperators

const _SYMBOL_SOURCE_FALLBACKS = (:central, :upwind, :standard, :convergent)
const _SKIP_SOURCE_TYPES = Set([SourceOfCoefficientsCombination])

const _SOURCE_TO_CITE = Dict(
    "DienerDorbandSchnetterTiglio2007" => "Diener2007",
    "Mattsson2012" => "Mattsson2012",
    "Mattsson2014" => "Mattsson2014a",
    "Mattsson2017" => "Mattsson2017",
    "MattssonAlmquistCarpenter2014Extended" => "Mattsson2014b",
    "MattssonNordström2004" => "Mattsson2004b",
    "MattssonSvärdNordström2004" => "Mattsson2004b",
    "MattssonSvärdShoeybi2008" => "Mattsson2008",
    "WilliamsDuru2024" => "Williams2024",
)

const _CITE_SORT_INDEX = Dict(
    "Diener2007" => 1,
    "Mattsson2012" => 2,
    "Mattsson2014a" => 3,
    "Mattsson2017" => 4,
    "Mattsson2014b" => 5,
    "Mattsson2004b" => 6,
    "Mattsson2008" => 7,
    "Williams2024" => 8,
)

@inline _source_label(source) = string(nameof(typeof(source)))

function _construct_source_instance(T::DataType)
    if hasmethod(T, Tuple{})
        return (ok=true, source=T(), reason="")
    end

    if hasmethod(T, Tuple{Symbol})
        for sym in _SYMBOL_SOURCE_FALLBACKS
            try
                return (ok=true, source=T(sym), reason="")
            catch
            end
        end
        return (ok=false, source=nothing, reason="no compatible Symbol constructor argument")
    end

    return (ok=false, source=nothing, reason="no zero-arg/Symbol constructor")
end

function collect_sbp_sources()
    types_all = sort!(collect(subtypes(SourceOfCoefficients)); by=t -> string(t))
    sources = Any[]
    skipped = NamedTuple[]

    for T in types_all
        if isabstracttype(T) || !isconcretetype(T) || (T in _SKIP_SOURCE_TYPES)
            push!(skipped, (; type=string(T), reason="abstract/unsupported meta-type"))
            continue
        end

        built = _construct_source_instance(T)
        if built.ok
            push!(sources, built.source)
        else
            push!(skipped, (; type=string(T), reason=built.reason))
        end
    end

    return (sources=sources, skipped=skipped)
end

@inline _as_big_rational(x::Rational{BigInt}) = x
@inline _as_big_rational(x::Integer) = big(x) // 1
@inline _as_big_rational(x::Rational{<:Integer}) = big(numerator(x)) // big(denominator(x))

function _as_big_rational(x::AbstractFloat)
    isfinite(x) || throw(ArgumentError("Cannot convert non-finite floating-point value `$x` to Rational{BigInt}."))
    return rationalize(BigInt, BigFloat(x))
end

function _as_big_rational(x::Real)
    throw(ArgumentError("Could not convert value of type $(typeof(x)) to Rational{BigInt}."))
end

function _rationalize_matrix(A::AbstractMatrix{<:Real})
    m, n = size(A)
    Aq = Matrix{Rational{BigInt}}(undef, m, n)
    @inbounds for j in 1:n
        for i in 1:m
            Aq[i, j] = _as_big_rational(A[i, j])
        end
    end
    return Aq
end

function _high_precision_schur_values(A::AbstractMatrix{<:Real})
    Aq = _rationalize_matrix(A)
    Ahp = Matrix{Float64x4}(Aq)
    return schur(Ahp).values
end

@inline function _clean_small(x::Real; tol::Float64=1e-16)
    xf = Float64(x)
    return abs(xf) < tol ? 0.0 : xf
end

function _spectral_metrics(vals::AbstractVector{<:Complex}; tiny_zero_tol::Float64=1e-16)
    re = Float64.(real.(vals))
    im = Float64.(imag.(vals))
    mags = Float64.(abs.(vals))
    return (
        max_re=_clean_small(maximum(re); tol=tiny_zero_tol),
        min_re=_clean_small(minimum(re); tol=tiny_zero_tol),
        max_im=_clean_small(maximum(im); tol=tiny_zero_tol),
        min_im=_clean_small(minimum(im); tol=tiny_zero_tol),
        rho=_clean_small(maximum(mags); tol=tiny_zero_tol),
    )
end

function _build_banded_ops(source, order::Int; points::Int, R::Rational{BigInt}, p::Int, mode)
    h = R / (points - 1)
    if order == 4
        solved = sbp4_solve_accuracy_constraints(
            source;
            accuracy_order=4,
            points=points,
            h=h,
            R=R,
            p=p,
            mode=mode,
            verbose=false
        )
        return (
            r=solved.setup.r,
            D=solved.D,
            Geven=solved.setup.Geven,
            H=solved.setup.Hcart_half,
            B=solved.B,
            source=source,
        )
    elseif order == 6
        solved = sbp6_solve_accuracy_constraints(
            source;
            accuracy_order=6,
            points=points,
            h=h,
            R=R,
            p=p,
            mode=mode,
            verbose=false
        )
        return (
            r=solved.setup.r,
            D=solved.D,
            Geven=solved.setup.Geven,
            H=solved.setup.Hcart_half,
            B=solved.B,
            source=source,
        )
    end
    throw(ArgumentError("Unsupported banded order: $order"))
end

function _assemble_hyperbolic_block(ops; bc::Symbol)
    bc in (:reflecting, :absorbing) ||
        throw(ArgumentError("Unsupported boundary condition `$bc`; use `:reflecting` or `:absorbing`."))

    D = Matrix(ops.D)
    G = Matrix(ops.Geven)
    n = size(D, 1)
    size(D, 1) == size(D, 2) || throw(DimensionMismatch("`D` must be square."))
    size(G, 1) == size(G, 2) == n || throw(DimensionMismatch("`G` must be $(n)x$(n)."))

    T = promote_type(eltype(D), eltype(G), eltype(ops.H), eltype(ops.B))
    Z = zeros(T, n, n)
    L = [Z Matrix{T}(D); Matrix{T}(G) Z]

    BNN = convert(T, ops.B[end, end])
    HNN = convert(T, ops.H[end, end])
    HNN == zero(T) && throw(ArgumentError("`H[end,end]` must be nonzero for SAT closure."))
    inv_hnn = one(T) / HNN

    if bc == :absorbing
        coeff = -(BNN / convert(T, 2)) * inv_hnn
        L[n, n] += coeff
        L[n, 2 * n] += coeff
        L[2 * n, n] += coeff
        L[2 * n, 2 * n] += coeff
    elseif bc == :reflecting
        L[n, 2 * n] += -(BNN * inv_hnn)
    end

    return L
end

@inline _fmt_sci(x::Real) = @sprintf("%.6e", Float64(x))

@inline function _source_cell(row)
    if haskey(_SOURCE_TO_CITE, row.label)
        return "\\cite{" * row.cite_key * "}"
    end
    return row.label
end

function _export_tables_to_dir(paths::Vector{String},
                               export_tables_dir::Union{Nothing, AbstractString})
    isnothing(export_tables_dir) && return
    mkpath(export_tables_dir)
    for src in paths
        dst = joinpath(export_tables_dir, basename(src))
        cp(src, dst; force = true)
        println("Copied: ", dst)
    end
end

function _write_table(io, rows::AbstractVector; caption::AbstractString, label::AbstractString)
    println(io, "    \\begin{table}[htbp]")
    println(io, "        \\centering")
    println(io, "        \\small")
    println(io, "        \\setlength{\\tabcolsep}{4pt}")
    println(io, "        \\begin{tabular}{p{0.30\\linewidth}rrrrr}")
    println(io, "            \\toprule")
    println(io, "            Source & \$\\max \\Re(\\lambda)\$ & \$\\min \\Re(\\lambda)\$ & \$\\max \\Im(\\lambda)\$ & \$\\min \\Im(\\lambda)\$ & \$\\rho(L)\$ \\\\")
    println(io, "            \\midrule")
    for row in rows
        m = row.metrics
        println(io, "            ", _source_cell(row), " & ", _fmt_sci(m.max_re), " & ", _fmt_sci(m.min_re), " & ",
            _fmt_sci(m.max_im), " & ", _fmt_sci(m.min_im), " & ", _fmt_sci(m.rho), " \\\\")
    end
    println(io, "            \\bottomrule")
    println(io, "        \\end{tabular}")
    println(io, "        \\caption{", caption, "}")
    println(io, "        \\label{", label, "}")
    println(io, "    \\end{table}")
end

function _sort_rows!(rows::Vector)
    sort!(rows; by=row -> (get(_CITE_SORT_INDEX, row.cite_key, 10_000), row.label))
    return rows
end

function _compute_order_rows(order::Int;
        points::Int,
        R::Rational{BigInt},
        p::Int,
        mode,
        tiny_zero_tol::Float64)
    gathered = collect_sbp_sources()
    rows_hyper_ref = NamedTuple[]
    rows_hyper_rad = NamedTuple[]
    rows_laplacian = NamedTuple[]
    failures = NamedTuple[]

    total = length(gathered.sources)
    ok = 0
    for (idx, src) in enumerate(gathered.sources)
        label = _source_label(src)
        cite_key = get(_SOURCE_TO_CITE, label, label)
        try
            ops = _build_banded_ops(src, order; points=points, R=R, p=p, mode=mode)

            L_ref = _assemble_hyperbolic_block(ops; bc=:reflecting)
            L_rad = _assemble_hyperbolic_block(ops; bc=:absorbing)
            L_lap = Matrix(ops.D * ops.Geven)

            metrics_ref = _spectral_metrics(_high_precision_schur_values(L_ref); tiny_zero_tol=tiny_zero_tol)
            metrics_rad = _spectral_metrics(_high_precision_schur_values(L_rad); tiny_zero_tol=tiny_zero_tol)
            metrics_lap = _spectral_metrics(_high_precision_schur_values(L_lap); tiny_zero_tol=tiny_zero_tol)

            push!(rows_hyper_ref, (; label=label, cite_key=cite_key, metrics=metrics_ref))
            push!(rows_hyper_rad, (; label=label, cite_key=cite_key, metrics=metrics_rad))
            push!(rows_laplacian, (; label=label, cite_key=cite_key, metrics=metrics_lap))
            ok += 1
            @printf("[order=%d] [%d/%d] OK   %s\n", order, idx, total, label)
        catch err
            msg = sprint(showerror, err)
            push!(failures, (; label=label, error=msg))
            @printf("[order=%d] [%d/%d] FAIL %s :: %s\n", order, idx, total, label, msg)
        end
    end

    _sort_rows!(rows_hyper_ref)
    _sort_rows!(rows_hyper_rad)
    _sort_rows!(rows_laplacian)
    return (
        hyper_reflective=rows_hyper_ref,
        hyper_radiative=rows_hyper_rad,
        laplacian=rows_laplacian,
        failures=failures,
        skipped=gathered.skipped,
        success_count=ok,
        total_count=total,
    )
end

function write_hyperbolic_tables(path::AbstractString, order_rows::Dict{Int,Any};
        points::Int,
        p::Int,
        R::Rational{BigInt})
    mkpath(dirname(path))
    rtxt = @sprintf("%.1f", Float64(R))
    open(path, "w") do io
        println(io, "\\begin{widetext}")
        for order in (4, 6)
            data = order_rows[order]
            _write_table(
                io,
                data.hyper_reflective;
                caption="Reflective first-order hyperbolic SAT operator spectral metrics (banded mass) at order=$(order), N=$(points), p=$(p), R=$(rtxt) (successful sources only).",
                label="tab:hyperbolic-reflective-banded-order$(order)",
            )
            println(io)
            _write_table(
                io,
                data.hyper_radiative;
                caption="Radiative first-order hyperbolic SAT operator spectral metrics (banded mass) at order=$(order), N=$(points), p=$(p), R=$(rtxt) (successful sources only).",
                label="tab:hyperbolic-radiative-banded-order$(order)",
            )
            if order != 6
                println(io)
            end
            println(io)
        end
        println(io, "\\end{widetext}")
    end
end

function write_laplacian_divg_tables(path::AbstractString, order_rows::Dict{Int,Any};
        points::Int,
        p::Int,
        R::Rational{BigInt})
    mkpath(dirname(path))
    rtxt = @sprintf("%.1f", Float64(R))
    open(path, "w") do io
        println(io, "\\begin{widetext}")
        for order in (4, 6)
            data = order_rows[order]
            _write_table(
                io,
                data.laplacian;
                caption="Banded-mass symmetric Laplacian operator \$L=\\mathrm{Div}\\,G\$ spectral metrics at order=$(order), N=$(points), p=$(p), R=$(rtxt) (successful sources only).",
                label="tab:laplacian-divg-banded-order$(order)",
            )
            if order != 6
                println(io)
                println(io)
            else
                println(io)
            end
        end
        println(io, "\\end{widetext}")
    end
end

function generate_banded_tables(;
        points::Int=31,
        p::Int=2,
        R::Rational{BigInt}=big(30) // big(1),
        tiny_zero_tol::Float64=1e-16,
        mode=SafeMode(),
        out_hyper::AbstractString=joinpath("papers", "hyperbolic_sat_spectrum_tables_banded.tex"),
        out_lap::AbstractString=joinpath("papers", "laplacian_divg_spectrum_tables_banded.tex"),
        export_tables_dir::Union{Nothing, AbstractString}="/home/svretina/PhD/mypapers/Spherical-SBP-Operators-Paper/Tables")
    results_by_order = Dict{Int,Any}()
    for order in (4, 6)
        println("\n=== Computing order ", order, " ===")
        results_by_order[order] = _compute_order_rows(
            order;
            points=points,
            R=R,
            p=p,
            mode=mode,
            tiny_zero_tol=tiny_zero_tol
        )
        data = results_by_order[order]
        @printf("order=%d summary: %d/%d succeeded, %d failed, %d skipped constructors\n",
            order, data.success_count, data.total_count, length(data.failures), length(data.skipped))
    end

    write_hyperbolic_tables(out_hyper, results_by_order; points=points, p=p, R=R)
    write_laplacian_divg_tables(out_lap, results_by_order; points=points, p=p, R=R)
    _export_tables_to_dir([String(out_hyper), String(out_lap)], export_tables_dir)

    println("\nWrote: ", out_hyper)
    println("Wrote: ", out_lap)
    return results_by_order
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_banded_tables()
end
