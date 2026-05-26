using InteractiveUtils: subtypes
using GenericSchur
using LinearAlgebra: I, schur
using MultiFloats: Float64x4
using Printf: @printf, @sprintf

if !isdefined(@__MODULE__, :SphericalSBPOperators)
    @eval using SphericalSBPOperators
end
using SummationByPartsOperators

const _SYMBOL_SOURCE_FALLBACKS = (:central, :upwind, :standard, :convergent)
const _SKIP_SOURCE_TYPES = Set([SourceOfCoefficientsCombination])
const _DEFAULT_EXPORT_TABLES_DIR = "/home/svretina/PhD/mypapers/Spherical-SBP-Operators-Paper/Tables"

const _SOURCE_TO_SHORT = Dict("DienerDorbandSchnetterTiglio2007" => "Diener2007",
                              "Mattsson2012" => "Mattsson2012",
                              "Mattsson2014" => "Mattsson2014a",
                              "Mattsson2017" => "Mattsson2017",
                              "MattssonAlmquistCarpenter2014Extended" => "Mattsson2014b",
                              "MattssonNordström2004" => "Mattsson2004a",
                              "MattssonSvärdNordström2004" => "Mattsson2004b",
                              "MattssonSvärdShoeybi2008" => "Mattsson2008",
                              "WilliamsDuru2024" => "Williams2024")

const _SOURCE_TO_BIB = Dict("DienerDorbandSchnetterTiglio2007" => "Diener2007",
                            "Mattsson2012" => "Mattsson2012",
                            "Mattsson2014" => "Mattsson2014a",
                            "Mattsson2017" => "Mattsson2017",
                            "MattssonAlmquistCarpenter2014Extended" => "Mattsson2014b",
                            "MattssonNordström2004" => "Mattsson2004a",
                            "MattssonSvärdNordström2004" => "Mattsson2004b",
                            "MattssonSvärdShoeybi2008" => "Mattsson2008",
                            "WilliamsDuru2024" => "Williams2024")

const _SHORT_SORT_INDEX = Dict("Diener2007" => 1,
                               "Mattsson2012" => 2,
                               "Mattsson2014a" => 3,
                               "Mattsson2017" => 4,
                               "Mattsson2014b" => 5,
                               "Mattsson2004a" => 6,
                               "Mattsson2004b" => 7,
                               "Mattsson2008" => 8,
                               "Williams2024" => 9)

@inline _source_label(source) = string(nameof(typeof(source)))

@inline function _source_ctor_symbol(source)
    T = typeof(source)
    if hasmethod(T, Tuple{})
        return nothing
    elseif hasmethod(T, Tuple{Symbol})
        for sym in _SYMBOL_SOURCE_FALLBACKS
            try
                T(sym)
                return sym
            catch
            end
        end
    end
    return nothing
end

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

function collect_sbp_sources()
    types_all = sort!(collect(subtypes(SourceOfCoefficients)); by = t -> string(t))
    sources = Any[]
    skipped = NamedTuple[]

    for T in types_all
        if isabstracttype(T) || !isconcretetype(T) || (T in _SKIP_SOURCE_TYPES)
            push!(skipped, (; type = string(T), reason = "abstract/unsupported meta-type"))
            continue
        end

        built = _construct_source_instance(T)
        if built.ok
            push!(sources, built.source)
        else
            push!(skipped, (; type = string(T), reason = built.reason))
        end
    end

    return (sources = sources, skipped = skipped)
end

function spectrum_legend_reference_rows()
    grouped = Dict{String, Vector{NamedTuple{(:source_label, :symbol), Tuple{String, Union{Nothing, Symbol}}}}}()

    for source in collect_sbp_sources().sources
        source_label = _source_label(source)
        short_name = get(_SOURCE_TO_SHORT, source_label, nothing)
        isnothing(short_name) && continue
        rows = get!(grouped, short_name, NamedTuple{(:source_label, :symbol), Tuple{String, Union{Nothing, Symbol}}}[])
        push!(rows, (; source_label, symbol = _source_ctor_symbol(source)))
    end

    out = NamedTuple[]
    for short_name in sort!(collect(keys(grouped)); by = key -> get(_SHORT_SORT_INDEX, key, 10_000))
        members = sort!(grouped[short_name]; by = row -> row.source_label)
        push!(out, (; legend_entry = short_name, members = members))
    end
    return out
end

function write_spectrum_legend_reference_table(path::AbstractString;
                                               include_symbol_column::Bool = false)
    mkpath(dirname(path))
    rows = spectrum_legend_reference_rows()

    open(path, "w") do io
        println(io, "\\begin{table}[htbp]")
        println(io, "    \\centering")
        println(io, "    \\small")
        println(io, "    \\setlength{\\tabcolsep}{5pt}")
        if include_symbol_column
            println(io, "    \\begin{tabular}{p{0.20\\linewidth}p{0.52\\linewidth}p{0.18\\linewidth}}")
            println(io, "        \\toprule")
            println(io, "        Legend entry & Source family in \\texttt{SummationByPartsOperators} & Constructor symbol \\\\")
        else
            println(io, "    \\begin{tabular}{p{0.22\\linewidth}p{0.66\\linewidth}}")
            println(io, "        \\toprule")
            println(io, "        Legend entry & Source family in \\texttt{SummationByPartsOperators} \\\\")
        end
        println(io, "        \\midrule")

        for row in rows
            family_text = join(["\\texttt{" * member.source_label * "}" for member in row.members], ", ")
            if include_symbol_column
                symbol_text = join([isnothing(member.symbol) ? "--" : "\\texttt{:" * String(member.symbol) * "}"
                                    for member in row.members], ", ")
                println(io, "        \\texttt{", row.legend_entry, "} & ", family_text, " & ", symbol_text, " \\\\")
            else
                println(io, "        \\texttt{", row.legend_entry, "} & ", family_text, " \\\\")
            end
        end

        println(io, "        \\bottomrule")
        println(io, "    \\end{tabular}")
        if include_symbol_column
            println(io,
                    "    \\caption{Correspondence between spectrum-plot legend entries, the underlying \\texttt{SummationByPartsOperators} source families, and any required constructor symbol. A double dash means the zero-argument constructor is used.}")
            println(io, "    \\label{tab:spectrum-legend-reference-symbols}")
        else
            println(io,
                    "    \\caption{Correspondence between spectrum-plot legend entries and the underlying \\texttt{SummationByPartsOperators} source families.}")
            println(io, "    \\label{tab:spectrum-legend-reference}")
        end
        println(io, "\\end{table}")
    end

    println("Wrote: ", path)
    return path
end

function write_spectrum_shortname_citation_table(path::AbstractString)
    mkpath(dirname(path))

    rows = NamedTuple[]
    for source in collect_sbp_sources().sources
        source_label = _source_label(source)
        short_name = get(_SOURCE_TO_SHORT, source_label, nothing)
        bib_key = get(_SOURCE_TO_BIB, source_label, nothing)
        (isnothing(short_name) || isnothing(bib_key)) && continue
        push!(rows, (; short_name, bib_key, source_label))
    end

    sort!(rows; by = row -> (get(_SHORT_SORT_INDEX, row.short_name, 10_000), row.source_label))

    open(path, "w") do io
        println(io, "\\begin{table}[htbp]")
        println(io, "    \\centering")
        println(io, "    \\small")
        println(io, "    \\setlength{\\tabcolsep}{5pt}")
        println(io, "    \\begin{tabular}{p{0.22\\linewidth}p{0.60\\linewidth}}")
        println(io, "        \\toprule")
        println(io, "        Short name & Citation \\\\")
        println(io, "        \\midrule")
        for row in rows
            println(io, "        \\texttt{", row.short_name, "} & \\cite{", row.bib_key, "} \\\\")
        end
        println(io, "        \\bottomrule")
        println(io, "    \\end{tabular}")
        println(io, "    \\caption{Short spectrum labels and their corresponding bibliography entries.}")
        println(io, "    \\label{tab:spectrum-shortname-citations}")
        println(io, "\\end{table}")
    end

    println("Wrote: ", path)
    return path
end

@inline _as_big_rational(x::Rational{BigInt}) = x
@inline _as_big_rational(x::Integer) = big(x) // 1
@inline _as_big_rational(x::Rational{<:Integer}) = big(numerator(x)) // big(denominator(x))

function _as_big_rational(x::AbstractFloat)
    isfinite(x) ||
        throw(ArgumentError("Cannot convert non-finite floating-point value `$x` to Rational{BigInt}."))
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

@inline function _clean_small(x::Real; tol::Float64 = 1e-16)
    xf = Float64(x)
    return abs(xf) < tol ? 0.0 : xf
end

function _spectral_metrics(vals::AbstractVector{<:Complex}; tiny_zero_tol::Float64 = 1e-16)
    re = Float64.(real.(vals))
    im = Float64.(imag.(vals))
    mags = Float64.(abs.(vals))
    return (max_re = _clean_small(maximum(re); tol = tiny_zero_tol),
            min_re = _clean_small(minimum(re); tol = tiny_zero_tol),
            max_im = _clean_small(maximum(im); tol = tiny_zero_tol),
            min_im = _clean_small(minimum(im); tol = tiny_zero_tol),
            rho = _clean_small(maximum(mags); tol = tiny_zero_tol))
end

function _build_banded_ops(source, order::Int; points::Int, R::Rational{BigInt}, p::Int,
                           mode)
    order in (4, 6) ||
        throw(ArgumentError("Unsupported banded order: $order"))

    if order == 6
        return SphericalSBPOperators.non_diagonal_exp_spherical_operators(source;
                                                                          points = points,
                                                                          R = R,
                                                                          p = p,
                                                                          mode = mode,
                                                                          outer_boundary_closure_help = false,
                                                                          verbose = false)
    end

    return SphericalSBPOperators.non_diagonal_spherical_operators(source;
                                                                  accuracy_order = order,
                                                                  points = points,
                                                                  R = R,
                                                                  p = p,
                                                                  mode = mode,
                                                                  verbose = false)
end

@inline function _normalize_spectrum_boundary_model(boundary_model::Symbol)
    boundary_model in (:sat, :hard) ||
        throw(ArgumentError("Unsupported boundary model `$boundary_model`; use `:sat` or `:hard`."))
    return boundary_model
end

@inline _boundary_model_tag(boundary_model::Symbol) =
    String(_normalize_spectrum_boundary_model(boundary_model))

@inline function _spectrum_bc_operator_kind(bc::Symbol)
    bc === :reflecting && return :dirichlet
    bc === :absorbing && return :absorbing
    throw(ArgumentError("Unsupported spectrum boundary condition `$bc`; use `:reflecting` or `:absorbing`."))
end

function _hard_boundary_projection(ops; bc::Symbol, enforce_origin::Bool = false)
    bc in (:reflecting, :absorbing) ||
        throw(ArgumentError("Unsupported hard boundary condition `$bc`; use `:reflecting` or `:absorbing`."))
    bc_kind = _spectrum_bc_operator_kind(bc)
    n = length(ops.r)
    P = Matrix{Float64}(I, 2 * n, 2 * n)

    if enforce_origin && n >= 1
        P[n + 1, :] .= 0.0
    end

    if bc_kind === :dirichlet
        P[n, :] .= 0.0
    elseif bc_kind === :absorbing
        P[n, :] .= 0.0
        P[n, 2 * n] = -1.0
    else
        P[2 * n, :] .= 0.0
    end

    return P
end

function _assemble_hyperbolic_block(ops; bc::Symbol, boundary_model::Symbol = :sat,
                                    enforce_origin::Bool = false)
    bc in (:reflecting, :absorbing) ||
        throw(ArgumentError("Unsupported boundary condition `$bc`; use `:reflecting` or `:absorbing`."))
    boundary_model = _normalize_spectrum_boundary_model(boundary_model)
    bc_kind = _spectrum_bc_operator_kind(bc)
    n = length(ops.r)
    if boundary_model === :sat
        J = Matrix{Float64}(SphericalSBPOperators.wave_system_jac_prototype(ops;
                                                                            boundary_condition = bc_kind))
        SphericalSBPOperators.wave_system_jac!(J,
                                               zeros(Float64, 2 * n),
                                               SphericalSBPOperators.WaveODEParams(ops;
                                                                                   boundary_condition = bc_kind,
                                                                                   enforce_origin = enforce_origin),
                                               0.0)
        return J
    end

    J = Matrix{Float64}(SphericalSBPOperators.wave_system_jac_prototype(ops;
                                                                        boundary_condition = :none))
    SphericalSBPOperators.wave_system_jac!(J,
                                           zeros(Float64, 2 * n),
                                           SphericalSBPOperators.WaveODEParams(ops;
                                                                               boundary_condition = :none,
                                                                               enforce_origin = false),
                                           0.0)
    P = _hard_boundary_projection(ops; bc = bc, enforce_origin = enforce_origin)
    return P * J * P
end

@inline _fmt_sci(x::Real) = @sprintf("%.6e", Float64(x))

@inline function _source_cell(row)
    if haskey(_SOURCE_TO_SHORT, row.label) && haskey(row, :bib_key)
        return "\\texttt{" * row.short_name * "}\\;\\cite{" * row.bib_key * "}"
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

function _write_table(io, rows::AbstractVector; caption::AbstractString,
                      label::AbstractString)
    println(io, "    \\begin{table}[htbp]")
    println(io, "        \\centering")
    println(io, "        \\small")
    println(io, "        \\setlength{\\tabcolsep}{4pt}")
    println(io, "        \\begin{tabular}{p{0.30\\linewidth}rrrrr}")
    println(io, "            \\toprule")
    println(io,
            "            Source & \$\\max \\Re(\\lambda)\$ & \$\\min \\Re(\\lambda)\$ & \$\\max \\Im(\\lambda)\$ & \$\\min \\Im(\\lambda)\$ & \$\\rho(L)\$ \\\\")
    println(io, "            \\midrule")
    for row in rows
        m = row.metrics
        println(io, "            ", _source_cell(row), " & ", _fmt_sci(m.max_re), " & ",
                _fmt_sci(m.min_re), " & ",
                _fmt_sci(m.max_im), " & ", _fmt_sci(m.min_im), " & ", _fmt_sci(m.rho),
                " \\\\")
    end
    println(io, "            \\bottomrule")
    println(io, "        \\end{tabular}")
    println(io, "        \\caption{", caption, "}")
    println(io, "        \\label{", label, "}")
    println(io, "    \\end{table}")
end

function _sort_rows!(rows::Vector)
    sort!(rows; by = row -> (get(_SHORT_SORT_INDEX, row.short_name, 10_000), row.label))
    return rows
end

function _compute_order_rows(order::Int;
                             points::Int,
                             R::Rational{BigInt},
                             p::Int,
                             mode,
                             tiny_zero_tol::Float64,
                             boundary_model::Symbol = :sat)
    gathered = collect_sbp_sources()
    rows_hyper_ref = NamedTuple[]
    rows_hyper_rad = NamedTuple[]
    rows_laplacian = NamedTuple[]
    failures = NamedTuple[]

    total = length(gathered.sources)
    ok = 0
    for (idx, src) in enumerate(gathered.sources)
        label = _source_label(src)
        short_name = get(_SOURCE_TO_SHORT, label, label)
        bib_key = get(_SOURCE_TO_BIB, label, nothing)
        try
            ops = _build_banded_ops(src, order; points = points, R = R, p = p, mode = mode)

            L_ref = _assemble_hyperbolic_block(ops; bc = :reflecting,
                                               boundary_model = boundary_model)
            L_rad = _assemble_hyperbolic_block(ops; bc = :absorbing,
                                               boundary_model = boundary_model)
            L_lap = Matrix(ops.D * ops.Geven)

            metrics_ref = _spectral_metrics(_high_precision_schur_values(L_ref);
                                            tiny_zero_tol = tiny_zero_tol)
            metrics_rad = _spectral_metrics(_high_precision_schur_values(L_rad);
                                            tiny_zero_tol = tiny_zero_tol)
            metrics_lap = _spectral_metrics(_high_precision_schur_values(L_lap);
                                            tiny_zero_tol = tiny_zero_tol)

            push!(rows_hyper_ref,
                  (; label = label, short_name = short_name, bib_key = bib_key, metrics = metrics_ref))
            push!(rows_hyper_rad,
                  (; label = label, short_name = short_name, bib_key = bib_key, metrics = metrics_rad))
            push!(rows_laplacian,
                  (; label = label, short_name = short_name, bib_key = bib_key, metrics = metrics_lap))
            ok += 1
            @printf("[order=%d] [%d/%d] OK   %s\n", order, idx, total, label)
        catch err
            msg = sprint(showerror, err)
            push!(failures, (; label = label, error = msg))
            @printf("[order=%d] [%d/%d] FAIL %s :: %s\n", order, idx, total, label, msg)
        end
    end

    _sort_rows!(rows_hyper_ref)
    _sort_rows!(rows_hyper_rad)
    _sort_rows!(rows_laplacian)
    return (hyper_reflective = rows_hyper_ref,
            hyper_radiative = rows_hyper_rad,
            laplacian = rows_laplacian,
            failures = failures,
            skipped = gathered.skipped,
            success_count = ok,
            total_count = total)
end

function write_hyperbolic_tables(path::AbstractString, order_rows::Dict{Int, Any};
                                 points::Int,
                                 p::Int,
                                 R::Rational{BigInt})
    mkpath(dirname(path))
    rtxt = @sprintf("%.1f", Float64(R))
    open(path, "w") do io
        println(io, "\\begin{widetext}")
        for order in (4, 6)
            data = order_rows[order]
            _write_table(io,
                         data.hyper_reflective;
                         caption = "Reflective first-order hyperbolic SAT operator spectral metrics (banded mass) at order=$(order), N=$(points), p=$(p), R=$(rtxt) (successful sources only).",
                         label = "tab:hyperbolic-reflective-banded-order$(order)",)
            println(io)
            _write_table(io,
                         data.hyper_radiative;
                         caption = "Radiative first-order hyperbolic SAT operator spectral metrics (banded mass) at order=$(order), N=$(points), p=$(p), R=$(rtxt) (successful sources only).",
                         label = "tab:hyperbolic-radiative-banded-order$(order)",)
            if order != 6
                println(io)
            end
            println(io)
        end
        println(io, "\\end{widetext}")
    end
end

function write_laplacian_divg_tables(path::AbstractString, order_rows::Dict{Int, Any};
                                     points::Int,
                                     p::Int,
                                     R::Rational{BigInt})
    mkpath(dirname(path))
    rtxt = @sprintf("%.1f", Float64(R))
    open(path, "w") do io
        println(io, "\\begin{widetext}")
        for order in (4, 6)
            data = order_rows[order]
            _write_table(io,
                         data.laplacian;
                         caption = "Banded-mass symmetric Laplacian operator \$L=\\mathrm{Div}\\,G\$ spectral metrics at order=$(order), N=$(points), p=$(p), R=$(rtxt) (successful sources only).",
                         label = "tab:laplacian-divg-banded-order$(order)",)
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
                                points::Int = 31,
                                p::Int = 2,
                                R::Rational{BigInt} = big(30) // big(1),
                                tiny_zero_tol::Float64 = 1e-16,
                                mode = SafeMode(),
                                out_hyper::AbstractString = joinpath("tables",
                                                                     "hyperbolic_sat_spectrum_tables_banded.tex"),
                                out_lap::AbstractString = joinpath("tables",
                                                                   "laplacian_divg_spectrum_tables_banded.tex"),
                                export_tables_dir::Union{Nothing, AbstractString} = _DEFAULT_EXPORT_TABLES_DIR)
    results_by_order = Dict{Int, Any}()
    for order in (4, 6)
        println("\n=== Computing order ", order, " ===")
        results_by_order[order] = _compute_order_rows(order;
                                                      points = points,
                                                      R = R,
                                                      p = p,
                                                      mode = mode,
                                                      tiny_zero_tol = tiny_zero_tol)
        data = results_by_order[order]
        @printf("order=%d summary: %d/%d succeeded, %d failed, %d skipped constructors\n",
                order, data.success_count, data.total_count, length(data.failures),
                length(data.skipped))
    end

    write_hyperbolic_tables(out_hyper, results_by_order; points = points, p = p, R = R)
    write_laplacian_divg_tables(out_lap, results_by_order; points = points, p = p, R = R)
    _export_tables_to_dir([String(out_hyper), String(out_lap)], export_tables_dir)

    println("\nWrote: ", out_hyper)
    println("Wrote: ", out_lap)
    return results_by_order
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_banded_tables()
end
