using Printf: @printf, @sprintf

if !isdefined(@__MODULE__, :SphericalSBPOperators)
    @eval using SphericalSBPOperators
end
using SummationByPartsOperators

# Reuse source collection, high-precision Schur pipeline, metrics, and
# generic table writer/export helpers.
include(joinpath(@__DIR__, "generate_banded_spectrum_tables.jl"))

function _build_diagonal_ops(source, order::Int;
                             points::Int,
                             R::Rational{BigInt},
                             p::Int,
                             mode)
    ops = SphericalSBPOperators.diagonal_spherical_operators(source;
                                                    accuracy_order = order,
                                                    N = points,
                                                    R = R,
                                                    p = p,
                                                    mode = mode)
    return (r = ops.r,
            D = ops.D,
            Geven = ops.Geven,
            H = ops.H,
            B = ops.B,
            source = source)
end

function _compute_order_rows_diagonal(order::Int;
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
        short_name = get(_SOURCE_TO_SHORT, label, label)
        bib_key = get(_SOURCE_TO_BIB, label, nothing)
        try
            ops = _build_diagonal_ops(src, order; points = points, R = R, p = p,
                                      mode = mode)

            L_ref = _assemble_hyperbolic_block(ops; bc = :reflecting)
            L_rad = _assemble_hyperbolic_block(ops; bc = :absorbing)
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
            @printf("[diagonal order=%d] [%d/%d] OK   %s\n", order, idx, total, label)
        catch err
            msg = sprint(showerror, err)
            push!(failures, (; label = label, error = msg))
            @printf("[diagonal order=%d] [%d/%d] FAIL %s :: %s\n", order, idx, total, label,
                    msg)
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

function write_hyperbolic_tables_diagonal(path::AbstractString, order_rows::Dict{Int, Any};
                                          orders::Tuple{Vararg{Int}},
                                          points::Int,
                                          p::Int,
                                          R::Rational{BigInt})
    mkpath(dirname(path))
    rtxt = @sprintf("%.1f", Float64(R))
    open(path, "w") do io
        println(io, "\\begin{widetext}")
        for (k, order) in enumerate(orders)
            data = order_rows[order]
            _write_table(io,
                         data.hyper_reflective;
                         caption = "Reflective first-order hyperbolic SAT operator spectral metrics (diagonal mass) at order=$(order), N=$(points), p=$(p), R=$(rtxt) (successful sources only).",
                         label = "tab:hyperbolic-reflective-diagonal-order$(order)",)
            println(io)
            _write_table(io,
                         data.hyper_radiative;
                         caption = "Radiative first-order hyperbolic SAT operator spectral metrics (diagonal mass) at order=$(order), N=$(points), p=$(p), R=$(rtxt) (successful sources only).",
                         label = "tab:hyperbolic-radiative-diagonal-order$(order)",)
            if k != length(orders)
                println(io)
                println(io)
            else
                println(io)
            end
        end
        println(io, "\\end{widetext}")
    end
end

function write_laplacian_divg_tables_diagonal(path::AbstractString,
                                              order_rows::Dict{Int, Any};
                                              orders::Tuple{Vararg{Int}},
                                              points::Int,
                                              p::Int,
                                              R::Rational{BigInt})
    mkpath(dirname(path))
    rtxt = @sprintf("%.1f", Float64(R))
    open(path, "w") do io
        println(io, "\\begin{widetext}")
        for (k, order) in enumerate(orders)
            data = order_rows[order]
            _write_table(io,
                         data.laplacian;
                         caption = "Diagonal-mass symmetric Laplacian operator \$L=\\mathrm{Div}\\,G\$ spectral metrics at order=$(order), N=$(points), p=$(p), R=$(rtxt) (successful sources only).",
                         label = "tab:laplacian-divg-diagonal-order$(order)",)
            if k != length(orders)
                println(io)
                println(io)
            else
                println(io)
            end
        end
        println(io, "\\end{widetext}")
    end
end

function generate_diagonal_tables(;
                                  orders::Tuple{Vararg{Int}} = (4, 6, 8),
                                  points::Int = 31,
                                  p::Int = 2,
                                  R::Rational{BigInt} = big(30) // big(1),
                                  tiny_zero_tol::Float64 = 1e-16,
                                  mode = SafeMode(),
                                  out_hyper::AbstractString = joinpath("tables",
                                                                       "hyperbolic_sat_spectrum_tables_diagonal.tex"),
                                  out_lap::AbstractString = joinpath("tables",
                                                                     "laplacian_divg_spectrum_tables_diagonal.tex"),
                                  export_tables_dir::Union{Nothing, AbstractString} = _DEFAULT_EXPORT_TABLES_DIR)
    results_by_order = Dict{Int, Any}()
    for order in orders
        println("\n=== Computing diagonal order ", order, " ===")
        results_by_order[order] = _compute_order_rows_diagonal(order;
                                                               points = points,
                                                               R = R,
                                                               p = p,
                                                               mode = mode,
                                                               tiny_zero_tol = tiny_zero_tol)
        data = results_by_order[order]
        @printf("diagonal order=%d summary: %d/%d succeeded, %d failed, %d skipped constructors\n",
                order, data.success_count, data.total_count, length(data.failures),
                length(data.skipped))
    end

    write_hyperbolic_tables_diagonal(out_hyper,
                                     results_by_order;
                                     orders = orders,
                                     points = points,
                                     p = p,
                                     R = R)
    write_laplacian_divg_tables_diagonal(out_lap,
                                         results_by_order;
                                         orders = orders,
                                         points = points,
                                         p = p,
                                         R = R)
    _export_tables_to_dir([String(out_hyper), String(out_lap)], export_tables_dir)

    println("\nWrote: ", out_hyper)
    println("Wrote: ", out_lap)
    return results_by_order
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_diagonal_tables()
end
