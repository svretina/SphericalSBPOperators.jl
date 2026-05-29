using Printf: @printf, @sprintf

if !isdefined(@__MODULE__, :SphericalSBPOperators)
    @eval using SphericalSBPOperators
end
using SummationByPartsOperators

# Reuse source collection, high-precision Schur pipeline, metric extraction,
# and generic table writer helpers.
include(joinpath(@__DIR__, "generate_banded_spectrum_tables.jl"))

function _build_staggered_ops(source, order::Int;
                              points::Int,
                              R,
                              p::Int,
                              mode,
                              method::Symbol)
    return SphericalSBPOperators.staggered_spherical_operators(source;
                                                               accuracy_order = order,
                                                               N = points,
                                                               R = R,
                                                               p = p,
                                                               method = method,
                                                               mode = mode)
end

function _staggered_variant_tag(method::Symbol)
    method in (:standard, :naive) ||
        throw(ArgumentError("Unsupported staggered method `$method`. Use `:standard` or `:naive`."))
    return method === :standard ? "staggered" : "staggered_naive"
end

function _staggered_variant_caption(method::Symbol)
    method in (:standard, :naive) ||
        throw(ArgumentError("Unsupported staggered method `$method`. Use `:standard` or `:naive`."))
    return method === :standard ? "staggered" : "staggered, naive"
end

function _is_staggered_unsupported_source_error(err)
    return err isa MethodError &&
           (err.f === SummationByPartsOperators.first_derivative_coefficients ||
            err.f === SummationByPartsOperators.derivative_operator ||
            err.f === SummationByPartsOperators.mass_matrix ||
            err.f === SummationByPartsOperators.grid) ||
           (err isa ArgumentError &&
            startswith(sprint(showerror, err), "ArgumentError: Non-uniform staggered half-grid detected."))
end

function _compute_order_rows_staggered(order::Int;
                                       points::Int,
                                       R,
                                       p::Int,
                                       mode,
                                       method::Symbol,
                                       tiny_zero_tol::Float64)
    gathered = collect_sbp_sources()
    rows_hyper_ref = NamedTuple[]
    rows_hyper_rad = NamedTuple[]
    rows_laplacian = NamedTuple[]
    failures = NamedTuple[]
    skipped = copy(gathered.skipped)

    total = length(gathered.sources)
    ok = 0
    for (idx, src) in enumerate(gathered.sources)
        label = _source_label(src)
        short_name = get(_SOURCE_TO_SHORT, label, label)
        bib_key = get(_SOURCE_TO_BIB, label, nothing)
        try
            ops = _build_staggered_ops(src, order; points = points, R = R, p = p,
                                       mode = mode, method = method)

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
            @printf("[staggered method=%s order=%d] [%d/%d] OK   %s\n", String(method), order,
                    idx, total, label)
        catch err
            if _is_staggered_unsupported_source_error(err)
                reason = sprint(showerror, err)
                push!(skipped, (; type = string(typeof(src)), reason = reason))
                @printf("[staggered method=%s order=%d] [%d/%d] SKIP %s :: %s\n",
                        String(method), order, idx, total, label, reason)
                continue
            end
            msg = sprint(showerror, err)
            push!(failures, (; label = label, error = msg))
            @printf("[staggered method=%s order=%d] [%d/%d] FAIL %s :: %s\n", String(method),
                    order, idx, total, label, msg)
        end
    end

    _sort_rows!(rows_hyper_ref)
    _sort_rows!(rows_hyper_rad)
    _sort_rows!(rows_laplacian)
    return (hyper_reflective = rows_hyper_ref,
            hyper_radiative = rows_hyper_rad,
            laplacian = rows_laplacian,
            failures = failures,
            skipped = skipped,
            success_count = ok,
            total_count = total)
end

function write_hyperbolic_tables_staggered(path::AbstractString, order_rows::Dict{Int, Any};
                                           orders::Tuple{Vararg{Int}},
                                           points::Int,
                                           p::Int,
                                           R,
                                           method::Symbol)
    mkpath(dirname(path))
    rtxt = @sprintf("%.1f", Float64(R))
    variant_tag = _staggered_variant_tag(method)
    variant_caption = _staggered_variant_caption(method)
    open(path, "w") do io
        println(io, "\\begin{widetext}")
        for (k, order) in enumerate(orders)
            data = order_rows[order]
            _write_table(io,
                         data.hyper_reflective;
                         caption = "Reflective first-order hyperbolic SAT operator spectral metrics ($(variant_caption)) at order=$(order), N=$(points), p=$(p), R=$(rtxt) (successful sources only).",
                         label = "tab:hyperbolic-reflective-$(variant_tag)-order$(order)",)
            println(io)
            _write_table(io,
                         data.hyper_radiative;
                         caption = "Radiative first-order hyperbolic SAT operator spectral metrics ($(variant_caption)) at order=$(order), N=$(points), p=$(p), R=$(rtxt) (successful sources only).",
                         label = "tab:hyperbolic-radiative-$(variant_tag)-order$(order)",)
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

function write_laplacian_divg_tables_staggered(path::AbstractString,
                                               order_rows::Dict{Int, Any};
                                               orders::Tuple{Vararg{Int}},
                                               points::Int,
                                               p::Int,
                                               R,
                                               method::Symbol)
    mkpath(dirname(path))
    rtxt = @sprintf("%.1f", Float64(R))
    variant_tag = _staggered_variant_tag(method)
    variant_caption = _staggered_variant_caption(method)
    open(path, "w") do io
        println(io, "\\begin{widetext}")
        for (k, order) in enumerate(orders)
            data = order_rows[order]
            _write_table(io,
                         data.laplacian;
                         caption = "Staggered Laplacian operator \$L=\\mathrm{Div}\\,G\$ spectral metrics ($(variant_caption)) at order=$(order), N=$(points), p=$(p), R=$(rtxt) (successful sources only).",
                         label = "tab:laplacian-divg-$(variant_tag)-order$(order)",)
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

function generate_staggered_tables(;
                                   orders::Tuple{Vararg{Int}} = (4, 6),
                                   points::Int = 31,
                                   p::Int = 2,
                                   R = big(30) // big(1),
                                   tiny_zero_tol::Float64 = 1e-16,
                                   method::Symbol = :standard,
                                   mode = SafeMode(),
                                   out_hyper::Union{Nothing, AbstractString} = nothing,
                                   out_lap::Union{Nothing, AbstractString} = nothing,
                                   export_tables_dir::Union{Nothing, AbstractString} = _DEFAULT_EXPORT_TABLES_DIR)
    Rq = _as_big_rational(R)
    variant_tag = _staggered_variant_tag(method)
    out_hyper_resolved = isnothing(out_hyper) ?
                         joinpath("tables",
                                  "hyperbolic_sat_spectrum_tables_$(variant_tag).tex") :
                         String(out_hyper)
    out_lap_resolved = isnothing(out_lap) ?
                       joinpath("tables",
                                "laplacian_divg_spectrum_tables_$(variant_tag).tex") :
                       String(out_lap)
    results_by_order = Dict{Int, Any}()
    for order in orders
        println("\n=== Computing $(variant_tag) order ", order, " ===")
        results_by_order[order] = _compute_order_rows_staggered(order;
                                                                points = points,
                                                                R = Rq,
                                                                p = p,
                                                                mode = mode,
                                                                method = method,
                                                                tiny_zero_tol = tiny_zero_tol)
        data = results_by_order[order]
        @printf("%s order=%d summary: %d/%d succeeded, %d failed, %d skipped constructors\n",
                variant_tag, order, data.success_count, data.total_count, length(data.failures),
                length(data.skipped))
    end

    write_hyperbolic_tables_staggered(out_hyper_resolved,
                                      results_by_order;
                                      orders = orders,
                                      points = points,
                                      p = p,
                                      R = Rq,
                                      method = method)
    write_laplacian_divg_tables_staggered(out_lap_resolved,
                                          results_by_order;
                                          orders = orders,
                                          points = points,
                                          p = p,
                                          R = Rq,
                                          method = method)
    _export_tables_to_dir([out_hyper_resolved, out_lap_resolved], export_tables_dir)

    println("\nWrote: ", out_hyper_resolved)
    println("Wrote: ", out_lap_resolved)
    return results_by_order
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_staggered_tables()
end
