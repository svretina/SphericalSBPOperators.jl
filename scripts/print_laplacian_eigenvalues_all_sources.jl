using InteractiveUtils: subtypes
using LinearAlgebra: eigen
using Printf: @printf, @sprintf
using SparseArrays: findnz, sparse, spzeros

using SphericalSBPOperators
using SummationByPartsOperators

const _SYMBOL_SOURCE_FALLBACKS = (:central, :upwind, :standard, :convergent)
const _SKIP_SOURCE_TYPES = Set([SourceOfCoefficientsCombination])

@inline _source_label(source) = string(nameof(typeof(source)))

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

function _build_spherical_ops(source;
                              N::Int,
                              order::Int,
                              method::Symbol,
                              p::Int,
                              R::Real,
                              mode)
    N > 1 || throw(ArgumentError("`N` must be > 1."))
    order > 0 || throw(ArgumentError("`order` must be positive."))
    p >= 0 || throw(ArgumentError("`p` must satisfy p >= 0."))

    if method === :diagonal
        return diagonal_spherical_operators(source;
                                   accuracy_order = order,
                                   N = N,
                                   R = R,
                                   p = p,
                                   mode = mode)
    elseif method === :banded
        return non_diagonal_spherical_operators(source;
                                                accuracy_order = order,
                                                N = N,
                                                R = R,
                                                p = p,
                                                mode = mode)
    end

    throw(ArgumentError("`method` must be `:diagonal` or `:banded`; got `$method`."))
end

function _laplacian_eigs(ops)
    D = getproperty(ops, :D)
    G = hasproperty(ops, :Geven) ? getproperty(ops, :Geven) : getproperty(ops, :G)
    vals = eigen(Matrix{Float64}(D * G)).values
    perm = sortperm(real.(vals); rev = true)
    return vals[perm]
end

"""
    first_order_hyperbolic_spatial_operator(ops;
                                            boundary_condition_reflective=:reflecting,
                                            boundary_condition_radiative=:absorbing,
                                            enforce_origin=false)

Build first-order hyperbolic spatial operators using the exact SAT Jacobian path
from the active wave solver (`wave_system_jac!` + `wave_system_jac_prototype`).
This keeps spectrum analysis consistent with `solve_wave_ode`.
"""
function first_order_hyperbolic_spatial_operator(ops;
                                                 boundary_condition_reflective::Symbol = :reflecting,
                                                 boundary_condition_radiative::Symbol = :absorbing,
                                                 enforce_origin::Bool = false)
    hasproperty(ops, :r) || throw(ArgumentError("`ops` must provide field `r`."))
    n = length(getproperty(ops, :r))
    n > 0 || throw(ArgumentError("`ops.r` must be non-empty."))

    A_none = Matrix{Float64}(wave_system_jac_prototype(ops; boundary_condition = :none))
    wave_system_jac!(A_none,
                     zeros(Float64, 2 * n),
                     WaveODEParams(ops; boundary_condition = :none,
                                   enforce_origin = enforce_origin),
                     0.0)

    A_reflective = Matrix{Float64}(wave_system_jac_prototype(ops;
                                                             boundary_condition = boundary_condition_reflective))
    wave_system_jac!(A_reflective,
                     zeros(Float64, 2 * n),
                     WaveODEParams(ops; boundary_condition = boundary_condition_reflective,
                                   enforce_origin = enforce_origin),
                     0.0)

    A_radiative = Matrix{Float64}(wave_system_jac_prototype(ops;
                                                            boundary_condition = boundary_condition_radiative))
    wave_system_jac!(A_radiative,
                     zeros(Float64, 2 * n),
                     WaveODEParams(ops; boundary_condition = boundary_condition_radiative,
                                   enforce_origin = enforce_origin),
                     0.0)

    M_base = sparse(A_none)
    M_reflective = sparse(A_reflective)
    M_radiative = sparse(A_radiative)
    SAT_reflective = M_reflective - M_base
    SAT_radiative = M_radiative - M_base

    return (; M_base, SAT_reflective, SAT_radiative, M_reflective, M_radiative,
            P_N = nothing)
end

"""
    first_order_hyperbolic_spectrum(ops; kwargs...)

Compute eigenvalues of the reflective and radiative first-order hyperbolic
spatial operators with SAT boundary terms, using the same SAT model as
the wave solver.
"""
function first_order_hyperbolic_spectrum(ops;
                                         c::Real = 1.0,
                                         τ1::Real = 1.0,
                                         τ2::Real = 1.0,
                                         τ3::Real = 1.0,
                                         enforce_origin::Bool = false)
    if c != 1.0 || τ1 != 1.0 || τ2 != 1.0 || τ3 != 1.0
        throw(ArgumentError("Wave-solver SAT model does not use custom `(c, τ1, τ2, τ3)` in this script. " *
                            "Use defaults `(1.0, 1.0, 1.0, 1.0)` for consistency with `solve_wave_ode`."))
    end
    mats = first_order_hyperbolic_spatial_operator(ops;
                                                   boundary_condition_reflective = :reflecting,
                                                   boundary_condition_radiative = :absorbing,
                                                   enforce_origin = enforce_origin)
    λ_reflective = eigen(Matrix{Float64}(mats.M_reflective)).values
    λ_radiative = eigen(Matrix{Float64}(mats.M_radiative)).values
    return (;
            reflective = λ_reflective,
            radiative = λ_radiative,
            M_base = mats.M_base,
            M_reflective = mats.M_reflective,
            M_radiative = mats.M_radiative,
            SAT_reflective = mats.SAT_reflective,
            SAT_radiative = mats.SAT_radiative,
            P_N = mats.P_N)
end

"""
    print_first_order_hyperbolic_sat_spectra_all_sources(N, order;
                                                         method=:diagonal,
                                                         p=2, R=1.0, c=1.0,
                                                         τ1=1.0, τ2=1.0, τ3=1.0,
                                                         radiative_tol=1e-12,
                                                         mode=SafeMode(),
                                                         print_skipped=true,
                                                         print_failures=false)

Loop over all SBP sources, construct operators, build the first-order hyperbolic
spatial operator with SAT terms (reflective and radiative), and compute/store
both eigenvalue sets.
"""
function print_first_order_hyperbolic_sat_spectra_all_sources(N::Integer,
                                                              order::Integer;
                                                              method::Symbol = :diagonal,
                                                              p::Int = 2,
                                                              R::Real = 1.0,
                                                              c::Real = 1.0,
                                                              τ1::Real = 1.0,
                                                              τ2::Real = 1.0,
                                                              τ3::Real = 1.0,
                                                              radiative_tol::Real = 1e-12,
                                                              enforce_radiative_dissipative::Bool = false,
                                                              mode = SafeMode(),
                                                              print_skipped::Bool = true,
                                                              print_failures::Bool = false)
    Nint = Int(N)
    ord = Int(order)

    gathered = collect_sbp_sources()
    sources = gathered.sources
    skipped_ctor = gathered.skipped

    @printf("Hyperbolic SAT spectra | method=%s | N=%d | order=%d | p=%d | R=%g | c=%g\n",
            string(method), Nint, ord, p, float(R), float(c))
    @printf("Detected source instances: %d\n", length(sources))

    if print_skipped && !isempty(skipped_ctor)
        println("\nSkipped source types during constructor discovery:")
        for row in skipped_ctor
            @printf("  - %s: %s\n", row.type, row.reason)
        end
    end

    results = NamedTuple[]
    failures = NamedTuple[]

    for src in sources
        label = _source_label(src)
        try
            ops = _build_spherical_ops(src;
                                       N = Nint,
                                       order = ord,
                                       method = method,
                                       p = p,
                                       R = R,
                                       mode = mode)

            spec = first_order_hyperbolic_spectrum(ops; c = c, τ1 = τ1, τ2 = τ2, τ3 = τ3)

            λref = spec.reflective
            λrad = spec.radiative
            max_re_ref = maximum(real.(λref))
            max_im_ref = maximum(abs.(imag.(λref)))
            max_re_rad = maximum(real.(λrad))
            max_im_rad = maximum(abs.(imag.(λrad)))
            npos_ref = count(x -> x > 0.0, real.(λref))
            npos_rad = count(x -> x > 0.0, real.(λrad))
            radiative_ok = max_re_rad <= radiative_tol

            @printf("\n%s\n", label)
            @printf("  Reflective: max(Re)=%.6e, max(|Im|)=%.6e, n_pos=%d\n",
                    max_re_ref, max_im_ref, npos_ref)
            @printf("  Radiative : max(Re)=%.6e, max(|Im|)=%.6e, n_pos=%d, dissipative=%s (tol=%.1e)\n",
                    max_re_rad, max_im_rad, npos_rad, string(radiative_ok),
                    float(radiative_tol))

            push!(results,
                  (;
                   source = src,
                   label = label,
                   reflective_eigvals = λref,
                   radiative_eigvals = λrad,
                   max_re_reflective = max_re_ref,
                   max_im_reflective = max_im_ref,
                   n_pos_reflective = npos_ref,
                   max_re_radiative = max_re_rad,
                   max_im_radiative = max_im_rad,
                   n_pos_radiative = npos_rad,
                   radiative_dissipative = radiative_ok,
                   P_N = spec.P_N,
                   M_reflective = spec.M_reflective,
                   M_radiative = spec.M_radiative))
        catch err
            msg = sprint(showerror, err)
            push!(failures, (; source = src, label = label, error = msg))
            if print_failures
                @printf("\n[FAIL] %s\n", label)
                @printf("  FAILED: %s\n", msg)
            end
        end
    end

    @printf("\nCompleted: %d succeeded, %d failed.\n", length(results), length(failures))
    if !isempty(results)
        max_re_rad_all = maximum(getfield.(results, :max_re_radiative))
        max_re_ref_all = maximum(getfield.(results, :max_re_reflective))
        n_viol = count(row -> !row.radiative_dissipative, results)
        @printf("Worst max(Re): reflective=%.6e, radiative=%.6e\n", max_re_ref_all,
                max_re_rad_all)
        @printf("Radiative dissipativity violations (max(Re) > tol): %d/%d (tol=%.1e)\n",
                n_viol, length(results), float(radiative_tol))
        if enforce_radiative_dissipative && n_viol > 0
            throw(ArgumentError("Radiative SAT is not strictly dissipative for all successful sources at this setup."))
        end
    end
    return (results = results, failures = failures, skipped = skipped_ctor)
end

function _spectral_condition_number_nonzero(eigvals::AbstractVector{<:Number};
                                            zero_tol::Union{Nothing, Real} = nothing)
    mags = abs.(eigvals)
    max_abs = isempty(mags) ? 0.0 : maximum(mags)
    tol = isnothing(zero_tol) ? max(1e-12, 1e3 * eps(Float64) * max(1.0, max_abs)) :
          Float64(zero_tol)
    nonzero_mags = mags[mags .> tol]
    zero_count = count(m -> m <= tol, mags)

    if isempty(nonzero_mags)
        return (cond = Inf, max_abs = max_abs, min_nonzero_abs = 0.0,
                zero_count = zero_count, tol = tol)
    end

    min_nonzero_abs = minimum(nonzero_mags)
    cond = max_abs / min_nonzero_abs
    return (; cond, max_abs, min_nonzero_abs, zero_count, tol)
end

@inline function _latex_escape(s::AbstractString)
    out = replace(s, "\\" => raw"\textbackslash{}")
    return replace(out,
                   "_" => raw"\_",
                   "&" => raw"\&",
                   "%" => raw"\%",
                   "#" => raw"\#",
                   "\$" => "\\\$",
                   "{" => raw"\{",
                   "}" => raw"\}")
end

@inline _latex_float(x::Real) = isfinite(x) ? @sprintf("%.6e", x) : raw"\infty"

@inline function _publication_metadata_text(publication_source)
    return sprint() do io
        show(IOContext(io, :compact => false), publication_source)
    end
end

@inline function _normalize_citation_key(label::AbstractString)
    key_seed = replace(String(label),
                       "ä" => "a", "Ä" => "A",
                       "ö" => "o", "Ö" => "O",
                       "å" => "a", "Å" => "A",
                       "ü" => "u", "Ü" => "U",
                       "é" => "e", "É" => "E")
    key = lowercase(replace(key_seed, r"[^A-Za-z0-9]+" => "_"))
    key = replace(key, r"_+" => "_")
    return isempty(key) ? "sbp_source" : strip(key, '_')
end

@inline function _parse_authors_year(line::AbstractString)
    m = match(r"^(.*)\((\d{4})\)\s*$", strip(line))
    m === nothing && return nothing
    raw_authors = strip(m.captures[1])
    author_parts = filter(!isempty, strip.(split(raw_authors, ",")))
    authors = isempty(author_parts) ? raw_authors : join(author_parts, " and ")
    year = parse(Int, m.captures[2])
    return (; authors, year)
end

@inline function _parse_journal_volume_pages(line::AbstractString)
    m = match(r"^(.*)\s+([0-9]+(?:\.[0-9]+)?),\s*pp\.\s*([0-9]+)\s*-\s*([0-9]+)\.?\s*$",
              strip(line))
    m === nothing && return nothing

    journal = strip(m.captures[1])
    volnum = m.captures[2]
    pages = string(m.captures[3], "--", m.captures[4])
    if occursin('.', volnum)
        parts = split(volnum, '.')
        return (; journal, volume = parts[1], number = parts[2], pages)
    end
    return (; journal, volume = volnum, number = nothing, pages)
end

function _fallback_bib_entry(label::AbstractString, publication_text::AbstractString)
    key = _normalize_citation_key(label)
    note = replace(strip(replace(publication_text, '\n' => ' ')),
                   "{" => "(",
                   "}" => ")")
    entry = """
@misc{$key,
  author = {SummationByPartsOperators.jl},
  title = {Reference metadata for $label},
  note = {$note}
}
"""
    return (; key, entry)
end

function _citation_entry(label::String, publication_source)
    publication_text = _publication_metadata_text(publication_source)
    lines = filter(!isempty, strip.(split(publication_text, '\n')))
    if isempty(lines)
        return _fallback_bib_entry(label, publication_text)
    end

    ay_idx = findfirst(line -> !isnothing(_parse_authors_year(line)), lines)
    isnothing(ay_idx) && return _fallback_bib_entry(label, publication_text)

    journal_idx = findfirst(line -> occursin("pp.", line) && !isnothing(match(r"\d", line)),
                            lines[(ay_idx + 1):end])
    if isnothing(journal_idx)
        return _fallback_bib_entry(label, publication_text)
    end
    journal_idx += ay_idx

    ay = _parse_authors_year(lines[ay_idx])
    jvp = _parse_journal_volume_pages(lines[journal_idx])
    if isnothing(ay) || isnothing(jvp)
        return _fallback_bib_entry(label, publication_text)
    end

    key = _normalize_citation_key(label)
    title_block = strip(join(lines[(ay_idx + 1):(journal_idx - 1)], " "))
    title = replace(title_block, r"\.\s*$" => "")
    isempty(title) && return _fallback_bib_entry(label, publication_text)
    io = IOBuffer()
    println(io, "@article{$key,")
    println(io, "  author = {", ay.authors, "},")
    println(io, "  title = {", title, "},")
    println(io, "  journal = {", jvp.journal, "},")
    println(io, "  year = {", ay.year, "},")
    println(io, "  volume = {", jvp.volume, "},")
    if !isnothing(jvp.number)
        println(io, "  number = {", jvp.number, "},")
    end
    println(io, "  pages = {", jvp.pages, "}")
    println(io, "}")
    return (; key, entry = String(take!(io)))
end

function write_laplacian_report_latex(results::AbstractVector;
                                      out_tex::AbstractString = joinpath("papers",
                                                                         "laplacian_eigenvalues_all_sources.tex"),
                                      out_bib::AbstractString = joinpath("papers",
                                                                         "laplacian_eigenvalues_all_sources.bib"),
                                      method::Symbol,
                                      N::Integer,
                                      order::Integer,
                                      p::Integer,
                                      R::Real,
                                      zero_eig_tol::Union{Nothing, Real})
    isempty(results) && throw(ArgumentError("No successful source results to write."))

    mkpath(dirname(out_tex))
    mkpath(dirname(out_bib))

    ordered_rows = sort!(collect(results); by = row -> row.label)
    seen = Set{String}()
    bib_entries = String[]
    table_rows = String[]

    for row in ordered_rows
        cite = _citation_entry(row.label, row.publication_source)
        if !(cite.key in seen)
            push!(seen, cite.key)
            push!(bib_entries, strip(cite.entry))
        end

        src_cell = string(_latex_escape(row.label), " \\cite{", cite.key, "}")
        push!(table_rows,
              string(src_cell, " & ",
                     _latex_float(row.max_abs_re), " & ",
                     _latex_float(row.max_abs_im), " & ",
                     _latex_float(row.spectral_radius), " & ",
                     _latex_float(row.cond_nonzero), " \\\\"))
    end

    open(out_bib, "w") do io
        for entry in bib_entries
            println(io, entry)
            println(io)
        end
    end

    bib_filename = basename(out_bib)
    bib_base = splitext(bib_filename)[1]
    caption = "Laplacian spectral metrics for successful SBP sources (method=$(method), N=$(N), order=$(order), p=$(p), R=$(R)). Conditioning uses all eigenvalues with abs(lambda) != 0 exactly (zero tolerance = $(zero_eig_tol))."

    open(out_tex, "w") do io
        println(io, raw"\documentclass{article}")
        println(io, raw"\usepackage[utf8]{inputenc}")
        println(io, raw"\usepackage{booktabs}")
        println(io, raw"\usepackage{geometry}")
        println(io, raw"\geometry{margin=1in}")
        println(io, raw"\begin{filecontents*}{", bib_filename, "}")
        for entry in bib_entries
            println(io, entry)
            println(io)
        end
        println(io, raw"\end{filecontents*}")
        println(io)
        println(io, raw"\begin{document}")
        println(io, raw"\begin{table}[htbp]")
        println(io, raw"\centering")
        println(io, raw"\small")
        println(io, raw"\setlength{\tabcolsep}{4pt}")
        println(io, raw"\begin{tabular}{p{0.44\linewidth}rrrr}")
        println(io, raw"\toprule")
        println(io,
                "Source & \$\\max |\\Re(\\lambda)|\$ & \$\\max |\\Im(\\lambda)|\$ & \$\\rho(L)\$ & \$\\kappa_{\\neq 0}(L)\$ \\\\")
        println(io, raw"\midrule")
        for row_line in table_rows
            println(io, row_line)
        end
        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
        println(io, raw"\caption{", _latex_escape(caption), "}")
        println(io, raw"\label{tab:laplacian-spectrum-all-sources}")
        println(io, raw"\end{table}")
        println(io)
        println(io, raw"\bibliographystyle{plain}")
        println(io, raw"\bibliography{", bib_base, "}")
        println(io, raw"\end{document}")
    end

    return (out_tex = out_tex, out_bib = out_bib, rows = ordered_rows)
end

function generate_laplacian_report(N::Integer, order::Integer;
                                   method::Symbol = :diagonal,
                                   p::Int = 2,
                                   R::Real = 1.0,
                                   mode = SafeMode(),
                                   print_skipped::Bool = false,
                                   print_failures::Bool = false,
                                   zero_eig_tol::Union{Nothing, Real} = 0.0,
                                   out_tex::AbstractString = joinpath("papers",
                                                                      "laplacian_eigenvalues_all_sources.tex"),
                                   out_bib::AbstractString = joinpath("papers",
                                                                      "laplacian_eigenvalues_all_sources.bib"))
    run = print_laplacian_eigenvalues_all_sources(N, order;
                                                  method = method,
                                                  p = p,
                                                  R = R,
                                                  mode = mode,
                                                  print_skipped = print_skipped,
                                                  print_failures = print_failures,
                                                  zero_eig_tol = zero_eig_tol)

    report = write_laplacian_report_latex(run.results;
                                          out_tex = out_tex,
                                          out_bib = out_bib,
                                          method = method,
                                          N = N,
                                          order = order,
                                          p = p,
                                          R = R,
                                          zero_eig_tol = zero_eig_tol)

    println("\nSaved LaTeX report: ", report.out_tex)
    println("Saved BibTeX file: ", report.out_bib)
    return (results = run.results, failures = run.failures, skipped = run.skipped,
            out_tex = report.out_tex, out_bib = report.out_bib)
end

"""
    print_laplacian_eigenvalues_all_sources(N, order;
                                            method=:diagonal,
                                            p=2,
                                            R=1.0,
                                            mode=SafeMode(),
                                            print_skipped=true,
                                            print_failures=false)

Loop over all concrete `SummationByPartsOperators.SourceOfCoefficients` sources,
construct spherical SBP operators, and print Laplacian eigenvalues `eig(D*G)` for each.

Arguments:
- `N`: resolution parameter forwarded directly to constructors (no `±1` shift)
- `order`: gradient/operator order (for unified `:banded`, supported values are 4, 6)
- `method`: `:diagonal` or `:banded`
"""
function print_laplacian_eigenvalues_all_sources(N::Integer,
                                                 order::Integer;
                                                 method::Symbol = :diagonal,
                                                 p::Int = 2,
                                                 R::Real = 1.0,
                                                 mode = SafeMode(),
                                                 print_skipped::Bool = true,
                                                 print_failures::Bool = false,
                                                 zero_eig_tol::Union{Nothing, Real} = 0.0)
    Nint = Int(N)
    ord = Int(order)

    gathered = collect_sbp_sources()
    sources = gathered.sources
    skipped_ctor = gathered.skipped

    @printf("Method: %s | N=%d | order=%d | p=%d | R=%g\n",
            string(method), Nint, ord, p, float(R))
    @printf("Detected source instances: %d\n", length(sources))

    if print_skipped && !isempty(skipped_ctor)
        println("\nSkipped source types during constructor discovery:")
        for row in skipped_ctor
            @printf("  - %s: %s\n", row.type, row.reason)
        end
    end

    results = NamedTuple[]
    failures = NamedTuple[]

    for src in sources
        label = _source_label(src)
        try
            ops = _build_spherical_ops(src;
                                       N = Nint,
                                       order = ord,
                                       method = method,
                                       p = p,
                                       R = R,
                                       mode = mode)
            publication_source = hasproperty(ops, :source) ? getproperty(ops, :source) : src
            eigvals = _laplacian_eigs(ops)
            cond_data = _spectral_condition_number_nonzero(eigvals; zero_tol = zero_eig_tol)
            max_abs_re = maximum(abs.(real.(eigvals)))
            max_re = maximum(real.(eigvals))
            max_abs_im = maximum(abs.(imag.(eigvals)))
            @printf("\n[%d/%d] %s\n", length(results)+1, length(sources), label)
            @printf("  count=%d, max(|Re(λ)|)=%.6e, max(Re(λ))=%.6e, max(|Im(λ)|)=%.6e\n",
                    length(eigvals), max_abs_re, max_re, max_abs_im)
            @printf("  cond_nonzero=%.6e (max|λ|=%.6e, min_nonzero|λ|=%.6e, zero_count=%d, tol=%.3e)\n",
                    cond_data.cond, cond_data.max_abs, cond_data.min_nonzero_abs,
                    cond_data.zero_count, cond_data.tol)
            push!(results,
                  (;
                   source = src,
                   label = label,
                   publication_source = publication_source,
                   eigvals = eigvals,
                   max_abs_re = max_abs_re,
                   max_re = max_re,
                   max_abs_im = max_abs_im,
                   spectral_radius = cond_data.max_abs,
                   cond_nonzero = cond_data.cond,
                   min_nonzero_abs = cond_data.min_nonzero_abs))
        catch err
            msg = sprint(showerror, err)
            push!(failures, (; source = src, label = label, error = msg))
            if print_failures
                @printf("\n[FAIL] %s\n", label)
                @printf("  FAILED: %s\n", msg)
            end
        end
    end

    @printf("\nCompleted: %d succeeded, %d failed.\n", length(results), length(failures))
    if !isempty(results)
        best_idx = argmin(getfield.(results, :max_abs_re))
        best = results[best_idx]
        println("\nBest source by smallest max(|Re(λ)|):")
        @printf("  %s\n", best.label)
        @printf("  max(|Re(λ)|)=%.6e, max(Re(λ))=%.6e, max(|Im(λ)|)=%.6e\n",
                best.max_abs_re, best.max_re, best.max_abs_im)
    end
    return (results = results, failures = failures, skipped = skipped_ctor)
end

if abspath(PROGRAM_FILE) == @__FILE__
    N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 33
    order = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6
    method = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :diagonal
    p = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 2
    R = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1.0
    write_report = length(ARGS) >= 6 ? parse(Bool, ARGS[6]) : false
    out_tex = length(ARGS) >= 7 ? ARGS[7] :
              joinpath("papers", "laplacian_eigenvalues_all_sources.tex")
    out_bib = length(ARGS) >= 8 ? ARGS[8] :
              joinpath("papers", "laplacian_eigenvalues_all_sources.bib")

    if write_report
        generate_laplacian_report(N, order;
                                  method = method,
                                  p = p,
                                  R = R,
                                  mode = SafeMode(),
                                  zero_eig_tol = 0.0,
                                  out_tex = out_tex,
                                  out_bib = out_bib)
    else
        print_laplacian_eigenvalues_all_sources(N, order;
                                                method = method,
                                                p = p,
                                                R = R,
                                                mode = SafeMode(),
                                                zero_eig_tol = 0.0)
    end
end
