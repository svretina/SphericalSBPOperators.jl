# Run from the repository environment, for example:
#   julia --project=. scripts/make_paper_plots.jl
#
include(joinpath(@__DIR__, "generate_banded_spectrum_plots.jl"))

using CairoMakie: Auto, Axis, Colorbar, Figure, Label, Legend, axislegend, colgap!,
                  heatmap!, hlines!, lines!, rowgap!, rowsize!, save, scatter!,
                  with_theme, xlims!, ylims!
using JLD2: load
using LaTeXStrings: @L_str, latexstring
using Printf: @printf
if !isdefined(@__MODULE__, :SphericalSBPOperators)
    @eval using SphericalSBPOperators
end
using SummationByPartsOperators: MattssonNordström2004, SafeMode

const PAPER_SIM_DIR = joinpath(@__DIR__, "..", "data", "sims")
const PAPER_PLOTS_DIR = joinpath(@__DIR__, "..", "plots", "paper")
const PAPER_TIME_EVOLUTION_DIR = joinpath(@__DIR__, "..", "plots", "time_evolution")
const PAPER_SPECTRA_DIR = joinpath(@__DIR__, "..", "plots", "spectra")
const PAPER_WAVE_ENERGY_DIR = joinpath(@__DIR__, "..", "plots", "wave", "energy")
const PAPER_WAVE_SNAPSHOT_DIR = joinpath(@__DIR__, "..", "plots", "wave", "snapshots")
const PAPER_WAVE_CONVERGENCE_DIR = joinpath(@__DIR__, "..", "plots", "wave", "convergence")
const PAPER_WAVE_CONVERGENCE_SNAPSHOT_DIR = joinpath(@__DIR__, "..", "plots", "wave",
                                                     "convergence_snapshots")
const PAPER_WAVE_ALL_RESOLUTION_ERROR_SNAPSHOT_DIR = joinpath(@__DIR__, "..", "plots",
                                                              "wave",
                                                              "all_resolution_error_snapshots")
const PAPER_WAVE_ALL_RESOLUTION_ANALYTIC_ERROR_DIR = joinpath(@__DIR__, "..", "plots",
                                                              "wave",
                                                              "all_resolution_analytic_error")
const PAPER_WAVE_ALL_RESOLUTION_ANALYTIC_ERROR_SNAPSHOT_DIR = PAPER_WAVE_SNAPSHOT_DIR
const PAPER_DIVERGENCE_COMPARISON_DIR = joinpath(@__DIR__, "..", "plots", "paper",
                                                 "divergence_comparison")
const PAPER_DIV_ORIGIN_CONVERGENCE_DIR = joinpath(@__DIR__, "..", "plots",
                                                  "div_origin_convergence")
const PAPER_SNAPSHOT_TIMES = (5.0, 15.0)
const PAPER_RESOLUTION_TRIPLET = (:run_h, :run_h2, :run_h4)
const PAPER_SCALED_ERROR_RESOLUTION_TRIPLET = (:run_h4, :run_h8, :run_h16)
const PAPER_SCALED_ERROR_RESOLUTION_TRIPLET_FINE = (:run_h8, :run_h16, :run_h32)
const PAPER_AXIS_LABEL_SIZE = 20
const PAPER_TICK_LABEL_SIZE = 24
const PAPER_COLORBAR_TICK_LABEL_SIZE = 20
const PAPER_PANEL_TITLE_SIZE = 20
const PAPER_FIGURE_TITLE_SIZE = 36
const PAPER_TIME_EVOLUTION_TITLE_SIZE = 28
const PAPER_LEGEND_LABEL_SIZE = 26
const PAPER_DIVERGENCE_COMPARISON_LEGEND_LABEL_SIZE = 24
const PAPER_LEGEND_ROW_GAP = 8
const PAPER_LEGEND_MARGIN = (16, 12, 12, 10)
const PAPER_AXIS_LABEL_PADDING = 10
const PAPER_ENERGY_AXIS_LABEL_SIZE = 34
const PAPER_ENERGY_TICK_LABEL_SIZE = 30
const PAPER_ENERGY_TITLE_SIZE = 34
const PAPER_ENERGY_YLABEL_PADDING = 0
const PAPER_ENERGY_LEGEND_ROW_GAP = 8
const PAPER_ENERGY_LEGEND_COL_GAP = 28
const PAPER_ENERGY_LEGEND_PATCHSIZE = (44, 18)
const PAPER_ENERGY_FIGURE_SIZE = (920, 690)
const PAPER_ENERGY_GRID_COLOR = (:gray65, 0.28)
const PAPER_ENERGY_ZERO_COLOR = (:black, 0.45)

function _paper_family_tag(family::Symbol)
    family === :diagonal && return "diagonal"
    family === :non_diagonal && return "non_diagonal"
    family === :non_diagonal_exp && return "non_diagonal_exp"
    family === :mixed_order_diagonal && return "mixed_order_diagonal"
    family === :staggered && return "staggered"
    throw(ArgumentError("Unsupported simulation family `$family`."))
end

function _paper_boundary_group(boundary_condition::Symbol)
    boundary_condition in (:reflecting, :reflective, :dirichlet) && return :reflective
    return :radiative
end

function _paper_scaled_error_triplet(choice::Symbol)
    choice === :h4_h8_h16 && return PAPER_SCALED_ERROR_RESOLUTION_TRIPLET
    choice === :h8_h16_h32 && return PAPER_SCALED_ERROR_RESOLUTION_TRIPLET_FINE
    throw(ArgumentError("Unknown scaled/convergence resolution triplet `$choice`. Use :h4_h8_h16 or :h8_h16_h32."))
end

function _paper_boundary_label(boundary_group::Symbol)
    boundary_group === :reflective && return "reflective"
    boundary_group === :radiative && return "radiative"
    return string(boundary_group)
end

function _paper_initial_data_stem(initial_data_kind::Symbol)
    if initial_data_kind === :even_gaussian_pi_zero_psi
        return "gundlach_wave"
    elseif initial_data_kind === :even_gaussian_pi
        return "incoming_gaussian_wave"
    elseif initial_data_kind === :characteristic_incoming_bump
        return "characteristic_wave"
    elseif initial_data_kind === :regular_left_moving_spherical_gaussian
        return "regular_left_moving_spherical_gaussian_wave"
    end
    return "$(lowercase(replace(string(initial_data_kind), r"[^A-Za-z0-9]+" => "_")))_wave"
end

function _paper_parse_sim_file(path::AbstractString)
    name = basename(path)

    m = match(r"^(gundlach_wave|incoming_gaussian_wave|characteristic_wave|regular_left_moving_spherical_gaussian_wave)_(\d+)th_order_(mixed_order_diagonal|diagonal|non_diagonal|non_diagonal_exp(?:_no_outer_boundary_help)?|staggered)(?:_(.+))?\.jld2$",
              name)
    if m !== nothing
        wave_stem = m.captures[1]
        suffix = something(m.captures[4], "")
        boundary_tag = ""
        alg_tag = ""
        eltype_tag = ""

        if !isempty(suffix)
            tokens = split(suffix, '_')
            if !isempty(tokens) && occursin(r"^cfl", first(tokens))
                tokens = tokens[2:end]
            end
            boundary_tokens = Set(["dirichlet", "reflecting", "reflective", "absorbing",
                                      "radiative", "none"])

            if length(tokens) >= 3 && first(tokens) in boundary_tokens
                boundary_tag = first(tokens)
                eltype_tag = last(tokens)
                alg_tag = join(tokens[2:(end - 1)], "_")
            elseif length(tokens) >= 2
                eltype_tag = last(tokens)
                alg_tag = join(tokens[1:(end - 1)], "_")
            elseif length(tokens) == 1
                boundary_tag = first(tokens)
            end
        end

        return (ok = true,
                preferred = alg_tag == "tsitpap8" && eltype_tag == "float64",
                wave_stem = wave_stem,
                boundary_tag = boundary_tag,
                alg_tag = alg_tag,
                eltype_tag = eltype_tag,
                order = parse(Int, m.captures[2]),
                family = Symbol(m.captures[3]))
    end

    # Legacy typo retained so old files can still be loaded if newer files are absent.
    m = match(r"^gunclach_wave_runs_(\d+)th_order_(diagonal|non_diagonal)\.jld2$", name)
    if m !== nothing
        return (ok = true,
                preferred = false,
                wave_stem = "gunclach_wave_runs",
                boundary_tag = "",
                alg_tag = "",
                eltype_tag = "",
                order = parse(Int, m.captures[1]),
                family = Symbol(m.captures[2]))
    end

    return (ok = false,
            preferred = false,
            wave_stem = "",
            boundary_tag = "",
            alg_tag = "",
            eltype_tag = "",
            order = 0,
            family = :unknown)
end

function _paper_file_rank(parsed; prefer_legacy::Bool)
    if prefer_legacy
        return parsed.alg_tag == "" && parsed.eltype_tag == "" ? 2 :
               parsed.preferred ? 1 : 0
    end

    stem_rank = parsed.wave_stem == "incoming_gaussian_wave" ? 2 :
                parsed.wave_stem == "gundlach_wave" ? 1 : 0
    boundary_rank = parsed.boundary_tag == "dirichlet" ? 2 :
                    isempty(parsed.boundary_tag) ? 1 : 0
    tag_rank = parsed.preferred ? 2 :
               (parsed.alg_tag == "" && parsed.eltype_tag == "" ? 1 : 0)
    return 100 * stem_rank + 10 * boundary_rank + tag_rank
end

function _paper_sol(simulation)
    return hasproperty(simulation, :sol) ? getproperty(simulation, :sol) : simulation
end

function _paper_grid_spacing(simulation)
    sol = _paper_sol(simulation)
    r = Float64.(sol.r)
    length(r) >= 2 ||
        throw(ArgumentError("Simulation grid must contain at least two points."))

    dr = diff(r)
    h = first(dr)
    atol = 100 * eps(Float64) * max(maximum(abs.(dr)), 1.0)
    maximum(abs.(dr .- h)) <= atol ||
        throw(ArgumentError("Simulation grid is not uniformly spaced."))
    return h
end

function _paper_diagnostics(simulation)
    hasproperty(simulation, :diagnostics) ||
        throw(ArgumentError("Simulation does not contain saved diagnostics."))
    return getproperty(simulation, :diagnostics)
end

function _paper_number_token(x::Real)
    txt = @sprintf("%.12g", Float64(x))
    txt = replace(txt, "." => "p")
    txt = replace(txt, "-" => "m")
    txt = replace(txt, "+" => "")
    return txt
end

function _paper_run_token(entry, resolution::Symbol)
    diag = _paper_metadata_run(entry.metadata, resolution).realized.diagnostics
    N = Int(diag.Nh) - 1
    return "R$(_paper_number_token(diag.domain_right))_p$(diag.p)_N$(N)_dt$(_paper_number_token(diag.dt_effective))_T$(_paper_number_token(diag.T_final))"
end

function _paper_energy_run_token(entry, resolutions)
    first_diag = _paper_metadata_run(entry.metadata,
                                     first(resolutions)).realized.diagnostics
    last_diag = _paper_metadata_run(entry.metadata, last(resolutions)).realized.diagnostics
    N_first = Int(first_diag.Nh) - 1
    N_last = Int(last_diag.Nh) - 1
    return "R$(_paper_number_token(first_diag.domain_right))_p$(first_diag.p)_N$(N_first)-$(N_last)_dt$(_paper_number_token(first_diag.dt_effective))-$(_paper_number_token(last_diag.dt_effective))_T$(_paper_number_token(first_diag.T_final))"
end

function _paper_cfl_token(entry, resolutions)
    diag = _paper_metadata_run(entry.metadata, first(resolutions)).realized.diagnostics
    return "cfl$(_paper_number_token(diag.cfl))"
end

function _paper_case_label(family::Symbol, order::Real)
    family_label = family === :diagonal ? "diagonal" :
                   family === :non_diagonal ? "non-diagonal" :
                   family === :non_diagonal_exp ? "non-diagonal" :
                   family === :mixed_order_diagonal ? "mixed-order diagonal" :
                   string(family)
    return "$(family_label), order $(order)"
end

function _paper_resolution_rank(label::Symbol)
    label === :run_h && return 1
    m = match(r"^run_h(\d+)$", String(label))
    m === nothing && return typemax(Int)
    return parse(Int, m.captures[1])
end

function _paper_resolution_denominator(label::Symbol)
    label === :run_h && return 1
    m = match(r"^run_h(\d+)$", String(label))
    m === nothing &&
        throw(ArgumentError("Unsupported resolution label `$label`; expected `:run_h` or `:run_hN`."))
    return parse(Int, m.captures[1])
end

function _paper_resolution_fraction(label::Symbol)
    denom = _paper_resolution_denominator(label)
    return denom == 1 ? "1" : "1/$(denom)"
end

@inline _paper_resolution_curve_label(label::Symbol) = "h=$(_paper_resolution_fraction(label))"

@inline function _paper_resolution_curve_latex(label::Symbol)
    return latexstring("h = ", _paper_resolution_fraction(label))
end

function _paper_resolution_curve_label_aligned(label::Symbol; max_denom_digits::Integer)
    denom = _paper_resolution_denominator(label)
    fraction = denom == 1 ? "1" : "1/$(denom)"
    width = max(1, Int(max_denom_digits))
    return "h = $(lpad(fraction, width + 2))"
end

function _paper_resolution_color(index::Integer)
    palette = Makie.wong_colors()
    return palette[mod1(index, length(palette))]
end

function _paper_ordinal(order::Integer)
    suffix = order % 100 in (11, 12, 13) ? "th" :
             order % 10 == 1 ? "st" :
             order % 10 == 2 ? "nd" :
             order % 10 == 3 ? "rd" : "th"
    return string(order, suffix)
end

function _paper_operator_title_label(family::Symbol, order::Real)
    order_int = Int(round(Float64(order)))
    family_label = family === :diagonal ? "diagonal" :
                   family === :non_diagonal ? "non-diagonal" :
                   family === :non_diagonal_exp ? "non-diagonal" :
                   family === :mixed_order_diagonal ? "mixed-order diagonal" :
                   family === :staggered ? "staggered" :
                   string(family)
    return "$(_paper_ordinal(order_int)) order $(family_label) operator"
end

function _paper_resolution_scale_power(label::Symbol, order::Real)
    denom = _paper_resolution_denominator(label)
    return denom == 1 ? 0 : Int(round(log2(denom) * Float64(order)))
end

function _paper_resolution_scale_label(label::Symbol, order::Real)
    denom = _paper_resolution_denominator(label)
    denom == 1 && return "scale = 1"
    order_int = Int(round(Float64(order)))
    return "scale = $(denom)^$(order_int)"
end

function _paper_resolution_scale_latex(label::Symbol, order::Real)
    denom = _paper_resolution_denominator(label)
    denom == 1 && return latexstring("\\mathrm{scale} = 1")
    order_int = Int(round(Float64(order)))
    return latexstring("h = ", _paper_resolution_fraction(label),
                       ",\\; \\mathrm{scale} = ", string(denom), "^{", string(order_int),
                       "}")
end

function _paper_resolution_scale_label_aligned(label::Symbol,
                                               order::Real;
                                               max_denom_digits::Integer)
    denom = _paper_resolution_denominator(label)
    order_int = Int(round(Float64(order)))
    fraction = denom == 1 ? "1" : "1/$(denom)"
    scale_value = denom == 1 ? "1" : "$(denom)^$(order_int)"
    width = max(1, Int(max_denom_digits))
    return "h = $(lpad(fraction, width + 2)), scale = $(lpad(scale_value, width + 1 + length(string(order_int))))"
end

function _paper_initial_data_summary(entry)
    if hasproperty(entry.metadata, :shared) &&
       hasproperty(entry.metadata.shared, :initial_data) &&
       hasproperty(entry.metadata.shared.initial_data, :summary)
        return entry.metadata.shared.initial_data.summary
    end

    run_labels = _paper_available_resolutions(entry.data)
    isempty(run_labels) &&
        throw(ArgumentError("No saved resolutions were found for this simulation entry."))
    run = _paper_run(entry.data, first(run_labels))
    hasproperty(run, :initial_data) && hasproperty(run.initial_data, :summary) &&
        return run.initial_data.summary

    throw(ArgumentError("Could not recover the initial-data summary from the saved simulation entry."))
end

function _paper_supported_analytic_initial_data(kind::Symbol)
    return kind in (:even_gaussian_pi_zero_psi, :regular_left_moving_spherical_gaussian)
end

function _paper_entry_boundary_condition(entry)
    if hasproperty(entry.metadata, :boundary_condition)
        return entry.metadata.boundary_condition
    elseif hasproperty(entry.metadata, :shared) &&
           hasproperty(entry.metadata.shared, :wave_config) &&
           hasproperty(entry.metadata.shared.wave_config, :boundary_condition)
        return entry.metadata.shared.wave_config.boundary_condition
    end
    return :none
end

function _paper_analytic_reference(entry)
    kind = entry.metadata.initial_data_kind
    _paper_supported_analytic_initial_data(kind) ||
        throw(ArgumentError("No analytic reference is defined for initial-data kind `$kind`."))
    boundary_condition = _paper_entry_boundary_condition(entry)
    summary = _paper_initial_data_summary(entry)
    hasproperty(summary, :amplitude) ||
        throw(ArgumentError("Initial-data summary is missing `amplitude`."))
    hasproperty(summary, :d) || throw(ArgumentError("Initial-data summary is missing `d`."))

    if kind === :even_gaussian_pi_zero_psi
        hasproperty(summary, :r0) ||
            throw(ArgumentError("Initial-data summary is missing `r0`."))
        return (kind = kind,
                amplitude = Float64(summary.amplitude),
                r0 = Float64(summary.r0),
                d = Float64(summary.d),
                boundary_condition = boundary_condition)
    end

    hasproperty(summary, :R) || throw(ArgumentError("Initial-data summary is missing `R`."))
    rc = hasproperty(summary, :rc) ? Float64(summary.rc) : 0.5 * Float64(summary.R)
    return (kind = kind,
            amplitude = Float64(summary.amplitude),
            R = Float64(summary.R),
            d = Float64(summary.d),
            rc = rc,
            boundary_condition = boundary_condition)
end

function _paper_analytic_solution(time::Real, r, reference::NamedTuple)
    return SphericalSBPOperators.analytic_wave_solution(time, r, reference)
end

function _paper_available_resolutions(data::AbstractDict)
    labels = Symbol[]
    for key in keys(data)
        keystr = String(key)
        startswith(keystr, "run_h") || continue
        push!(labels, Symbol(keystr))
    end
    sort!(labels; by = _paper_resolution_rank)
    return Tuple(labels)
end

function _paper_legacy_metadata(path::AbstractString, data::AbstractDict)
    run_labels = _paper_available_resolutions(data)
    isempty(run_labels) &&
        throw(ArgumentError("Legacy simulation bundle has no `run_h*` entries: $path"))
    base_run = _paper_run(data, first(run_labels))
    base_wave_config = hasproperty(base_run, :wave_config) ?
                       getproperty(base_run, :wave_config) :
                       (boundary_condition = base_run.diagnostics.boundary_condition,
                        initial_data_kind = base_run.initial_data.kind)
    base_operator_config = hasproperty(base_run, :operator_config) ?
                           getproperty(base_run, :operator_config) :
                           (family = base_run.diagnostics.family,
                            accuracy_order = base_run.diagnostics.accuracy_order,
                            p = base_run.diagnostics.p,
                            R = base_run.diagnostics.domain_right)
    boundary_group = _paper_boundary_group(base_wave_config.boundary_condition)
    run_entries = (;
                   (label => begin
                        run = _paper_run(data, label)
                        operator_config = hasproperty(run, :operator_config) ?
                                          getproperty(run, :operator_config) :
                                          (family = run.diagnostics.family,
                                           accuracy_order = run.diagnostics.accuracy_order,
                                           p = run.diagnostics.p,
                                           R = run.diagnostics.domain_right)
                        wave_config = hasproperty(run, :wave_config) ?
                                      getproperty(run, :wave_config) :
                                      (boundary_condition = run.diagnostics.boundary_condition,
                                       initial_data_kind = run.initial_data.kind,
                                       dt = run.diagnostics.dt_effective,
                                       T_final = run.diagnostics.T_final)
                        (resolution_label = label,
                         operator_config = operator_config,
                         wave_config = wave_config,
                         initial_data = run.initial_data,
                         realized = (alg_type = string(typeof(run.alg)),
                                     ops_type = string(typeof(run.ops)),
                                     diagnostics = run.diagnostics))
                    end for label in run_labels)...)

    return (schema_version = 1,
            data_file = basename(path),
            created_at_local = "",
            boundary_group = boundary_group,
            family = base_operator_config.family,
            accuracy_order = base_operator_config.accuracy_order,
            initial_data_kind = base_run.initial_data.kind,
            initial_data_stem = _paper_initial_data_stem(base_run.initial_data.kind),
            resolution_labels = run_labels,
            shared = (operator_config = base_operator_config,
                      wave_config = base_wave_config,
                      initial_data = base_run.initial_data),
            runs = run_entries)
end

function _paper_bundle_metadata(data::AbstractDict, path::AbstractString)
    return haskey(data, "metadata") ? data["metadata"] : _paper_legacy_metadata(path, data)
end

function _paper_outer_boundary_closure_help(metadata)
    hasproperty(metadata, :shared) || return nothing
    shared = getproperty(metadata, :shared)
    hasproperty(shared, :operator_config) || return nothing
    operator_config = getproperty(shared, :operator_config)
    hasproperty(operator_config, :outer_boundary_closure_help) || return nothing
    return Bool(getproperty(operator_config, :outer_boundary_closure_help))
end

function _paper_case_variant(metadata)
    family = hasproperty(metadata, :family) ? getproperty(metadata, :family) : nothing
    if family === :non_diagonal_exp
        help = _paper_outer_boundary_closure_help(metadata)
        help === false && return :no_outer_boundary_help
        help === true && return :outer_boundary_help
    end
    return :default
end

function _paper_family_file_tag(family::Symbol, metadata)
    if family === :non_diagonal_exp
        return _paper_case_variant(metadata) === :no_outer_boundary_help ?
               "non_diagonal_exp_no_outer_boundary_help" :
               "non_diagonal_exp"
    end
    return _paper_family_tag(family)
end

function _paper_metadata_run(metadata, resolution::Symbol)
    hasproperty(metadata.runs, resolution) ||
        throw(ArgumentError("Metadata does not contain resolution `$resolution`."))
    return getproperty(metadata.runs, resolution)
end

function _paper_case_out_dir(base_dir::AbstractString, boundary_group::Symbol)
    path = joinpath(base_dir, _paper_boundary_label(boundary_group))
    mkpath(path)
    return path
end

function _paper_case_cfl_out_dir(base_dir::AbstractString,
                                 boundary_group::Symbol,
                                 cfl_token::AbstractString)
    path = joinpath(_paper_case_out_dir(base_dir, boundary_group), cfl_token)
    mkpath(path)
    return path
end

function _paper_list_simulation_bundles(; sim_dir::AbstractString = joinpath(PAPER_SIM_DIR,
                                                                             "reflective"))
    isdir(sim_dir) || throw(ArgumentError("Simulation directory does not exist: $sim_dir"))

    bundles = NamedTuple[]
    for (root, _, files) in walkdir(sim_dir)
        for name in sort(files)
            endswith(name, ".jld2") || continue
            path = joinpath(root, name)
            path_norm = replace(normpath(path), '\\' => '/')
            occursin(r"/archive[^/]*/", path_norm) && continue
            data = load(path)
            metadata = _paper_bundle_metadata(data, path)
            push!(bundles, (path = path,
                            data = data,
                            metadata = metadata,
                            boundary_group = metadata.boundary_group,
                            family = metadata.family,
                            order = metadata.accuracy_order,
                            variant = _paper_case_variant(metadata)))
        end
    end

    sort!(bundles; by = entry -> (_paper_boundary_label(entry.boundary_group),
                                  string(entry.family),
                                  string(entry.variant),
                                  entry.order,
                                  basename(entry.path)))
    return bundles
end

function _paper_save_case_figure(fig,
                                 base_dir::AbstractString,
                                 boundary_group::Symbol,
                                 filename::AbstractString)
    case_out_dir = _paper_case_out_dir(base_dir, boundary_group)
    path = joinpath(case_out_dir, filename)
    save(path, fig)

    # Clean up legacy flat exports from before boundary-specific subdirectories
    # were introduced for paper wave plots.
    legacy_path = joinpath(base_dir, filename)
    if legacy_path != path && isfile(legacy_path)
        rm(legacy_path)
    end

    return path
end

function _paper_case_sort_key(entry)
    return (_paper_boundary_label(entry.boundary_group),
            string(entry.family),
            string(entry.variant),
            entry.order)
end

function _paper_time_index(simulation, t_target::Real)
    sol = _paper_sol(simulation)
    t = Float64.(sol.t)
    isempty(t) && throw(ArgumentError("Simulation has no saved time points."))
    tval = Float64(t_target)
    idx = argmin(abs.(t .- tval))
    return idx, t[idx]
end

function _paper_run(data::AbstractDict, resolution::Symbol)
    key = string(resolution)
    haskey(data, key) ||
        throw(ArgumentError("Simulation data does not contain `$key`; available keys are $(sort(collect(keys(data))))."))
    return data[key]
end

function _paper_missing_runs(data::AbstractDict,
                             resolutions::Tuple{Vararg{Symbol}})
    missing = Symbol[]
    @inbounds for resolution in resolutions
        haskey(data, string(resolution)) || push!(missing, resolution)
    end
    return missing
end

@inline function _paper_energy_ticklabel(v::Real; digits::Int = 6)
    txt = @sprintf("%.*e", digits, Float64(v))
    txt = replace(txt, r"(\.\d*?[1-9])0+e" => s"\1e")
    txt = replace(txt, r"\.0+e" => "e")
    return txt
end

@inline function _paper_energy_plain_ticklabel(v::Real; digits::Int = 6)
    txt = @sprintf("%.*f", digits, Float64(v))
    txt = replace(txt, r"(\.\d*?[1-9])0+$" => s"\1")
    txt = replace(txt, r"\.0+$" => "")
    return txt == "-0" ? "0" : txt
end

@inline function _paper_energy_tickformat(values)
    return [_paper_energy_ticklabel(v) for v in values]
end

@inline function _paper_energy_plain_tickformat(values)
    return [_paper_energy_plain_ticklabel(v) for v in values]
end

function _paper_move_legend_lb!(fig)
    legends = [obj for obj in fig.content if obj isa Legend]
    isempty(legends) && return fig

    for leg in legends
        leg.halign[] = :center
        leg.valign[] = :center
        leg.orientation[] = :horizontal
        leg.nbanks[] = max(1, length(leg.entrygroups[]))
        leg.margin[] = (10, 10, 2, 2)
        leg.padding[] = (2, 2, 2, 2)
        leg.rowgap[] = PAPER_ENERGY_LEGEND_ROW_GAP
        leg.colgap[] = PAPER_ENERGY_LEGEND_COL_GAP
        leg.patchsize[] = PAPER_ENERGY_LEGEND_PATCHSIZE
    end
    return fig
end

function _paper_apply_energy_axis_style!(fig; yscale::Symbol = :identity)
    for obj in fig.content
        obj isa Axis || continue
        obj.xgridvisible[] = true
        obj.ygridvisible[] = true
        obj.xgridcolor[] = PAPER_ENERGY_GRID_COLOR
        obj.ygridcolor[] = PAPER_ENERGY_GRID_COLOR
        obj.xgridwidth[] = 1.0
        obj.ygridwidth[] = 1.0
        obj.xminorgridvisible[] = false
        obj.yminorgridvisible[] = false
        obj.xminorticksvisible[] = true
        obj.yminorticksvisible[] = yscale !== :symlog
        obj.xminortickalign[] = 1.0
        obj.yminortickalign[] = 1.0
        obj.xtickalign[] = 1.0
        obj.ytickalign[] = 1.0
        obj.xticksmirrored[] = true
        obj.yticksmirrored[] = true
        obj.xticksize[] = 8
        obj.yticksize[] = 8
        obj.xminorticksize[] = 4
        obj.yminorticksize[] = 4
        obj.xtickwidth[] = 1.25
        obj.ytickwidth[] = 1.25
        obj.xminortickwidth[] = 1.0
        obj.yminortickwidth[] = 1.0
        obj.spinewidth[] = 1.5
        obj.titlefont[] = :regular
    end
    return fig
end

function _paper_apply_energy_typography!(fig)
    for obj in fig.content
        if obj isa Axis
            obj.xlabelsize[] = PAPER_ENERGY_AXIS_LABEL_SIZE
            obj.ylabelsize[] = PAPER_ENERGY_AXIS_LABEL_SIZE
            obj.xlabelpadding[] = 2
            obj.xticklabelsize[] = PAPER_ENERGY_TICK_LABEL_SIZE
            obj.yticklabelsize[] = PAPER_ENERGY_TICK_LABEL_SIZE
            obj.yticklabelfont[] = "cmr10"
            obj.titlesize[] = PAPER_ENERGY_TITLE_SIZE
            obj.titlefont[] = :regular
            obj.ylabelpadding[] = PAPER_ENERGY_YLABEL_PADDING
        elseif obj isa Legend
            obj.labelsize[] = max(Float64(obj.labelsize[]), PAPER_TICK_LABEL_SIZE)
            obj.rowgap[] = max(Float64(obj.rowgap[]), PAPER_ENERGY_LEGEND_ROW_GAP)
            obj.colgap[] = max(Float64(obj.colgap[]), PAPER_ENERGY_LEGEND_COL_GAP)
        end
    end
    return fig
end

function _paper_energy_plot_values(energy::AbstractVector{<:Real};
                                   normalize::Bool = false,
                                   difference_mode::Symbol = :none)
    if normalize && difference_mode != :none
        throw(ArgumentError("`normalize=true` cannot be combined with `difference_mode=$(difference_mode)`."))
    end

    energy0 = Float64(energy[1])
    values = if normalize
        Float64.(energy ./ energy[1])
    elseif difference_mode == :none
        Float64.(energy)
    elseif difference_mode == :absolute
        Float64.(energy .- energy[1])
    elseif difference_mode == :relative
        abs(energy0) > 0 ||
            throw(ArgumentError("Relative energy difference is undefined because E(0) = 0."))
        Float64.((energy .- energy[1]) ./ energy0)
    else
        throw(ArgumentError("Unsupported `difference_mode=$(difference_mode)`. Use :none, :absolute, or :relative."))
    end

    ylabel = if difference_mode == :absolute
        L"E(t) - E(0)"
    elseif difference_mode == :relative
        L"\frac{E(t) - E(0)}{E(0)}"
    elseif normalize
        L"E(t) / E(0)"
    else
        L"E(t)"
    end

    return values, ylabel
end

function _paper_energy_scale_power(histories::AbstractVector;
                                   normalize::Bool = false,
                                   difference_mode::Symbol = :none)
    (normalize || difference_mode == :none) && return 0

    nonzero = Float64[]
    for history in histories, value in history.energy
        v = abs(Float64(value))
        v > 0 && push!(nonzero, v)
    end
    isempty(nonzero) && return 0

    maxabs = maximum(nonzero)
    exponent = floor(Int, log10(maxabs))
    return abs(exponent) >= 3 ? -exponent : 0
end

function _paper_energy_scaled_ylabel(scale_power::Int;
                                     normalize::Bool = false,
                                     difference_mode::Symbol = :none)
    if difference_mode == :absolute
        return L"E(t) - E(0)"
    elseif difference_mode == :relative
        return L"\frac{E(t) - E(0)}{E(0)}"
    elseif normalize
        return L"E(t) / E(0)"
    end

    return L"E(t)"
end

function _paper_superscript_text(n::Integer)
    chars = Dict('-' => '⁻',
                 '0' => '⁰',
                 '1' => '¹',
                 '2' => '²',
                 '3' => '³',
                 '4' => '⁴',
                 '5' => '⁵',
                 '6' => '⁶',
                 '7' => '⁷',
                 '8' => '⁸',
                 '9' => '⁹')
    return join(get(chars, ch, ch) for ch in string(n))
end

function _paper_energy_scale_annotation(scale_power::Int)
    scale_power == 0 && return nothing
    return Makie.rich("×10",
                      Makie.superscript(string(-scale_power));
                      font = "cmr10")
end

function _paper_symlog_threshold(histories::AbstractVector)
    nonzero = Float64[]
    for history in histories, value in history.energy
        v = abs(Float64(value))
        v > 0 && push!(nonzero, v)
    end
    isempty(nonzero) && return 1.0
    return 10.0^floor(log10(minimum(nonzero)))
end

function _paper_energy_axis_scale(histories::AbstractVector,
                                  yscale::Symbol,
                                  symlog_threshold::Union{Nothing, Real})
    yscale === :identity && return identity
    if yscale === :symlog
        threshold = isnothing(symlog_threshold) ?
                    _paper_symlog_threshold(histories) :
                    Float64(symlog_threshold)
        threshold > 0 ||
            throw(ArgumentError("`symlog_threshold` must be positive, got $(threshold)."))
        return Makie.Symlog10(threshold)
    end
    throw(ArgumentError("Unsupported energy yscale `$(yscale)`. Use :identity or :symlog."))
end

function _paper_energy_symlog_ticks(histories::AbstractVector, threshold::Real)
    ymin = minimum(minimum(history.energy) for history in histories)
    ymax = maximum(maximum(history.energy) for history in histories)
    maxabs = max(abs(ymin), abs(ymax))
    maxabs > 0 || return ([0.0], ["0"])

    lo_exp = floor(Int, log10(Float64(threshold)))
    hi_exp = ceil(Int, log10(maxabs))
    mags = Float64[]
    for exp in lo_exp:hi_exp
        value = 10.0^exp
        value <= maxabs && push!(mags, value)
    end
    push!(mags, Float64(threshold))
    mags = sort(unique(mags))

    ticks = Float64[]
    ymin < 0 && append!(ticks, -reverse(mags))
    ymin <= 0 <= ymax && push!(ticks, 0.0)
    ymax > 0 && append!(ticks, mags)

    labels = _paper_energy_plain_tickformat(ticks)
    return (ticks, labels)
end

function _paper_energy_linear_ticks(histories::AbstractVector)
    maxabs = maximum(maximum(abs, history.energy) for history in histories)
    maxabs > 0 || return [0.0]

    upper = 1.1 * maxabs
    raw_step = upper / 2.5
    exponent = floor(Int, log10(raw_step))
    base = 10.0^exponent
    mantissa = raw_step / base
    nice = mantissa <= 1 ? 1.0 :
           mantissa <= 2 ? 2.0 :
           mantissa <= 2.5 ? 2.5 :
           mantissa <= 5 ? 5.0 : 10.0
    step = nice * base
    n = max(1, ceil(Int, upper / step))
    return collect((-n):n) .* step
end

function _paper_plot_wave_energy_histories(simulations::AbstractVector;
                                           normalize::Bool = false,
                                           difference_mode::Symbol = :none,
                                           resolution_labels::AbstractVector{Symbol},
                                           labels::AbstractVector{<:AbstractString},
                                           title::AbstractString,
                                           legend_position::Symbol = :lb,
                                           yscale::Symbol = :identity,
                                           symlog_threshold::Union{Nothing, Real} = nothing)
    isempty(simulations) &&
        throw(ArgumentError("Need at least one simulation to plot energy histories."))
    length(labels) == length(simulations) == length(resolution_labels) ||
        throw(DimensionMismatch("`labels`, `resolution_labels`, and `simulations` must have the same length."))

    histories = map(simulations) do sim
        sol = _paper_sol(sim)
        yvals, ylabel = _paper_energy_plot_values(Float64.(sol.energy);
                                                  normalize = normalize,
                                                  difference_mode = difference_mode)
        return (t = Float64.(sol.t), energy = yvals, ylabel = ylabel)
    end

    scale_power = _paper_energy_scale_power(histories;
                                            normalize = normalize,
                                            difference_mode = difference_mode)
    scale_factor = 10.0^scale_power
    scaled_histories = scale_power == 0 ? histories :
                       [(; history..., energy = history.energy .* scale_factor)
                        for history in histories]
    ylabel = _paper_energy_scaled_ylabel(scale_power;
                                         normalize = normalize,
                                         difference_mode = difference_mode)
    scale_annotation = _paper_energy_scale_annotation(scale_power)
    plot_title = title == "Wave Energy vs Time" ? "Error in Energy conservation" : title
    resolved_symlog_threshold = yscale === :symlog ?
                                (isnothing(symlog_threshold) ?
                                 _paper_symlog_threshold(scaled_histories) :
                                 Float64(symlog_threshold) * scale_factor) :
                                nothing
    axis_yscale = _paper_energy_axis_scale(scaled_histories, yscale,
                                           resolved_symlog_threshold)
    axis_yticks = yscale === :symlog ?
                  _paper_energy_symlog_ticks(scaled_histories, resolved_symlog_threshold) :
                  (scale_power == 0 ? Makie.automatic :
                   _paper_energy_linear_ticks(scaled_histories))
    axis_ytickformat = yscale === :symlog ? Makie.automatic :
                       (scale_power == 0 ? _paper_energy_tickformat :
                        _paper_energy_plain_tickformat)

    fig = with_theme(SphericalSBPOperators.mytheme_aps()) do
        with_theme(SphericalSBPOperators.mytheme_aps_spectrum(axis_labelsize = PAPER_ENERGY_AXIS_LABEL_SIZE,
                                                              tick_labelsize = PAPER_ENERGY_TICK_LABEL_SIZE,
                                                              title_size = PAPER_ENERGY_TITLE_SIZE,
                                                              legend_labelsize = PAPER_LEGEND_LABEL_SIZE,
                                                              legend_rowgap = PAPER_ENERGY_LEGEND_ROW_GAP,
                                                              legend_colgap = PAPER_ENERGY_LEGEND_COL_GAP,
                                                              patchsize = PAPER_ENERGY_LEGEND_PATCHSIZE,
                                                              legend_padding = (2, 2, 2, 2),
                                                              legend_margin = (10, 10, 2, 2),
                                                              figure_padding = (12, 18, 10, 8),
                                                              gridcolor = PAPER_ENERGY_GRID_COLOR)) do
        fig = Figure(size = PAPER_ENERGY_FIGURE_SIZE)
        ax = Axis(fig[1, 1], xlabel = L"t", ylabel = ylabel, title = plot_title,
                  titlefont = :regular,
                  ytickformat = axis_ytickformat,
                  yscale = axis_yscale,
                  yticks = axis_yticks,
                  xminorgridvisible = false,
                  yminorgridvisible = false,
                  xgridcolor = PAPER_ENERGY_GRID_COLOR,
                  ygridcolor = PAPER_ENERGY_GRID_COLOR)
        for i in eachindex(scaled_histories, labels, resolution_labels)
            history = scaled_histories[i]
            lines!(ax, history.t, history.energy; linewidth = 2.8,
                   color = (_paper_resolution_color(i), 0.9),
                   linestyle = :solid,
                   label = labels[i])
        end
        maxabs = maximum(maximum(abs, history.energy) for history in scaled_histories)
        maxabs > 0 && yscale === :identity && ylims!(ax, -1.1 * maxabs, 1.1 * maxabs)
        if !isnothing(scale_annotation)
            Label(fig[1, 1, Left()], scale_annotation;
                  halign = :right,
                  valign = :top,
                  font = "cmr10",
                  fontsize = PAPER_ENERGY_TICK_LABEL_SIZE,
                  padding = (0, 2, 0, 0))
        end
        hlines!(ax, [0.0]; color = PAPER_ENERGY_ZERO_COLOR, linewidth = 1.2)
        xlims!(ax, 0.0, maximum(last(history.t) for history in scaled_histories) + 0.2)
        Legend(fig[2, 1], ax;
               orientation = :horizontal,
               framevisible = false,
               labelsize = PAPER_LEGEND_LABEL_SIZE,
               nbanks = max(1, length(labels)),
               patchsize = PAPER_ENERGY_LEGEND_PATCHSIZE,
               rowgap = PAPER_ENERGY_LEGEND_ROW_GAP,
               colgap = PAPER_ENERGY_LEGEND_COL_GAP,
               margin = (10, 10, 2, 2),
               padding = (2, 2, 2, 2))
        rowgap!(fig.layout, 8)
        rowsize!(fig.layout, 2, Auto(0.14))
        fig
        end
    end

    return (fig = fig,
            histories = scaled_histories,
            scale_power = scale_power,
            scale_factor = scale_factor)
end

function _paper_apply_typography!(fig)
    for obj in fig.content
        if obj isa Axis
            obj.xlabelsize[] = PAPER_AXIS_LABEL_SIZE
            obj.ylabelsize[] = PAPER_AXIS_LABEL_SIZE
            obj.xlabelpadding[] = max(Float64(obj.xlabelpadding[]), Float64(PAPER_AXIS_LABEL_PADDING))
            obj.ylabelpadding[] = max(Float64(obj.ylabelpadding[]), Float64(PAPER_AXIS_LABEL_PADDING))
            obj.xticklabelsize[] = PAPER_TICK_LABEL_SIZE
            obj.yticklabelsize[] = PAPER_TICK_LABEL_SIZE
            obj.titlesize[] = PAPER_PANEL_TITLE_SIZE
        elseif obj isa Legend
            obj.labelsize[] = PAPER_LEGEND_LABEL_SIZE
            obj.rowgap[] = PAPER_LEGEND_ROW_GAP
        elseif obj isa Label
            current = obj.fontsize[]
            obj.fontsize[] = current isa Number ? Float64(current) : PAPER_FIGURE_TITLE_SIZE
        end
    end
    return fig
end

function _paper_apply_analytic_error_typography!(fig)
    axis_labelsize = Float64(PAPER_AXIS_LABEL_SIZE + 10)
    axis_labelpadding = Float64(PAPER_AXIS_LABEL_PADDING + 5)
    ylabelpadding = Float64(PAPER_AXIS_LABEL_PADDING + 24)
    tick_labelsize = Float64(PAPER_TICK_LABEL_SIZE + 1)
    yticklabelspace = 40.0
    yticklabelpad = 8.0

    for obj in fig.content
        if obj isa Axis
            obj.xlabelsize[] = axis_labelsize
            obj.ylabelsize[] = axis_labelsize
            obj.xlabelpadding[] = axis_labelpadding
            obj.ylabelpadding[] = ylabelpadding
            obj.xticklabelsize[] = tick_labelsize
            obj.yticklabelsize[] = tick_labelsize
            obj.yticklabelspace[] = yticklabelspace
            obj.yticklabelpad[] = yticklabelpad
        end
    end
    return fig
end

function _paper_snapshot_overlay_series(simulation, t_target::Real)
    sol = _paper_sol(simulation)
    idx, t_saved = _paper_time_index(simulation, t_target)
    r = Float64.(sol.r)
    rpi = r .* Float64.(sol.Π[:, idx])
    rpsi = r .* Float64.(sol.Ψ[:, idx])
    return (r = r, rpi = rpi, rpsi = rpsi, t = t_saved, time_index = idx)
end

function _paper_cell_edges(v::AbstractVector)
    n = length(v)
    n >= 2 || throw(ArgumentError("Need at least two points to build cell edges."))
    edges = Vector{Float64}(undef, n + 1)
    for i in 2:n
        edges[i] = 0.5 * (Float64(v[i - 1]) + Float64(v[i]))
    end
    edges[1] = Float64(v[1]) - (edges[2] - Float64(v[1]))
    edges[end] = Float64(v[end]) + (Float64(v[end]) - edges[end - 1])
    return edges
end

function _paper_colorbar_ticks(lims; nticks::Int = 5)
    nticks >= 5 || throw(ArgumentError("`nticks` must be at least 5."))
    isodd(nticks) || throw(ArgumentError("`nticks` must be odd so the ticks can stay centered on zero."))
    lo, hi = Float64(lims[1]), Float64(lims[2])
    ticks = collect(range(lo, hi; length = nticks))
    labels = [_paper_energy_plain_ticklabel(tick; digits = 2) for tick in ticks]
    return (ticks, labels)
end

function _paper_nice_ceil(x::Real)
    x > 0 || return 1.0
    exponent = floor(Int, log10(Float64(x)))
    base = 10.0 ^ exponent
    mantissa = Float64(x) / base
    nice = mantissa <= 1 ? 1.0 :
           mantissa <= 1.2 ? 1.2 :
           mantissa <= 1.5 ? 1.5 :
           mantissa <= 2 ? 2.0 :
           mantissa <= 2.5 ? 2.5 :
           mantissa <= 3 ? 3.0 :
           mantissa <= 4 ? 4.0 :
           mantissa <= 5 ? 5.0 :
           mantissa <= 6 ? 6.0 :
           mantissa <= 8 ? 8.0 : 10.0
    return nice * base
end

function _paper_symmetric_limits(values)
    vf = Float64.(values)
    maxabs = max(abs(minimum(vf)), abs(maximum(vf)))
    return (-maxabs, maxabs)
end

function _paper_symmetric_colorbar_spec(values; nticks::Int = 5)
    nticks >= 5 || throw(ArgumentError("`nticks` must be at least 5."))
    isodd(nticks) || throw(ArgumentError("`nticks` must be odd so the ticks can stay centered on zero."))

    lo, hi = _paper_symmetric_limits(values)
    maxabs = max(abs(lo), abs(hi))
    nhalf = (nticks - 1) ÷ 2
    upper = _paper_nice_ceil(maxabs)
    lims = (-upper, upper)
    ticks = _paper_colorbar_ticks(lims; nticks = nticks)
    return (lims = lims, ticks = ticks)
end

function _paper_time_evolution_figure(simulation;
                                      title::AbstractString,
                                      interpolate::Bool = true)
    sol = _paper_sol(simulation)
    t = Float64.(sol.t)
    r = Float64.(sol.r)
    length(t) >= 2 ||
        throw(ArgumentError("Simulation must contain at least two saved time points."))
    length(r) >= 2 ||
        throw(ArgumentError("Simulation grid must contain at least two points."))

    pi_history = Float64.(sol.Π)
    psi_history = Float64.(sol.Ψ)
    redges = _paper_cell_edges(r)
    tedges = _paper_cell_edges(t)
    pi_spec = _paper_symmetric_colorbar_spec(pi_history; nticks = 5)
    psi_spec = _paper_symmetric_colorbar_spec(psi_history; nticks = 5)
    pi_limits = pi_spec.lims
    psi_limits = psi_spec.lims
    pi_ticks = pi_spec.ticks
    psi_ticks = psi_spec.ticks
    aspect_ratio = Float64(last(r)) / Float64(last(t))

    fig = with_theme(SphericalSBPOperators.mytheme_aps()) do
        fig = Figure(size = (1200, 430), figure_padding = (4, 24, 4, 4))
        rowgap!(fig.layout, 2)
        colgap!(fig.layout, 8)
        Label(fig[0, 1:2], title; fontsize = PAPER_TIME_EVOLUTION_TITLE_SIZE, font = :bold,
              padding = (0, 0, 0, 0))

        ax_pi = Axis(fig[1, 1];
                     xlabel = L"r",
                     ylabel = "t",
                     title = latexstring("\\Pi(t,r)"),
                     aspect = aspect_ratio)
        ax_psi = Axis(fig[1, 2];
                      xlabel = L"r",
                      ylabel = "t",
                      title = latexstring("\\Psi(t,r)"),
                      aspect = aspect_ratio)

        hm_pi = heatmap!(ax_pi, redges, tedges, pi_history;
                         colormap = :balance,
                         colorrange = pi_limits,
                         interpolate = interpolate,
                         rasterize = 5)
        hm_psi = heatmap!(ax_psi, redges, tedges, psi_history;
                          colormap = :balance,
                          colorrange = psi_limits,
                          interpolate = interpolate,
                          rasterize = 5)

        Colorbar(fig[2, 1], hm_pi;
                 vertical = false,
                 label = "",
                 height = 10,
                 ticks = pi_ticks,
                 ticklabelsize = PAPER_COLORBAR_TICK_LABEL_SIZE)
        Colorbar(fig[2, 2], hm_psi;
                 vertical = false,
                 label = "",
                 height = 10,
                 ticks = psi_ticks,
                 ticklabelsize = PAPER_COLORBAR_TICK_LABEL_SIZE)
        rowsize!(fig.layout, 2, Auto(0.06))
        fig
    end

    return fig
end

function _paper_convergence_snapshot_figure(sim1,
                                            sim2,
                                            sim3;
                                            family::Symbol,
                                            order = nothing,
                                            time::Real,
                                            h0::Real,
                                            resolution_labels::NTuple{3, AbstractString})
    _limits_without_origin(values, r) = length(values) > 1 && !isempty(r) &&
                                        iszero(first(r)) ?
                                        SphericalSBPOperators._finite_limits_with_padding(@view values[2:end]) :
                                        SphericalSBPOperators._finite_limits_with_padding(values)
    function _limits_above_radius(values, r, rmin::Real)
        kept = [Float64(values[i])
                for i in eachindex(values, r) if Float64(r[i]) >= Float64(rmin)]
        return isempty(kept) ? SphericalSBPOperators._finite_limits_with_padding(values) :
               SphericalSBPOperators._finite_limits_with_padding(kept)
    end

    snap1 = _paper_snapshot_overlay_series(sim1, time)
    snap2 = _paper_snapshot_overlay_series(sim2, snap1.t)
    snap3 = _paper_snapshot_overlay_series(sim3, snap1.t)

    pi_data = SphericalSBPOperators._gundlach_pointwise_error_history(sim1, sim2, sim3;
                                                                      field = :Π,
                                                                      expected_order = order,
                                                                      h0 = h0)
    psi_data = SphericalSBPOperators._gundlach_pointwise_error_history(sim1, sim2, sim3;
                                                                       field = :Ψ,
                                                                       expected_order = order,
                                                                       h0 = h0)
    time_index = SphericalSBPOperators._nearest_common_time_index(pi_data.t, snap1.t)
    effective_order = pi_data.expected_order
    r_err = Float64.(pi_data.r)
    # radial_weight = r_err .^ (p / 2)
    pi_err_h_h2 = pi_data.err_h_h2[:, time_index]
    pi_err_h2_h4 = pi_data.err_h2_h4[:, time_index]
    psi_err_h_h2 = psi_data.err_h_h2[:, time_index]
    psi_err_h2_h4 = psi_data.err_h2_h4[:, time_index]

    top_pi_limits = SphericalSBPOperators._finite_limits_with_padding(vcat(snap1.rpi,
                                                                           snap2.rpi,
                                                                           snap3.rpi))
    top_psi_limits = SphericalSBPOperators._finite_limits_with_padding(vcat(snap1.rpsi,
                                                                            snap2.rpsi,
                                                                            snap3.rpsi))
    bottom_pi_limits = _limits_above_radius(vcat(pi_err_h_h2, pi_err_h2_h4),
                                            vcat(r_err, r_err),
                                            3.0)
    bottom_psi_limits = _limits_without_origin(vcat(psi_err_h_h2, psi_err_h2_h4),
                                               vcat(r_err, r_err))

    title = "$(_paper_case_label(family, effective_order)), t = $(@sprintf("%.6g", snap1.t))"
    fig = with_theme(SphericalSBPOperators.mytheme_aps()) do
        fig = Figure(size = (1080, 860), figure_padding = (28, 18, 12, 12))
        Label(fig[0, 1:2], title; fontsize = PAPER_FIGURE_TITLE_SIZE, font = :bold)

        ax_rpi = Axis(fig[1, 1],
                      xlabel = L"r",
                      ylabel = L"r \Pi(t,r)",
                      xlabelsize = PAPER_AXIS_LABEL_SIZE,
                      ylabelsize = PAPER_AXIS_LABEL_SIZE,
                      xticklabelsize = PAPER_TICK_LABEL_SIZE,
                      yticklabelsize = PAPER_TICK_LABEL_SIZE)
        ax_rpsi = Axis(fig[1, 2],
                       xlabel = L"r",
                       ylabel = L"r \Psi(t,r)",
                       xlabelsize = PAPER_AXIS_LABEL_SIZE,
                       ylabelsize = PAPER_AXIS_LABEL_SIZE,
                       xticklabelsize = PAPER_TICK_LABEL_SIZE,
                       yticklabelsize = PAPER_TICK_LABEL_SIZE)
        ax_epi = Axis(fig[2, 1],
                      xlabel = L"r",
                      ylabel = latexstring("e_{\\Pi}(t,r)"),
                      xlabelsize = PAPER_AXIS_LABEL_SIZE,
                      ylabelsize = PAPER_AXIS_LABEL_SIZE,
                      xticklabelsize = PAPER_TICK_LABEL_SIZE,
                      yticklabelsize = PAPER_TICK_LABEL_SIZE)
        ax_epsi = Axis(fig[2, 2],
                       xlabel = L"r",
                       ylabel = latexstring("e_{\\Psi}(t,r)"),
                       xlabelsize = PAPER_AXIS_LABEL_SIZE,
                       ylabelsize = PAPER_AXIS_LABEL_SIZE,
                       xticklabelsize = PAPER_TICK_LABEL_SIZE,
                       yticklabelsize = PAPER_TICK_LABEL_SIZE)

        for (ax, limits) in ((ax_rpi, top_pi_limits),
                             (ax_rpsi, top_psi_limits),
                             (ax_epi, bottom_pi_limits),
                             (ax_epsi, bottom_psi_limits))
            hlines!(ax, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
            ylims!(ax, limits...)
        end

        xlims!(ax_rpi, minimum(snap3.r), maximum(snap3.r))
        xlims!(ax_rpsi, minimum(snap3.r), maximum(snap3.r))
        xlims!(ax_epi, minimum(r_err), maximum(r_err))
        xlims!(ax_epsi, minimum(r_err), maximum(r_err))

        lines!(ax_rpi, snap1.r, snap1.rpi; color = :royalblue4, linewidth = 2.5,
               label = resolution_labels[1])
        lines!(ax_rpi, snap2.r, snap2.rpi; color = :darkorange3, linewidth = 2.5,
               linestyle = :dash, label = resolution_labels[2])
        lines!(ax_rpi, snap3.r, snap3.rpi; color = :seagreen4, linewidth = 2.5,
               linestyle = :dot, label = resolution_labels[3])
        axislegend(ax_rpi; position = :rb, labelsize = PAPER_LEGEND_LABEL_SIZE,
                   framevisible = false)

        lines!(ax_rpsi, snap1.r, snap1.rpsi; color = :royalblue4, linewidth = 2.5,
               label = resolution_labels[1])
        lines!(ax_rpsi, snap2.r, snap2.rpsi; color = :darkorange3, linewidth = 2.5,
               linestyle = :dash, label = resolution_labels[2])
        lines!(ax_rpsi, snap3.r, snap3.rpsi; color = :seagreen4, linewidth = 2.5,
               linestyle = :dot, label = resolution_labels[3])
        axislegend(ax_rpsi; position = :rb, labelsize = PAPER_LEGEND_LABEL_SIZE,
                   framevisible = false)

        lines!(ax_epi, r_err, pi_err_h_h2; color = :royalblue4, linewidth = 2.5,
               label = "low-mid error")
        scatter!(ax_epi, r_err, pi_err_h_h2; color = :royalblue4, markersize = 7)
        lines!(ax_epi, r_err, pi_err_h2_h4; color = :darkorange3, linewidth = 2.5,
               linestyle = :dash, label = "mid-high error")
        scatter!(ax_epi, r_err, pi_err_h2_h4; color = :darkorange3, markersize = 7,
                 marker = :rect)
        axislegend(ax_epi; position = :rb, labelsize = PAPER_LEGEND_LABEL_SIZE,
                   framevisible = false)

        lines!(ax_epsi, r_err, psi_err_h_h2; color = :seagreen4, linewidth = 2.5,
               label = "low-mid error")
        scatter!(ax_epsi, r_err, psi_err_h_h2; color = :seagreen4, markersize = 7)
        lines!(ax_epsi, r_err, psi_err_h2_h4; color = :firebrick, linewidth = 2.5,
               linestyle = :dash, label = "mid-high error")
        scatter!(ax_epsi, r_err, psi_err_h2_h4; color = :firebrick, markersize = 7,
                 marker = :rect)
        axislegend(ax_epsi; position = :rb, labelsize = PAPER_LEGEND_LABEL_SIZE,
                   framevisible = false)

        fig
    end

    return (fig = fig, t = snap1.t, time_index = time_index)
end

function _paper_pointwise_difference_snapshot(sim_coarse,
                                              sim_fine;
                                              field::Symbol,
                                              time::Real,
                                              expected_order::Real,
                                              h0::Real)
    sol_coarse = _paper_sol(sim_coarse)
    sol_fine = _paper_sol(sim_fine)
    idx_coarse, t_coarse = _paper_time_index(sim_coarse, time)
    idx_fine, t_fine = _paper_time_index(sim_fine, t_coarse)

    r_coarse = Float64.(sol_coarse.r)
    history_coarse = SphericalSBPOperators._wave_field_history(sol_coarse, field)
    history_fine = SphericalSBPOperators._wave_field_history(sol_fine, field)
    u_coarse = Float64.(history_coarse[:, idx_coarse])
    u_fine = SphericalSBPOperators._align_reference_to_grid(r_coarse,
                                                            sol_fine.r,
                                                            history_fine[:, idx_fine])
    err = u_coarse .- u_fine
    h = _paper_grid_spacing(sim_coarse)
    scale = (Float64(h) / Float64(h0))^(-Float64(expected_order))

    return (r = r_coarse,
            err = err,
            scaled_err = scale .* err,
            h = Float64(h),
            scale = scale,
            t = t_coarse,
            reference_time = t_fine,
            field = SphericalSBPOperators._canonical_wave_field(field))
end

function _paper_all_resolution_error_snapshot_figure(simulations::AbstractVector,
                                                     resolution_labels::AbstractVector;
                                                     family::Symbol,
                                                     order::Real,
                                                     time::Real,
                                                     h0::Real)
    length(simulations) == length(resolution_labels) ||
        throw(DimensionMismatch("`simulations` and `resolution_labels` must have the same length."))
    length(simulations) >= 2 ||
        throw(ArgumentError("Need at least two resolutions to build error snapshots."))

    snaps = [_paper_snapshot_overlay_series(sim, time) for sim in simulations]
    snap_time = snaps[1].t
    pi_pairs = [_paper_pointwise_difference_snapshot(simulations[i], simulations[i + 1];
                                                     field = :Π,
                                                     time = snap_time,
                                                     expected_order = order,
                                                     h0 = h0)
                for i in 1:(length(simulations) - 1)]
    psi_pairs = [_paper_pointwise_difference_snapshot(simulations[i], simulations[i + 1];
                                                      field = :Ψ,
                                                      time = snap_time,
                                                      expected_order = order,
                                                      h0 = h0)
                 for i in 1:(length(simulations) - 1)]

    top_pi_limits = SphericalSBPOperators._finite_limits_with_padding(reduce(vcat,
                                                                             (snap.rpi
                                                                              for snap in snaps)))
    top_psi_limits = SphericalSBPOperators._finite_limits_with_padding(reduce(vcat,
                                                                              (snap.rpsi
                                                                               for snap in snaps)))
    bottom_pi_limits = SphericalSBPOperators._finite_limits_with_padding(reduce(vcat,
                                                                                (pair.scaled_err
                                                                                 for pair in pi_pairs)))
    bottom_psi_limits = SphericalSBPOperators._finite_limits_with_padding(reduce(vcat,
                                                                                 (pair.scaled_err
                                                                                  for pair in psi_pairs)))

    title = "$(_paper_operator_title_label(family, order)), t = $(@sprintf("%.6g", snap_time))"
    linestyles = [:solid, :dash, :dot, :dashdot, :dash, :dot]
    markers = [:circle, :rect, :utriangle, :diamond, :cross, :xcross]

    fig = with_theme(SphericalSBPOperators.mytheme_aps()) do
        fig = Figure(size = (1180, 860), figure_padding = (28, 18, 12, 12))
        Label(fig[0, 1:2], title; fontsize = PAPER_FIGURE_TITLE_SIZE, font = :bold)

        ax_rpi = Axis(fig[1, 1],
                      xlabel = L"r",
                      ylabel = L"r \Pi(t,r)")
        ax_rpsi = Axis(fig[1, 2],
                       xlabel = L"r",
                       ylabel = L"r \Psi(t,r)")
        ax_epi = Axis(fig[2, 1],
                      xlabel = L"r",
                      ylabel = latexstring("\\left(h/h_0\\right)^{-q}\\left(\\Pi_h - \\Pi_{h/2}\\right)"))
        ax_epsi = Axis(fig[2, 2],
                       xlabel = L"r",
                       ylabel = latexstring("\\left(h/h_0\\right)^{-q}\\left(\\Psi_h - \\Psi_{h/2}\\right)"))

        for (ax, limits) in ((ax_rpi, top_pi_limits),
                             (ax_rpsi, top_psi_limits),
                             (ax_epi, bottom_pi_limits),
                             (ax_epsi, bottom_psi_limits))
            hlines!(ax, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
            ylims!(ax, limits...)
        end

        xlims!(ax_rpi, minimum(snaps[end].r), maximum(snaps[end].r))
        xlims!(ax_rpsi, minimum(snaps[end].r), maximum(snaps[end].r))
        xlims!(ax_epi, minimum(pi_pairs[1].r), maximum(pi_pairs[1].r))
        xlims!(ax_epsi, minimum(psi_pairs[1].r), maximum(psi_pairs[1].r))

        for i in eachindex(snaps, resolution_labels)
            snap = snaps[i]
            color = colors[mod1(i, length(colors))]
            linestyle = linestyles[mod1(i, length(linestyles))]
            label = _paper_resolution_curve_label(resolution_labels[i])

            lines!(ax_rpi, snap.r, snap.rpi; color = color, linewidth = 2.5,
                   linestyle = linestyle, label = label)
            lines!(ax_rpsi, snap.r, snap.rpsi; color = color, linewidth = 2.5,
                   linestyle = linestyle, label = label)
        end
        axislegend(ax_rpi; position = :rb, labelsize = PAPER_LEGEND_LABEL_SIZE,
                   framevisible = false)
        axislegend(ax_rpsi; position = :rb, labelsize = PAPER_LEGEND_LABEL_SIZE,
                   framevisible = false)

        for i in eachindex(pi_pairs, psi_pairs)
            color = colors[mod1(i, length(colors))]
            linestyle = linestyles[mod1(i, length(linestyles))]
            marker = markers[mod1(i, length(markers))]
            coarse_label = _paper_resolution_fraction(resolution_labels[i])
            fine_label = _paper_resolution_fraction(resolution_labels[i + 1])
            scale_label = _paper_resolution_scale_label(resolution_labels[i], order)
            curve_label = "h = $(coarse_label) vs $(fine_label), $(scale_label)"

            lines!(ax_epi, pi_pairs[i].r, pi_pairs[i].scaled_err; color = color,
                   linewidth = 2.5, linestyle = linestyle, label = curve_label)
            scatter!(ax_epi, pi_pairs[i].r, pi_pairs[i].scaled_err; color = color,
                     markersize = 7, marker = marker)

            lines!(ax_epsi, psi_pairs[i].r, psi_pairs[i].scaled_err; color = color,
                   linewidth = 2.5, linestyle = linestyle, label = curve_label)
            scatter!(ax_epsi, psi_pairs[i].r, psi_pairs[i].scaled_err; color = color,
                     markersize = 7, marker = marker)
        end
        axislegend(ax_epi; position = :rb, labelsize = PAPER_LEGEND_LABEL_SIZE,
                   framevisible = false)
        axislegend(ax_epsi; position = :rb, labelsize = PAPER_LEGEND_LABEL_SIZE,
                   framevisible = false)

        fig
    end

    return (fig = fig, t = snap_time)
end

function _paper_analytic_snapshot(r::AbstractVector,
                                  time::Real,
                                  reference::NamedTuple)
    exact = _paper_analytic_solution(time, r, reference)
    rf = Float64.(r)
    return (r = rf,
            Π = Float64.(exact.Π),
            Ψ = Float64.(exact.Ψ),
            rpi = rf .* Float64.(exact.Π),
            rpsi = rf .* Float64.(exact.Ψ),
            t = Float64(time))
end

function _paper_pointwise_analytic_error_snapshot(simulation;
                                                  field::Symbol,
                                                  time::Real,
                                                  expected_order::Real,
                                                  h0::Real,
                                                  analytic_reference::NamedTuple)
    sol = _paper_sol(simulation)
    idx, t_saved = _paper_time_index(simulation, time)
    r = Float64.(sol.r)
    history = SphericalSBPOperators._wave_field_history(sol, field)
    u_num = Float64.(history[:, idx])
    exact = _paper_analytic_solution(t_saved, r, analytic_reference)
    field_sym = SphericalSBPOperators._canonical_wave_field(field)
    u_exact = field_sym === :Π ? Float64.(exact.Π) : Float64.(exact.Ψ)
    err = u_num .- u_exact
    h = _paper_grid_spacing(simulation)
    scale = (Float64(h) / Float64(h0))^(-Float64(expected_order))
    return (r = r,
            err = err,
            scaled_err = scale .* err,
            h = Float64(h),
            scale = scale,
            t = t_saved,
            field = field_sym)
end

function _paper_all_resolution_analytic_snapshot_figure(simulations::AbstractVector,
                                                        resolution_labels::AbstractVector;
                                                        family::Symbol,
                                                        order::Real,
                                                        time::Real,
                                                        analytic_reference::NamedTuple)
    length(simulations) == length(resolution_labels) ||
        throw(DimensionMismatch("`simulations` and `resolution_labels` must have the same length."))
    isempty(simulations) &&
        throw(ArgumentError("Need at least one resolution to build analytic snapshots."))

    snaps = [_paper_snapshot_overlay_series(sim, time) for sim in simulations]
    snap_time = snaps[1].t
    analytic_top = _paper_analytic_snapshot(snaps[end].r, snap_time, analytic_reference)

    top_pi_limits = SphericalSBPOperators._finite_limits_with_padding(vcat(reduce(vcat,
                                                                                  (snap.rpi
                                                                                   for snap in snaps)),
                                                                           analytic_top.rpi))
    top_psi_limits = SphericalSBPOperators._finite_limits_with_padding(vcat(reduce(vcat,
                                                                                   (snap.rpsi
                                                                                    for snap in snaps)),
                                                                            analytic_top.rpsi))

    title_family = replace(_paper_operator_title_label(family, order),
                           r" operator$" => " operators")
    title = "Solutions for $(title_family) at t = $(@sprintf("%.6g", snap_time))"

    fig = with_theme(SphericalSBPOperators.mytheme_aps()) do
        fig = Figure(size = (1260, 560), figure_padding = (40, 22, 16, 16))
        Label(fig[0, 1:2], title; fontsize = PAPER_FIGURE_TITLE_SIZE, font = :regular)

        ax_rpi = Axis(fig[1, 1], xlabel = L"r", ylabel = L"r \Pi(t,r)")
        ax_rpsi = Axis(fig[1, 2],
                       xlabel = L"r",
                       ylabel = L"r \Psi(t,r)",
                       ylabelpadding = Float64(PAPER_AXIS_LABEL_PADDING + 20))

        for (ax, limits) in ((ax_rpi, top_pi_limits),
                             (ax_rpsi, top_psi_limits))
            ylims!(ax, limits...)
        end

        xlims!(ax_rpi, minimum(analytic_top.r), maximum(analytic_top.r))
        xlims!(ax_rpsi, minimum(analytic_top.r), maximum(analytic_top.r))

        lines!(ax_rpi, analytic_top.r, analytic_top.rpi; color = :black, linewidth = 3.0,
               linestyle = :dash, label = latexstring("\\mathrm{analytic}"))
        lines!(ax_rpsi, analytic_top.r, analytic_top.rpsi; color = :black, linewidth = 3.0,
               linestyle = :dash, label = latexstring("\\mathrm{analytic}"))
        visible_resolution_indices = [i
                                      for i in eachindex(resolution_labels)
                                      if (_paper_resolution_curve_label(resolution_labels[i]) !=
                                          "h=1" &&
                                          _paper_resolution_curve_label(resolution_labels[i]) !=
                                          "h=1/2")]
        for (visible_i, source_i) in enumerate(visible_resolution_indices)
            snap = snaps[source_i]
            color = _paper_resolution_color(visible_i)
            label_latex = _paper_resolution_curve_latex(resolution_labels[source_i])
            lines!(ax_rpi, snap.r, snap.rpi; color = color, linewidth = 2.5,
                   linestyle = :solid, label = label_latex)
            lines!(ax_rpsi, snap.r, snap.rpsi; color = color, linewidth = 2.5,
                   linestyle = :solid, label = label_latex)
        end

        Legend(fig[2, 1:2], ax_rpi;
               orientation = :horizontal,
               nbanks = 1,
               tellwidth = false,
               tellheight = true,
               framevisible = false,
               halign = :center,
               valign = :center,
               labelsize = PAPER_LEGEND_LABEL_SIZE,
               patchsize = (34, 12),
               colgap = 20,
               rowgap = 6)

        rowgap!(fig.layout, 10)
        colgap!(fig.layout, 46)
        rowsize!(fig.layout, 2, Auto(0.18))

        fig
    end

    return (fig = fig, t = snap_time)
end

function _paper_all_resolution_analytic_error_figure(simulations::AbstractVector,
                                                     resolution_labels::AbstractVector;
                                                     family::Symbol,
                                                     order::Real,
                                                     time::Real,
                                                     h0::Real,
                                                     analytic_reference::NamedTuple)
    length(simulations) == length(resolution_labels) ||
        throw(DimensionMismatch("`simulations` and `resolution_labels` must have the same length."))
    isempty(simulations) &&
        throw(ArgumentError("Need at least one resolution to build analytic error plots."))

    snaps = [_paper_snapshot_overlay_series(sim, time) for sim in simulations]
    snap_time = snaps[1].t
    pi_errors = [_paper_pointwise_analytic_error_snapshot(sim;
                                                          field = :Π,
                                                          time = snap_time,
                                                          expected_order = order,
                                                          h0 = h0,
                                                          analytic_reference = analytic_reference)
                 for sim in simulations]
    psi_errors = [_paper_pointwise_analytic_error_snapshot(sim;
                                                           field = :Ψ,
                                                           time = snap_time,
                                                           expected_order = order,
                                                           h0 = h0,
                                                           analytic_reference = analytic_reference)
                  for sim in simulations]

    bottom_pi_limits = SphericalSBPOperators._finite_limits_with_padding(reduce(vcat,
                                                                                (err.scaled_err
                                                                                 for err in pi_errors)))
    bottom_psi_limits = SphericalSBPOperators._finite_limits_with_padding(reduce(vcat,
                                                                                 (err.scaled_err
                                                                                  for err in psi_errors)))

    title_family = replace(_paper_operator_title_label(family, order),
                           r" operator$" => " operators")
    title = "Scaled Errors for $(title_family) at t = $(@sprintf("%.6g", snap_time))"
    axis_labelsize = Float64(PAPER_AXIS_LABEL_SIZE + 12)
    ylabelpadding = Float64(PAPER_AXIS_LABEL_PADDING + 26)
    yticklabelspace = 38.0
    yticklabelpad = 8.0

    fig = with_theme(SphericalSBPOperators.mytheme_aps()) do
        fig = Figure(size = (1260, 560), figure_padding = (54, 22, 16, 16))
        Label(fig[0, 1:2], title; fontsize = PAPER_FIGURE_TITLE_SIZE, font = :regular)

        ax_epi = Axis(fig[1, 1],
                      xlabel = L"r",
                      ylabel = latexstring("h^{-q}\\left(\\Pi_h - \\Pi_{\\mathrm{analytic}}\\right)"),
                      xlabelsize = axis_labelsize,
                      ylabelsize = axis_labelsize,
                      ylabelpadding = ylabelpadding,
                      yticklabelspace = yticklabelspace,
                      yticklabelpad = yticklabelpad)
        ax_epsi = Axis(fig[1, 2],
                       xlabel = L"r",
                       ylabel = latexstring("h^{-q}\\left(\\Psi_h - \\Psi_{\\mathrm{analytic}}\\right)"),
                       xlabelsize = axis_labelsize,
                       ylabelsize = axis_labelsize,
                       ylabelpadding = ylabelpadding,
                       yticklabelspace = yticklabelspace,
                       yticklabelpad = yticklabelpad)

        for (ax, limits) in ((ax_epi, bottom_pi_limits),
                             (ax_epsi, bottom_psi_limits))
            hlines!(ax, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
            ylims!(ax, limits...)
        end

        xlims!(ax_epi, minimum(pi_errors[end].r), maximum(pi_errors[end].r))
        xlims!(ax_epsi, minimum(psi_errors[end].r), maximum(psi_errors[end].r))

        visible_resolution_indices = [i
                                      for i in eachindex(resolution_labels)
                                      if (_paper_resolution_curve_label(resolution_labels[i]) !=
                                          "h=1" &&
                                          _paper_resolution_curve_label(resolution_labels[i]) !=
                                          "h=1/2")]
        for (visible_i, source_i) in enumerate(visible_resolution_indices)
            color = _paper_resolution_color(visible_i)
            curve_label = _paper_resolution_scale_latex(resolution_labels[source_i], order)

            lines!(ax_epi, pi_errors[source_i].r, pi_errors[source_i].scaled_err;
                   color = color,
                   linewidth = 2.5, linestyle = :solid, label = curve_label)

            lines!(ax_epsi, psi_errors[source_i].r, psi_errors[source_i].scaled_err;
                   color = color,
                   linewidth = 2.5, linestyle = :solid, label = curve_label)
        end
        Legend(fig[2, 1:2], ax_epi;
               orientation = :horizontal,
               nbanks = 1,
               tellwidth = false,
               tellheight = true,
               framevisible = false,
               halign = :center,
               valign = :center,
               labelsize = PAPER_LEGEND_LABEL_SIZE,
               patchsize = (34, 12),
               colgap = 20,
               rowgap = 6)

        rowgap!(fig.layout, 10)
        colgap!(fig.layout, 46)
        rowsize!(fig.layout, 2, Auto(0.18))

        fig
    end

    return (fig = fig, t = snap_time)
end

@inline _paper_divergence_odd_monomial(r::AbstractVector, power::Int) = Float64.(r) .^ power

@inline _paper_divergence_odd_monomial_exact(r::AbstractVector, power::Int, p::Int) = (power +
                                                                                       p) .*
                                                                                      (Float64.(r) .^
                                                                                       (power -
                                                                                        1))

@inline _paper_divergence_gaussian_flux(r::AbstractVector, alpha::Real) = Float64.(r) .*
                                                                          exp.(-Float64(alpha) .*
                                                                               (Float64.(r) .^
                                                                                2))

@inline _paper_divergence_gaussian_exact(r::AbstractVector, alpha::Real) = (3 .-
                                                                            2 .*
                                                                            Float64(alpha) .*
                                                                            (Float64.(r) .^
                                                                             2)) .*
                                                                           exp.(-Float64(alpha) .*
                                                                                (Float64.(r) .^
                                                                                 2))

function _paper_divergence_sine_exact(r::AbstractVector, k::Real)
    rf = Float64.(r)
    out = similar(rf)
    kf = Float64(k)
    for i in eachindex(rf)
        ri = rf[i]
        if abs(ri) <= sqrt(eps(Float64))
            out[i] = 3 * kf
        else
            kri = kf * ri
            out[i] = 2 * sin(kri) / ri + kf * cos(kri)
        end
    end
    return out
end

function _paper_divergence_besselj1_flux(r::AbstractVector, k::Real)
    rf = Float64.(r)
    out = similar(rf)
    kf = Float64(k)
    for i in eachindex(rf)
        x = kf * rf[i]
        if abs(x) <= 1e-5
            x2 = x * x
            out[i] = x / 3 - x * x2 / 30 + x * x2 * x2 / 840
        else
            out[i] = sin(x) / (x * x) - cos(x) / x
        end
    end
    return out
end

function _paper_divergence_besselj1_exact(r::AbstractVector, k::Real)
    rf = Float64.(r)
    out = similar(rf)
    kf = Float64(k)
    for i in eachindex(rf)
        ri = rf[i]
        if abs(ri) <= sqrt(eps(Float64))
            out[i] = kf
        else
            out[i] = sin(kf * ri) / ri
        end
    end
    return out
end

function _paper_divergence_profile_data(r::AbstractVector, spec::NamedTuple, p::Int)
    kind = spec.kind
    if kind === :monomial
        power = Int(spec.power)
        return (u = _paper_divergence_odd_monomial(r, power),
                exact = _paper_divergence_odd_monomial_exact(r, power, p))
    elseif kind === :gaussian
        alpha = Float64(spec.alpha)
        return (u = _paper_divergence_gaussian_flux(r, alpha),
                exact = _paper_divergence_gaussian_exact(r, alpha))
    elseif kind === :sine
        k = Float64(spec.k)
        return (u = sin.(k .* Float64.(r)),
                exact = _paper_divergence_sine_exact(r, k))
    elseif kind === :bessel_j1
        k = Float64(spec.k)
        return (u = _paper_divergence_besselj1_flux(r, k),
                exact = _paper_divergence_besselj1_exact(r, k))
    elseif kind === :gaussian_plus_sine
        alpha = Float64(spec.alpha)
        delta = Float64(spec.delta)
        k = Float64(spec.k)
        base_u = _paper_divergence_gaussian_flux(r, alpha)
        base_exact = _paper_divergence_gaussian_exact(r, alpha)
        return (u = base_u .+ delta .* sin.(k .* Float64.(r)),
                exact = base_exact .+ delta .* _paper_divergence_sine_exact(r, k))
    end
    throw(ArgumentError("Unsupported divergence comparison spec kind `$kind`."))
end

function _paper_divergence_spec_title(spec::NamedTuple; h::Real, p::Int)
    spec.kind in (:gaussian, :sine, :bessel_j1, :gaussian_plus_sine, :monomial) ||
        throw(ArgumentError("Unsupported divergence comparison spec kind `$(spec.kind)`."))
    return "Spherical Divergence comparison"
end

function _paper_divergence_filename_stem(spec::NamedTuple)
    if spec.kind === :gaussian && get(spec, :baseline, false)
        return "current_analytic_gaussian_alpha1"
    elseif spec.kind === :gaussian && haskey(spec, :sigma)
        return "compact_bump_sigma_$(_paper_number_token(spec.sigma))"
    elseif spec.kind === :gaussian
        return "gaussian_alpha_$(_paper_number_token(spec.alpha))"
    elseif spec.kind === :sine
        return "sine_k_$(_paper_number_token(spec.k))"
    elseif spec.kind === :bessel_j1
        return "bessel_j1_k_$(_paper_number_token(spec.k))"
    elseif spec.kind === :gaussian_plus_sine
        return "gaussian_plus_sine_delta_$(_paper_number_token(spec.delta))_k_$(_paper_number_token(spec.k))"
    elseif spec.kind === :monomial
        return "monomial_power_$(Int(spec.power))"
    end
    throw(ArgumentError("Unsupported divergence comparison spec kind `$(spec.kind)`."))
end

function _paper_divergence_plot_limits(values::AbstractVector{<:Real};
                                       pad_frac::Float64 = 0.10)
    vmin, vmax = extrema(values)
    span = vmax - vmin
    scale = max(abs(vmin), abs(vmax), 1.0)
    pad = pad_frac * max(span, scale)
    return (vmin - pad, vmax + pad)
end

function _paper_divergence_error_limits(values::AbstractVector{<:Real};
                                        pad_frac::Float64 = 0.10)
    vmin, vmax = extrema(values)
    span = vmax - vmin
    scale = max(span, abs(vmin), abs(vmax), eps(Float64))
    pad = pad_frac * scale
    return (vmin - pad, vmax + pad)
end

function _paper_divergence_xlimits(values::AbstractVector{<:Real}; pad_frac::Float64 = 0.04)
    vmin, vmax = extrema(values)
    span = vmax - vmin
    scale = max(span, abs(vmax), eps(Float64))
    pad = pad_frac * scale
    return (max(0.0, vmin), vmax + pad)
end

@inline function _paper_divergence_plot_indices(values::AbstractVector,
                                                plot_points::Union{Nothing, Int})
    plot_points === nothing && return eachindex(values)
    return Base.OneTo(min(length(values), plot_points))
end

function _paper_divergence_error_curve_xy(rvals::AbstractVector{<:Real},
                                          errvals::AbstractVector{<:Real})
    isempty(rvals) && return (Float64[], Float64[])

    rplot = Float64.(rvals)
    eplot = Float64.(errvals)
    first(rplot) > 0 || return (rplot, eplot)
    return (vcat(0.0, rplot), vcat(first(eplot), eplot))
end

function _paper_divergence_family_builders(; source,
                                           accuracy_order::Int,
                                           h::Real,
                                           p::Int,
                                           npoints::Int,
                                           mode)
    R_staggered = (npoints - 0.5) * Float64(h)
    non_diagonal_builder = accuracy_order == 6 ?
                           (() -> SphericalSBPOperators.non_diagonal_exp_spherical_operators(source;
                                                                                              N = npoints,
                                                                                              h = h,
                                                                                              p = p,
                                                                                              mode = mode,
                                                                                              outer_boundary_closure_help = false)) :
                           (() -> SphericalSBPOperators.non_diagonal_spherical_operators(source;
                                                                                         accuracy_order = accuracy_order,
                                                                                         N = npoints,
                                                                                         h = h,
                                                                                         p = p,
                                                                                         mode = mode))
    return ((label = "Staggered",
             color = :darkorange3,
             build = () -> SphericalSBPOperators.staggered_spherical_operators(source;
                                                                               accuracy_order = accuracy_order,
                                                                               N = npoints,
                                                                               R = R_staggered,
                                                                               p = p,
                                                                               mode = mode)),
            (label = "Non-diagonal",
             color = :seagreen4,
             build = non_diagonal_builder))
end

function _paper_divergence_compute_data(spec::NamedTuple;
                                        source,
                                        accuracy_order::Int,
                                        h::Real,
                                        p::Int,
                                        npoints::Int,
                                        plot_points::Union{Nothing, Int},
                                        mode)
    families = NamedTuple[]
    for family in _paper_divergence_family_builders(; source = source,
                                                    accuracy_order = accuracy_order,
                                                    h = h,
                                                    p = p,
                                                    npoints = npoints,
                                                    mode = mode)
        ops = family.build()
        r = Float64.(ops.r)
        profile = _paper_divergence_profile_data(r, spec, p)
        approx = Float64.(SphericalSBPOperators.apply_divergence(ops, profile.u))
        error = approx .- profile.exact
        idx = _paper_divergence_plot_indices(r, plot_points)
        push!(families,
              (label = family.label,
               color = family.color,
               r = r,
               approx = approx,
               exact = profile.exact,
               error = error,
               plotted_max_abs_error = maximum(abs.(error[idx]))))
    end
    return families
end

function _paper_single_family_builder(family::Symbol;
                                      source,
                                      accuracy_order::Int,
                                      h::Real,
                                      p::Int,
                                      npoints::Int,
                                      mode)
    R_collocated = (npoints - 1) * Float64(h)
    R_staggered = (npoints - 0.5) * Float64(h)

    if family === :diagonal
        return (label = "Diagonal",
                color = :royalblue4,
                build = () -> SphericalSBPOperators.diagonal_spherical_operators(source;
                                                                                 accuracy_order = accuracy_order,
                                                                                 N = npoints -
                                                                                     1,
                                                                                 R = R_collocated,
                                                                                 p = p,
                                                                                 mode = mode))
    elseif family === :mixed_order_diagonal
        return (label = "Mixed-order diagonal",
                color = :mediumpurple4,
                build = () -> SphericalSBPOperators.mixed_order_diagonal_spherical_operators(source;
                                                                                             accuracy_order = accuracy_order,
                                                                                             N = npoints -
                                                                                                 1,
                                                                                             R = R_collocated,
                                                                                             p = p,
                                                                                             mode = mode))
    elseif family === :staggered
        return (label = "Staggered",
                color = :darkorange3,
                build = () -> SphericalSBPOperators.staggered_spherical_operators(source;
                                                                                  accuracy_order = accuracy_order,
                                                                                  N = npoints,
                                                                                  R = R_staggered,
                                                                                  p = p,
                                                                                  mode = mode))
    elseif family === :non_diagonal
        return (label = "Non-diagonal",
                color = :seagreen4,
                build = () -> SphericalSBPOperators.non_diagonal_spherical_operators(source;
                                                                                     accuracy_order = accuracy_order,
                                                                                     N = npoints,
                                                                                     h = h,
                                                                                     p = p,
                                                                                     mode = mode))
    end

    throw(ArgumentError("Unsupported operator family `$family`. Use :diagonal, :mixed_order_diagonal, :staggered, or :non_diagonal."))
end

function _paper_halving_grid_spacings(h_start::Real, h_stop::Real)
    h_start > 0 || throw(ArgumentError("`h_start` must be positive."))
    h_stop > 0 || throw(ArgumentError("`h_stop` must be positive."))
    h_stop <= h_start ||
        throw(ArgumentError("`h_stop` must be less than or equal to `h_start`."))

    hs = Float64[]
    h = Float64(h_start)
    stop_value = Float64(h_stop)
    while h >= stop_value
        push!(hs, h)
        h *= 0.5
    end
    return hs
end

function _paper_first_point_divergence_error_data(; family::Symbol,
                                                  power::Int,
                                                  source = MattssonNordström2004(),
                                                  accuracy_order::Int = 4,
                                                  p::Int = 2,
                                                  npoints::Int = 25,
                                                  h_start::Real = 1.0,
                                                  h_stop::Real = 1e-14,
                                                  mode = SafeMode())
    isodd(power) || throw(ArgumentError("`power` must be odd."))
    power >= 1 || throw(ArgumentError("`power` must be positive."))
    npoints >= 2 || throw(ArgumentError("`npoints` must be at least 2."))

    builder = _paper_single_family_builder(family;
                                           source = source,
                                           accuracy_order = accuracy_order,
                                           h = h_start,
                                           p = p,
                                           npoints = npoints,
                                           mode = mode)
    hs = _paper_halving_grid_spacings(h_start, h_stop)

    data = NamedTuple[]
    for h in hs
        ops = _paper_single_family_builder(family;
                                           source = source,
                                           accuracy_order = accuracy_order,
                                           h = h,
                                           p = p,
                                           npoints = npoints,
                                           mode = mode).build()
        r = Float64.(ops.r)
        u = _paper_divergence_odd_monomial(r, power)
        discrete = Float64(SphericalSBPOperators.apply_divergence(ops, u)[1])
        exact = Float64(_paper_divergence_odd_monomial_exact(r, power, p)[1])
        signed_error = discrete - exact
        push!(data,
              (h = Float64(h),
               r_first = r[1],
               discrete = discrete,
               exact = exact,
               signed_error = signed_error,
               abs_error = abs(signed_error)))
    end

    return (family = family,
            family_label = builder.label,
            color = builder.color,
            power = power,
            source = source,
            accuracy_order = accuracy_order,
            p = p,
            npoints = npoints,
            data = data)
end

function _paper_first_point_error_title(family_label::AbstractString,
                                        power::Int,
                                        accuracy_order::Int,
                                        p::Int,
                                        npoints::Int)
    family_tex = replace(family_label, " " => "\\ ")
    return latexstring("\\mathrm{First\\text{-}Point\\ Divergence\\ Error}:\\ ",
                       family_tex,
                       ",\\ D(r^{",
                       string(power),
                       "}),\\ s=",
                       string(accuracy_order),
                       ",\\ p=",
                       string(p),
                       ",\\ n=",
                       string(npoints))
end

function _paper_first_point_error_filename(family::Symbol,
                                           power::Int,
                                           accuracy_order::Int,
                                           p::Int,
                                           npoints::Int,
                                           h_start::Real,
                                           h_stop::Real)
    return "first_point_divergence_error_$(_paper_family_tag(family))_power$(power)_order$(accuracy_order)_p$(p)_n$(npoints)_h$(_paper_number_token(h_start))_to_$(_paper_number_token(h_stop)).pdf"
end

function _paper_all_families_first_point_error_filename(power::Int,
                                                        accuracy_order::Int,
                                                        p::Int,
                                                        npoints::Int,
                                                        h_start::Real,
                                                        h_stop::Real)
    return "first_point_divergence_error_all_families_power$(power)_order$(accuracy_order)_p$(p)_n$(npoints)_h$(_paper_number_token(h_start))_to_$(_paper_number_token(h_stop)).pdf"
end

function _paper_reference_order_curve(hs::AbstractVector{<:Real},
                                      abs_errors::AbstractVector{<:Real},
                                      order::Real;
                                      label = latexstring("h^{", string(order), "}"),
                                      visual_offset::Real = 2.0)
    positive_idx = findall(i -> isfinite(abs_errors[i]) && abs_errors[i] > 0, eachindex(hs))
    isempty(positive_idx) && return nothing

    anchor = positive_idx[cld(length(positive_idx), 2)]
    h_anchor = Float64(hs[anchor])
    err_anchor = Float64(abs_errors[anchor])
    scale = Float64(visual_offset) * err_anchor / (h_anchor^order)

    ref = scale .* (Float64.(hs) .^ order)
    return (values = ref, label = label)
end

"""
    plot_first_point_divergence_error_for_paper(; family, power, source=MattssonNordström2004(), accuracy_order=4, p=2, npoints=25, h_start=1.0, h_stop=1e-14, mode=SafeMode(), out_dir=PAPER_DIV_ORIGIN_CONVERGENCE_DIR, filename=nothing, display_figure=true, save_figure=true)

Build one operator family repeatedly while halving the grid spacing from `h_start`
down to `h_stop`, apply the divergence operator to `u(r) = r^power`, and plot the
absolute first-grid-point error against `h`.

The returned `data` table includes both the signed error
`discrete[1] - exact[1]` and its absolute value for every spacing.
"""
function plot_first_point_divergence_error_for_paper(;
                                                     family::Symbol,
                                                     power::Int,
                                                     source = MattssonNordström2004(),
                                                     accuracy_order::Int = 4,
                                                     p::Int = 2,
                                                     npoints::Int = 25,
                                                     h_start::Real = 1.0,
                                                     h_stop::Real = 1e-14,
                                                     mode = SafeMode(),
                                                     out_dir::AbstractString = PAPER_DIV_ORIGIN_CONVERGENCE_DIR,
                                                     filename::Union{Nothing,
                                                                     AbstractString} = nothing,
                                                     display_figure::Bool = true,
                                                     save_figure::Bool = true)
    result = _paper_first_point_divergence_error_data(; family = family,
                                                      power = power,
                                                      source = source,
                                                      accuracy_order = accuracy_order,
                                                      p = p,
                                                      npoints = npoints,
                                                      h_start = h_start,
                                                      h_stop = h_stop,
                                                      mode = mode)

    hs = [entry.h for entry in result.data]
    abs_errors = [entry.abs_error for entry in result.data]
    positive_errors = map(err -> max(err, floatmin(Float64)), abs_errors)
    finite_positive_errors = filter(isfinite, positive_errors)
    isempty(finite_positive_errors) &&
        throw(ArgumentError("No finite first-point errors were produced for plotting."))

    xmin = minimum(hs) / 1.2
    xmax = maximum(hs) * 1.2
    ymin = minimum(finite_positive_errors) / 2
    ymax = maximum(finite_positive_errors) * 2
    reference_order = accuracy_order - p
    reference_curve = reference_order > 0 ?
                      _paper_reference_order_curve(hs,
                                                   abs_errors,
                                                   reference_order;
                                                   label = latexstring("h^{",
                                                                       string(reference_order),
                                                                       "}")) :
                      nothing

    fig = with_theme(SphericalSBPOperators.mytheme_aps()) do
        fig = Figure(size = (1080, 860), figure_padding = (28, 18, 12, 12))
        ax = Axis(fig[1, 1];
                  xlabel = L"h",
                  ylabel = L"\left|(D u)_1 - (D u)^{\mathrm{exact}}_1\right|",
                  title = _paper_first_point_error_title(result.family_label,
                                                         power,
                                                         accuracy_order,
                                                         p,
                                                         npoints),
                  xscale = log10,
                  yscale = log10)
        lines!(ax, hs, positive_errors; color = result.color, linewidth = 2.5)
        if reference_curve !== nothing
            lines!(ax, hs, reference_curve.values;
                   color = :black,
                   linewidth = 2.0,
                   linestyle = :dash,
                   label = reference_curve.label)
        end
        scatter!(ax, hs, positive_errors;
                 color = result.color,
                 markersize = 10,
                 label = latexstring("\\mathrm{",
                                     replace(result.family_label, " " => "\\ "),
                                     "}"))
        axislegend(ax; position = :rb, labelsize = PAPER_LEGEND_LABEL_SIZE,
                   framevisible = false)
        xlims!(ax, xmax, xmin)
        ylims!(ax, ymin, ymax)
        fig
    end

    _paper_apply_typography!(fig)
    display_figure && display(fig)

    save_path = nothing
    if save_figure
        mkpath(out_dir)
        plot_filename = isnothing(filename) ?
                        _paper_first_point_error_filename(family,
                                                          power,
                                                          accuracy_order,
                                                          p,
                                                          npoints,
                                                          h_start,
                                                          h_stop) :
                        String(filename)
        save_path = joinpath(out_dir, plot_filename)
        save(save_path, fig)
    end

    return merge(result, (fig = fig, save_path = save_path))
end

function _paper_all_family_first_point_divergence_error_data(; power::Int,
                                                             source = MattssonNordström2004(),
                                                             accuracy_order::Int = 4,
                                                             p::Int = 2,
                                                             npoints::Int = 25,
                                                             h_start::Real = 1.0,
                                                             h_stop::Real = 1e-14,
                                                             mode = SafeMode())
    families = (:diagonal, :non_diagonal, :staggered, :mixed_order_diagonal)
    return map(family -> _paper_first_point_divergence_error_data(; family = family,
                                                                  power = power,
                                                                  source = source,
                                                                  accuracy_order = accuracy_order,
                                                                  p = p,
                                                                  npoints = npoints,
                                                                  h_start = h_start,
                                                                  h_stop = h_stop,
                                                                  mode = mode),
               families)
end

function _paper_all_families_first_point_error_title(power::Int,
                                                     accuracy_order::Int,
                                                     p::Int,
                                                     npoints::Int)
    return latexstring("\\mathrm{First\\text{-}Point\\ Divergence\\ Error\\ Comparison}:\\ D(r^{",
                       string(power),
                       "}),\\ s=",
                       string(accuracy_order),
                       ",\\ p=",
                       string(p),
                       ",\\ n=",
                       string(npoints))
end

function _paper_should_plot_origin_convergence_family(family_data;
                                                      mixed_order_visibility_threshold::Real = 1e-24)
    if family_data.family === :mixed_order_diagonal
        return maximum(getfield.(family_data.data, :abs_error)) >
               Float64(mixed_order_visibility_threshold)
    end
    return true
end

"""
    plot_all_families_first_point_divergence_error_for_paper(; power, source=MattssonNordström2004(), accuracy_order=4, p=2, npoints=25, h_start=1.0, h_stop=1e-14, mode=SafeMode(), out_dir=PAPER_DIV_ORIGIN_CONVERGENCE_DIR, filename=nothing, display_figure=true, save_figure=true)

Overlay the first-grid-point divergence error curves for the diagonal,
non-diagonal, staggered, and mixed-order diagonal families on one log-log plot.

The returned `families` vector contains the same per-family data tables produced by
`plot_first_point_divergence_error_for_paper`.
"""
function plot_all_families_first_point_divergence_error_for_paper(;
                                                                  power::Int,
                                                                  source = MattssonNordström2004(),
                                                                  accuracy_order::Int = 4,
                                                                  p::Int = 2,
                                                                  npoints::Int = 25,
                                                                  h_start::Real = 1.0,
                                                                  h_stop::Real = 1e-14,
                                                                  mode = SafeMode(),
                                                                  mixed_order_visibility_threshold::Real = 1e-24,
                                                                  out_dir::AbstractString = PAPER_DIV_ORIGIN_CONVERGENCE_DIR,
                                                                  filename::Union{Nothing,
                                                                                  AbstractString} = nothing,
                                                                  display_figure::Bool = true,
                                                                  save_figure::Bool = true)
    all_families = _paper_all_family_first_point_divergence_error_data(; power = power,
                                                                       source = source,
                                                                       accuracy_order = accuracy_order,
                                                                       p = p,
                                                                       npoints = npoints,
                                                                       h_start = h_start,
                                                                       h_stop = h_stop,
                                                                       mode = mode)
    families = filter(family -> _paper_should_plot_origin_convergence_family(family;
                                                                             mixed_order_visibility_threshold = mixed_order_visibility_threshold),
                      all_families)
    isempty(families) &&
        throw(ArgumentError("No family curves remained after applying the origin-convergence visibility filter."))

    hs = [entry.h for entry in first(families).data]
    all_positive_errors = reduce(vcat,
                                 [[max(entry.abs_error, floatmin(Float64))
                                   for entry in family.data]
                                  for family in families])
    finite_positive_errors = filter(isfinite, all_positive_errors)
    isempty(finite_positive_errors) &&
        throw(ArgumentError("No finite first-point errors were produced for plotting."))

    xmin = minimum(hs) / 1.2
    xmax = maximum(hs) * 1.2
    ymin = minimum(finite_positive_errors) / 2
    ymax = maximum(finite_positive_errors) * 2
    reference_order = accuracy_order - p
    reference_curve = reference_order > 0 ?
                      _paper_reference_order_curve(hs,
                                                   [entry.abs_error
                                                    for entry in first(families).data],
                                                   reference_order;
                                                   label = latexstring("h^{",
                                                                       string(reference_order),
                                                                       "}")) :
                      nothing

    fig = with_theme(SphericalSBPOperators.mytheme_aps()) do
        fig = Figure(size = (1080, 860), figure_padding = (28, 18, 12, 12))
        ax = Axis(fig[1, 1];
                  xlabel = L"h",
                  ylabel = L"\left|(D u)_1 - (D u)^{\mathrm{exact}}_1\right|",
                  title = _paper_all_families_first_point_error_title(power,
                                                                      accuracy_order,
                                                                      p,
                                                                      npoints),
                  xscale = log10,
                  yscale = log10)

        for family in families
            errors = [max(entry.abs_error, floatmin(Float64)) for entry in family.data]
            lines!(ax, hs, errors; color = family.color, linewidth = 2.5)
            scatter!(ax, hs, errors;
                     color = family.color,
                     markersize = 10,
                     label = latexstring("\\mathrm{",
                                         replace(family.family_label, " " => "\\ "),
                                         "}"))
        end

        if reference_curve !== nothing
            lines!(ax, hs, reference_curve.values;
                   color = :black,
                   linewidth = 2.0,
                   linestyle = :dash,
                   label = reference_curve.label)
        end

        axislegend(ax; position = :rb, labelsize = PAPER_LEGEND_LABEL_SIZE,
                   framevisible = false)
        xlims!(ax, xmax, xmin)
        ylims!(ax, ymin, ymax)
        fig
    end

    _paper_apply_typography!(fig)
    display_figure && display(fig)

    save_path = nothing
    if save_figure
        mkpath(out_dir)
        plot_filename = isnothing(filename) ?
                        _paper_all_families_first_point_error_filename(power,
                                                                       accuracy_order,
                                                                       p,
                                                                       npoints,
                                                                       h_start,
                                                                       h_stop) :
                        String(filename)
        save_path = joinpath(out_dir, plot_filename)
        save(save_path, fig)
    end

    return (fig = fig, families = families, all_families = all_families,
            save_path = save_path)
end

function _paper_divergence_comparison_figure(spec::NamedTuple;
                                             source = MattssonNordström2004(),
                                             accuracy_order::Int = 4,
                                             h::Real = 1e-4,
                                             p::Int = 2,
                                             npoints::Int = 30,
                                             plot_points::Union{Nothing, Int} = 5,
                                             mode = SafeMode())
    data = _paper_divergence_compute_data(spec;
                                          source = source,
                                          accuracy_order = accuracy_order,
                                          h = h,
                                          p = p,
                                          npoints = npoints,
                                          plot_points = plot_points,
                                          mode = mode)

    fig = with_theme(SphericalSBPOperators.mytheme_aps()) do
        nfamilies = length(data)
        fig = Figure(size = (420 * nfamilies, 760), figure_padding = (12, 12, 6, 6))
        rowgap!(fig.layout, 8)
        colgap!(fig.layout, 8)
        Label(fig[0, 1:nfamilies],
              _paper_divergence_spec_title(spec; h = h, p = p);
              fontsize = PAPER_FIGURE_TITLE_SIZE)

        for (j, entry) in enumerate(data)
            idx = _paper_divergence_plot_indices(entry.r, plot_points)
            ax = Axis(fig[1, j],
                      xlabel = "r",
                      ylabel = L"D\,u",
                      title = latexstring("\\mathrm{", entry.label, "}"))
            lines!(ax, entry.r[idx], entry.exact[idx];
                   color = :black,
                   linewidth = 2.0,
                   linestyle = :dash,
                   label = "exact")
            scatter!(ax, entry.r[idx], entry.approx[idx];
                     color = entry.color,
                     markersize = 10,
                     label = "discrete")
            lines!(ax, entry.r[idx], entry.approx[idx];
                   color = entry.color,
                   linewidth = 2.0)
            axislegend(ax; position = :lt,
                       labelsize = PAPER_DIVERGENCE_COMPARISON_LEGEND_LABEL_SIZE,
                       margin = (18, 0, 0, 0))

            xlo, xhi = _paper_divergence_xlimits(entry.r[idx])
            ylo, yhi = _paper_divergence_plot_limits(vcat(entry.exact[idx],
                                                          entry.approx[idx]))
            xlims!(ax, xlo, xhi)
            ylims!(ax, ylo, yhi)
        end

        ax_err = Axis(fig[2, 1:length(data)],
                      xlabel = "r",
                      ylabel = L"\mathrm{discrete} - \mathrm{exact}",
                      title = L"\mathrm{Error}")
        hlines!(ax_err, [0.0]; color = :black, linewidth = 1.0, linestyle = :dash)
        for entry in data
            idx = _paper_divergence_plot_indices(entry.r, plot_points)
            line_r, line_err = _paper_divergence_error_curve_xy(entry.r[idx],
                                                                entry.error[idx])
            lines!(ax_err, line_r, line_err;
                   linewidth = 2.5,
                   color = entry.color,
                   label = "$(entry.label) (max = $(round(entry.plotted_max_abs_error; sigdigits = 4)))")
            scatter!(ax_err, entry.r[idx], entry.error[idx];
                     color = entry.color,
                     markersize = 8)
        end
        axislegend(ax_err; position = :rt,
                   labelsize = PAPER_DIVERGENCE_COMPARISON_LEGEND_LABEL_SIZE)

        err_x = reduce(vcat,
                       [_paper_divergence_error_curve_xy(entry.r[_paper_divergence_plot_indices(entry.r,
                                                                                                plot_points)],
                                                         entry.error[_paper_divergence_plot_indices(entry.r,
                                                                                                    plot_points)])[1]
                        for entry in data])
        err_y = reduce(vcat,
                       [entry.error[_paper_divergence_plot_indices(entry.r, plot_points)]
                        for entry in data])
        xlo, xhi = _paper_divergence_xlimits(err_x)
        ylo, yhi = _paper_divergence_error_limits(vcat(err_y, [0.0]))
        xlims!(ax_err, xlo, xhi)
        ylims!(ax_err, ylo, yhi)
        fig
    end

    _paper_apply_typography!(fig)
    return (fig = fig, families = data)
end

function _paper_divergence_specs(; perturbation_deltas = (1e-14, 1e-13, 1e-12),
                                 perturbation_k::Real = 8 * π,
                                 bessel_ks = (π, 4π, 8π, 16π),
                                 bump_sigmas = (0.5, 0.1, 0.05, 0.02),
                                 sine_ks = (π, 4π, 8π, 16π),
                                 gaussian_alphas = (10.0, 100.0, 1000.0))
    specs = NamedTuple[]
    push!(specs, (kind = :gaussian, alpha = 1.0, baseline = true))
    for delta in perturbation_deltas
        push!(specs,
              (kind = :gaussian_plus_sine, alpha = 1.0, delta = delta, k = perturbation_k))
    end
    for k in bessel_ks
        push!(specs, (kind = :bessel_j1, k = Float64(k)))
    end
    for sigma in bump_sigmas
        push!(specs,
              (kind = :gaussian, alpha = 1 / (Float64(sigma)^2), sigma = Float64(sigma)))
    end
    for k in sine_ks
        push!(specs, (kind = :sine, k = Float64(k)))
    end
    for alpha in gaussian_alphas
        push!(specs, (kind = :gaussian, alpha = Float64(alpha)))
    end
    return specs
end

"""
    plot_paper_divergence_comparison(; kwargs...)

Generate paper-style divergence-comparison figures for regular analytic radial flux
tests. By default this includes the current analytic Gaussian baseline together with
the requested test families `6`, `8`, `4`, `3`, and `1`, saving one figure per case.
"""
function plot_paper_divergence_comparison(;
                                          out_dir::AbstractString = PAPER_DIVERGENCE_COMPARISON_DIR,
                                          file_ext::AbstractString = "pdf",
                                          source = MattssonNordström2004(),
                                          accuracy_order::Int = 4,
                                          h::Real = 1e-4,
                                          p::Int = 2,
                                          npoints::Int = 30,
                                          plot_points::Union{Nothing, Int} = 5,
                                          mode = SafeMode(),
                                          perturbation_deltas = (1e-14, 1e-13, 1e-12),
                                          perturbation_k::Real = 8 * π,
                                          bessel_ks = (π, 4π, 8π, 16π),
                                          bump_sigmas = (0.5, 0.1, 0.05, 0.02),
                                          sine_ks = (π, 4π, 8π, 16π),
                                          gaussian_alphas = (10.0, 100.0, 1000.0))
    mkpath(out_dir)
    outputs = Dict{String, String}()
    specs = _paper_divergence_specs(; perturbation_deltas = perturbation_deltas,
                                    perturbation_k = perturbation_k,
                                    bessel_ks = bessel_ks,
                                    bump_sigmas = bump_sigmas,
                                    sine_ks = sine_ks,
                                    gaussian_alphas = gaussian_alphas)

    for spec in specs
        plotted = _paper_divergence_comparison_figure(spec;
                                                      source = source,
                                                      accuracy_order = accuracy_order,
                                                      h = h,
                                                      p = p,
                                                      npoints = npoints,
                                                      plot_points = plot_points,
                                                      mode = mode)
        stem = _paper_divergence_filename_stem(spec)
        path = joinpath(out_dir,
                        "divergence_comparison_$(stem)_order$(accuracy_order)_h$(_paper_number_token(h))_p$(p).$(file_ext)")
        save(path, plotted.fig)
        outputs[stem] = path
        println("Saved divergence comparison: ", path)
    end

    return outputs
end

"""
    load_simulation_files(; sim_dir=PAPER_SIM_DIR, prefer_legacy=false)

Load all recognized JLD2 simulation files in `data/sims`.

Returns a dictionary keyed by `(boundary_group, family, order, variant)`, where
`boundary_group` is `:radiative` or `:reflective` and `variant` separates
special cases such as the experimental no-outer-boundary-help bundle. When
multiple non-archived files map to the same case, bundles carrying in-file
`metadata` win, then the legacy filename ranking. Files inside `archive*`
directories are ignored completely.
"""
function load_simulation_files(; sim_dir::AbstractString = PAPER_SIM_DIR,
                               prefer_legacy::Bool = false)
    isdir(sim_dir) || throw(ArgumentError("Simulation directory does not exist: $sim_dir"))

    selected = Dict{Tuple{Symbol, Symbol, Int, Symbol}, NamedTuple}()
    for (root, _, files) in walkdir(sim_dir)
        for name in sort(files)
            endswith(name, ".jld2") || continue
            path = joinpath(root, name)
            path_norm = replace(normpath(path), '\\' => '/')
            occursin(r"/archive[^/]*/", path_norm) && continue
            data = load(path)
            metadata = _paper_bundle_metadata(data, path)

            key = (metadata.boundary_group,
                   metadata.family,
                   metadata.accuracy_order,
                   _paper_case_variant(metadata))
            old = get(selected, key, nothing)
            parsed = _paper_parse_sim_file(path)
            parsed_score = parsed.ok ?
                           _paper_file_rank(parsed; prefer_legacy = prefer_legacy) : 0
            metadata_score = haskey(data, "metadata") ? 1_000 : 0
            score = metadata_score + parsed_score
            if old === nothing || score > old.score
                selected[key] = (path = path,
                                 score = score,
                                 metadata = metadata,
                                 data = data)
            end
        end
    end

    simulations = Dict{Tuple{Symbol, Symbol, Int, Symbol}, NamedTuple}()
    for (key, meta) in selected
        boundary_group, family, order, variant = key
        simulations[key] = (boundary_group = boundary_group,
                            family = family,
                            variant = variant,
                            order = order,
                            path = meta.path,
                            metadata = meta.metadata,
                            data = meta.data)
        println("Loaded ",
                _paper_boundary_label(boundary_group), " / ",
                _paper_case_label(family, order),
                variant === :no_outer_boundary_help ? " [no outer boundary help]" : "",
                ": ",
                meta.path)
    end

    return simulations
end

"""
    plot_spectra_for_paper(; kwargs...)

Generate the spectrum plots for all active paper cases:

- diagonal orders 4, 6, and 8;
- non-diagonal orders 4 and 6.
"""
function plot_spectra_for_paper(;
                                out_dir::AbstractString = PAPER_SPECTRA_DIR,
                                points::Int = 31,
                                R::Real = 30,
                                p::Int = 2,
                                tiny_zero_tol::Float64 = 1.0e-16,
                                boundary_model::Symbol = :sat,
                                mode = SafeMode(),
                                width_mode::Symbol = :twocolumn,
                                export_plots_dir::Union{Nothing, AbstractString} = nothing)
    # Refresh the spectrum helpers so long-lived REPL sessions pick up the
    # CairoMakie PDF export path instead of any previously loaded GLMakie-based one.
    include(joinpath(@__DIR__, "generate_banded_spectrum_plots.jl"))

    Rq = big(numerator(rationalize(R))) // big(denominator(rationalize(R)))

    diagonal = generate_method_spectrum_plots(;
                                              method = :diagonal,
                                              orders = (4, 6, 8),
                                              points = points,
                                              R = Rq,
                                              p = p,
                                              tiny_zero_tol = tiny_zero_tol,
                                              boundary_model = boundary_model,
                                              mode = mode,
                                              width_mode = width_mode,
                                              axis_labelsize = PAPER_AXIS_LABEL_SIZE,
                                              tick_labelsize = PAPER_TICK_LABEL_SIZE,
                                              title_size = PAPER_FIGURE_TITLE_SIZE,
                                              legend_labelsize = PAPER_LEGEND_LABEL_SIZE,
                                              out_dir = joinpath(out_dir, "diagonal"),
                                              export_plots_dir = export_plots_dir)

    non_diagonal = generate_method_spectrum_plots(;
                                                  method = :banded,
                                                  orders = (4, 6),
                                                  points = points,
                                                  R = Rq,
                                                  p = p,
                                                  tiny_zero_tol = tiny_zero_tol,
                                                  boundary_model = boundary_model,
                                                  mode = mode,
                                                  width_mode = width_mode,
                                                  axis_labelsize = PAPER_AXIS_LABEL_SIZE,
                                                  tick_labelsize = PAPER_TICK_LABEL_SIZE,
                                                  title_size = PAPER_FIGURE_TITLE_SIZE,
                                                  legend_labelsize = PAPER_LEGEND_LABEL_SIZE,
                                                  out_dir = joinpath(out_dir,
                                                                     "non_diagonal"),
                                                  export_plots_dir = export_plots_dir)

    return (diagonal = diagonal, non_diagonal = non_diagonal)
end

"""
    plot_snapshots_for_paper(simulations=load_simulation_files(); kwargs...)

Save `Π`/`Ψ` snapshots at `t=5` and `t=15` for every loaded simulation case.
By default the highest saved resolution, `run_h16`, is used.
"""
function plot_snapshots_for_paper(simulations = load_simulation_files();
                                  out_dir::AbstractString = PAPER_WAVE_SNAPSHOT_DIR,
                                  times::Tuple{Vararg{<:Real}} = PAPER_SNAPSHOT_TIMES,
                                  resolution::Symbol = :run_h16,
                                  file_ext::AbstractString = "pdf")
    mkpath(out_dir)
    outputs = Dict{Tuple{Symbol, Symbol, Int, Float64}, String}()

    for key in sort(collect(keys(simulations));
                    by = x -> _paper_case_sort_key(simulations[x]))
        entry = simulations[key]
        boundary_group = entry.boundary_group
        family = entry.family
        order = entry.order
        sim = _paper_run(entry.data, resolution)
        case_out_dir = _paper_case_out_dir(out_dir, boundary_group)

        for t_target in times
            idx, t_saved = _paper_time_index(sim, t_target)
            title = "$(_paper_case_label(family, order)), t = $(@sprintf("%.6g", t_saved))"
            plotted = SphericalSBPOperators.plot_wave_snapshot(sim, idx; title = title)
            _paper_apply_typography!(plotted.fig)
            run_token = _paper_run_token(entry, resolution)
            path = joinpath(case_out_dir,
                            "snapshot_$(_paper_family_tag(family))_order$(order)_$(run_token)_$(resolution)_t$(_paper_number_token(t_saved)).$(file_ext)")
            save(path, plotted.fig)
            outputs[(boundary_group, family, order, Float64(t_target))] = path
            println("Saved snapshot: ", path)
        end
    end

    return outputs
end

"""
    plot_time_evolution_for_paper(simulations=load_simulation_files(); kwargs...)

Save space-time heatmaps of the evolution for both `Π` and `Ψ` for every loaded
simulation case. By default the highest saved resolution, `run_h16`, is used.
"""
function plot_time_evolution_for_paper(simulations = load_simulation_files();
                                       out_dir::AbstractString = PAPER_TIME_EVOLUTION_DIR,
                                       resolution::Symbol = :run_h16,
                                       interpolate::Bool = true,
                                       file_ext::AbstractString = "pdf")
    mkpath(out_dir)
    outputs = Dict{Tuple{Symbol, Symbol, Int}, String}()

    for key in sort(collect(keys(simulations));
                    by = x -> _paper_case_sort_key(simulations[x]))
        entry = simulations[key]
        boundary_group = entry.boundary_group
        family = entry.family
        order = entry.order
        sim = _paper_run(entry.data, resolution)
        case_out_dir = _paper_case_out_dir(out_dir, boundary_group)

        title = latexstring("\\mathrm{Time\\ evolution\\ with\\ }6^{\\mathrm{th}}\\ \\mathrm{order\\ non\\ diagonal\\ norm\\ operators.}")
        fig = _paper_time_evolution_figure(sim;
                                           title = title,
                                           interpolate = interpolate)
        _paper_apply_typography!(fig)

        run_token = _paper_run_token(entry, resolution)
        path = joinpath(case_out_dir,
                        "time_evolution_$(_paper_family_tag(family))_order$(order)_$(run_token)_$(resolution).$(file_ext)")
        save(path, fig; pt_per_unit = 1)
        outputs[(boundary_group, family, order)] = path
        println("Saved time-evolution plot: ", path)
    end

    return outputs
end

"""
    plot_energy_for_paper(simulations=nothing; kwargs...)

Save relative energy-conservation plots for each active simulation bundle.
When `simulations === nothing`, this scans `data/sims/reflective`, ignores
`archive*` folders, and generates one figure per `.jld2` bundle found there.
By default this overlays every available saved resolution except `run_h` and `run_h2`.
Supported `difference_mode` values are `:none`, `:absolute`, and `:relative`.
Use `yscale = :symlog` to switch the y-axis to a symmetric log scale.
"""
function plot_energy_for_paper(simulations = nothing;
                               out_dir::AbstractString = PAPER_WAVE_ENERGY_DIR,
                               sim_dir::AbstractString = joinpath(PAPER_SIM_DIR, "reflective"),
                               resolutions::Union{Nothing, Tuple{Vararg{Symbol}}} = nothing,
                               difference_mode::Symbol = :relative,
                               yscale::Symbol = :identity,
                               symlog_threshold::Union{Nothing, Real} = nothing,
                               file_ext::AbstractString = "pdf")
    mkpath(out_dir)
    outputs = Dict{String, String}()

    entries = if simulations === nothing
        _paper_list_simulation_bundles(; sim_dir = sim_dir)
    elseif simulations isa AbstractDict
        [merge((key = key,), simulations[key]) for key in sort(collect(keys(simulations));
                                                              by = x -> _paper_case_sort_key(simulations[x]))]
    else
        collect(simulations)
    end

    for entry in entries
        boundary_group = entry.boundary_group
        family = entry.family
        order = entry.order
        variant = entry.variant
        family_file_tag = _paper_family_file_tag(family, entry.metadata)
        base_resolutions = isnothing(resolutions) ?
                           Tuple(label
                                 for label in _paper_available_resolutions(entry.data)
                                 if label ∉ (:run_h, :run_h2)) :
                           resolutions
        selected_resolutions = Tuple(label
                                     for label in base_resolutions if label != :run_h2)
        isempty(selected_resolutions) && begin
            @warn "Skipping energy plot because there are no saved resolutions after removing `run_h` and `run_h2`." path=entry.path available_keys=sort(collect(keys(entry.data)))
            continue
        end

        sims = [_paper_run(entry.data, resolution) for resolution in selected_resolutions]
        labels = [_paper_resolution_curve_label(resolution)
                  for resolution in selected_resolutions]
        bundle_stem = replace(basename(entry.path), r"\.jld2$" => "")

        title = difference_mode in (:absolute, :relative) ?
                "Error in Energy conservation" :
                "Wave Energy vs Time"
        plotted = _paper_plot_wave_energy_histories(sims;
                                                    difference_mode = difference_mode,
                                                    resolution_labels = collect(selected_resolutions),
                                                    labels = labels,
                                                    title = title,
                                                    yscale = yscale,
                                                    symlog_threshold = symlog_threshold)
        _paper_move_legend_lb!(plotted.fig)
        _paper_apply_energy_axis_style!(plotted.fig; yscale = yscale)
        _paper_apply_typography!(plotted.fig)
        _paper_apply_energy_typography!(plotted.fig)
        filename = "energy_$(bundle_stem)_$(difference_mode).$(file_ext)"
        path = _paper_save_case_figure(plotted.fig,
                                       out_dir,
                                       boundary_group,
                                       filename)
        outputs[entry.path] = path
        println("Saved energy plot: ", path)
    end

    return outputs
end

"""
    plot_scaled_errors_for_paper(simulations=load_simulation_files(); kwargs...)

Save the static Gundlach scaled-error plots at `t=15` for every loaded simulation
case. By default the plotted nested runs are `run_h4`, `run_h8`, and `run_h16`,
but you can switch to the `run_h8`, `run_h16`, `run_h32` triplet with
`triplet_choice = :h8_h16_h32`. The normalization scale `h0` is taken from `run_h`.
"""
function plot_scaled_errors_for_paper(simulations = load_simulation_files();
                                      out_dir::AbstractString = PAPER_WAVE_CONVERGENCE_DIR,
                                      time::Real = 15.0,
                                      resolutions::NTuple{3, Symbol} = PAPER_SCALED_ERROR_RESOLUTION_TRIPLET,
                                      triplet_choice::Union{Nothing, Symbol} = nothing,
                                      h0_resolution::Symbol = :run_h,
                                      file_ext::AbstractString = "pdf")
    mkpath(out_dir)
    outputs = Dict{Tuple{Symbol, Symbol, Int}, String}()
    selected_resolutions = isnothing(triplet_choice) ? resolutions :
                           _paper_scaled_error_triplet(triplet_choice)

    for key in sort(collect(keys(simulations));
                    by = x -> _paper_case_sort_key(simulations[x]))
        println("Creating scaled error plots for $(key)...")
        entry = simulations[key]
        boundary_group = entry.boundary_group
        boundary_group === :radiative || continue
        family = entry.family
        order = entry.order
        required_runs = (h0_resolution, selected_resolutions...)
        missing_runs = _paper_missing_runs(entry.data, required_runs)
        if !isempty(missing_runs)
            @warn "Skipping scaled error plot due to missing saved runs." case=key missing_runs=missing_runs available_keys=sort(collect(keys(entry.data)))
            continue
        end

        sim_h0 = _paper_run(entry.data, h0_resolution)
        sim1 = _paper_run(entry.data, selected_resolutions[1])
        sim2 = _paper_run(entry.data, selected_resolutions[2])
        sim3 = _paper_run(entry.data, selected_resolutions[3])

        h0 = _paper_grid_spacing(sim_h0)
        plotted = SphericalSBPOperators.plot_pointwise_wave_scaled_errors_gundlach(sim1,
                                                                                   sim2,
                                                                                   sim3;
                                                                                   time = time,
                                                                                   h0 = h0)
        _paper_apply_typography!(plotted.fig)

        run_token = _paper_energy_run_token(entry, selected_resolutions)
        resolution_token = join(String.(selected_resolutions), "-")
        h0_token = "h0_from_$(h0_resolution)_$(_paper_number_token(h0))"
        case_out_dir = _paper_case_out_dir(out_dir, boundary_group)
        path = joinpath(case_out_dir,
                        "scaled_error_gundlach_$(_paper_family_tag(family))_order$(order)_$(run_token)_$(resolution_token)_$(h0_token)_t$(_paper_number_token(plotted.t)).$(file_ext)")
        save(path, plotted.fig)
        outputs[(boundary_group, family, order)] = path
        println("Saved scaled-error plot: ", path)
    end

    return outputs
end

"""
    plot_convergence_snapshots_for_paper(simulations=load_simulation_files(); kwargs...)

Save 2x2 convergence snapshot figures for every loaded simulation case. The top row
overlays `rΠ` and `rΨ` from the three requested resolutions at the chosen time. The
bottom row shows the corresponding Gundlach scaled-error curves built from the same
resolution triplet. By default that triplet is `run_h4`, `run_h8`, `run_h16`, and
you can switch to `run_h8`, `run_h16`, `run_h32` with
`triplet_choice = :h8_h16_h32`.
"""
function plot_convergence_snapshots_for_paper(simulations = load_simulation_files();
                                              out_dir::AbstractString = PAPER_WAVE_CONVERGENCE_SNAPSHOT_DIR,
                                              time::Real = 15.0,
                                              resolutions::NTuple{3, Symbol} = PAPER_SCALED_ERROR_RESOLUTION_TRIPLET,
                                              triplet_choice::Union{Nothing, Symbol} = nothing,
                                              h0_resolution::Symbol = :run_h,
                                              file_ext::AbstractString = "pdf")
    mkpath(out_dir)
    outputs = Dict{Tuple{Symbol, Symbol, Int}, String}()
    resolution_labels = ("low res", "mid res", "high res")
    selected_resolutions = isnothing(triplet_choice) ? resolutions :
                           _paper_scaled_error_triplet(triplet_choice)

    for key in sort(collect(keys(simulations));
                    by = x -> _paper_case_sort_key(simulations[x]))
        entry = simulations[key]
        boundary_group = entry.boundary_group
        boundary_group === :radiative || continue
        family = entry.family
        order = entry.order
        required_runs = (h0_resolution, selected_resolutions...)
        missing_runs = _paper_missing_runs(entry.data, required_runs)
        if !isempty(missing_runs)
            @warn "Skipping convergence snapshot due to missing saved runs." case=key missing_runs=missing_runs available_keys=sort(collect(keys(entry.data)))
            continue
        end

        sim_h0 = _paper_run(entry.data, h0_resolution)
        sim1 = _paper_run(entry.data, selected_resolutions[1])
        sim2 = _paper_run(entry.data, selected_resolutions[2])
        sim3 = _paper_run(entry.data, selected_resolutions[3])

        h0 = _paper_grid_spacing(sim_h0)
        plotted = _paper_convergence_snapshot_figure(sim1, sim2, sim3;
                                                     family = family,
                                                     order = order,
                                                     time = time,
                                                     h0 = h0,
                                                     resolution_labels = resolution_labels)
        _paper_apply_typography!(plotted.fig)

        run_token = _paper_energy_run_token(entry, selected_resolutions)
        resolution_token = join(String.(selected_resolutions), "-")
        h0_token = "h0_from_$(h0_resolution)_$(_paper_number_token(h0))"
        case_out_dir = _paper_case_out_dir(out_dir, boundary_group)
        path = joinpath(case_out_dir,
                        "convergence_snapshot_$(_paper_family_tag(family))_order$(order)_$(run_token)_$(resolution_token)_$(h0_token)_t$(_paper_number_token(plotted.t)).$(file_ext)")
        save(path, plotted.fig)
        outputs[(boundary_group, family, order)] = path
        println("Saved convergence snapshot: ", path)
    end

    return outputs
end

"""
    plot_all_resolution_error_snapshots_for_paper(simulations=load_simulation_files(); kwargs...)

Save 2x2 convergence-style snapshot figures for every loaded simulation case using all
available saved resolutions. The top row overlays `rΠ` and `rΨ` for every available
run. The bottom row shows the plain pointwise differences between adjacent nested
resolutions, scaled by `(h/h0)^(-q)` with `q` taken from the operator accuracy order.
"""
function plot_all_resolution_error_snapshots_for_paper(simulations = load_simulation_files();
                                                       out_dir::AbstractString = PAPER_WAVE_ALL_RESOLUTION_ERROR_SNAPSHOT_DIR,
                                                       time::Real = 15.0,
                                                       h0_resolution::Symbol = :run_h,
                                                       file_ext::AbstractString = "pdf")
    mkpath(out_dir)
    outputs = Dict{Tuple{Symbol, Symbol, Int}, String}()

    for key in sort(collect(keys(simulations));
                    by = x -> _paper_case_sort_key(simulations[x]))
        entry = simulations[key]
        boundary_group = entry.boundary_group
        boundary_group === :radiative || continue
        family = entry.family
        order = entry.order
        available_resolutions = collect(_paper_available_resolutions(entry.data))
        required_runs = (h0_resolution, available_resolutions...)
        missing_runs = _paper_missing_runs(entry.data, required_runs)
        if !isempty(missing_runs)
            @warn "Skipping all-resolution error snapshot due to missing saved runs." case=key missing_runs=missing_runs available_keys=sort(collect(keys(entry.data)))
            continue
        end
        length(available_resolutions) >= 2 || continue

        sim_h0 = _paper_run(entry.data, h0_resolution)
        sims = [_paper_run(entry.data, resolution) for resolution in available_resolutions]
        h0 = _paper_grid_spacing(sim_h0)
        plotted = _paper_all_resolution_error_snapshot_figure(sims, available_resolutions;
                                                              family = family,
                                                              order = order,
                                                              time = time,
                                                              h0 = h0)
        _paper_apply_typography!(plotted.fig)

        run_token = _paper_energy_run_token(entry, Tuple(available_resolutions))
        resolution_token = join(String.(available_resolutions), "-")
        h0_token = "h0_from_$(h0_resolution)_$(_paper_number_token(h0))"
        case_out_dir = _paper_case_out_dir(out_dir, boundary_group)
        path = joinpath(case_out_dir,
                        "all_resolution_error_snapshot_$(_paper_family_tag(family))_order$(order)_$(run_token)_$(resolution_token)_$(h0_token)_t$(_paper_number_token(plotted.t)).$(file_ext)")
        save(path, plotted.fig)
        outputs[(boundary_group, family, order)] = path
        println("Saved all-resolution error snapshot: ", path)
    end

    return outputs
end

"""
    plot_all_resolution_analytic_snapshots_for_paper(simulations=load_simulation_files(); kwargs...)

Save 1x2 analytic snapshot figures containing only the top-row solution panels from
the all-resolution analytic snapshot plots. When `simulations === nothing`, this
scans `data/sims/reflective`, ignores `archive*` folders, and generates one figure
per active `.jld2` bundle. Output files are written under
`plots/wave/snapshots/<boundary>/<cfl...>/`.
"""
function plot_all_resolution_analytic_snapshots_for_paper(simulations = nothing;
                                                          out_dir::AbstractString = PAPER_WAVE_ALL_RESOLUTION_ANALYTIC_ERROR_SNAPSHOT_DIR,
                                                          sim_dir::AbstractString = joinpath(PAPER_SIM_DIR, "reflective"),
                                                          time::Real = 15.0,
                                                          reflective_time::Union{Nothing, Real} = nothing,
                                                          h0_resolution::Symbol = :run_h,
                                                          file_ext::AbstractString = "pdf")
    mkpath(out_dir)
    outputs = Dict{String, String}()

    entries = if simulations === nothing
        _paper_list_simulation_bundles(; sim_dir = sim_dir)
    elseif simulations isa AbstractDict
        [merge((key = key,), simulations[key]) for key in sort(collect(keys(simulations));
                                                              by = x -> _paper_case_sort_key(simulations[x]))]
    else
        collect(simulations)
    end

    for entry in entries
        boundary_group = entry.boundary_group
        _paper_supported_analytic_initial_data(entry.metadata.initial_data_kind) || continue

        family = entry.family
        family_file_tag = _paper_family_file_tag(family, entry.metadata)
        order = entry.order
        case_time = boundary_group === :reflective && !isnothing(reflective_time) ?
                    Float64(reflective_time) : Float64(time)
        available_resolutions = collect(_paper_available_resolutions(entry.data))
        selected_resolutions = [resolution
                                for resolution in available_resolutions
                                if resolution != :run_h2]
        required_runs = (h0_resolution, selected_resolutions...)
        missing_runs = _paper_missing_runs(entry.data, required_runs)
        if !isempty(missing_runs)
            @warn "Skipping all-resolution analytic snapshot figure due to missing saved runs." path=entry.path missing_runs=missing_runs available_keys=sort(collect(keys(entry.data)))
            continue
        end
        isempty(selected_resolutions) && continue

        sim_h0 = _paper_run(entry.data, h0_resolution)
        h0 = _paper_grid_spacing(sim_h0)
        plotted = try
            sims = [_paper_run(entry.data, resolution)
                    for resolution in selected_resolutions]
            analytic_reference = _paper_analytic_reference(entry)
            _paper_all_resolution_analytic_snapshot_figure(sims,
                                                           selected_resolutions;
                                                           family = family,
                                                           order = order,
                                                           time = case_time,
                                                           analytic_reference = analytic_reference)
        catch err
            @warn "Skipping all-resolution analytic snapshot figure because the analytic reference is not valid for this case/time." path=entry.path boundary_condition=_paper_entry_boundary_condition(entry) error=sprint(showerror,
                                                                                                                                                                                                            err)
            continue
        end
        _paper_apply_typography!(plotted.fig)

        run_token = _paper_energy_run_token(entry, Tuple(selected_resolutions))
        initial_data_stem = _paper_initial_data_stem(entry.metadata.initial_data_kind)
        resolution_token = join(String.(selected_resolutions), "-")
        h0_token = "h0_from_$(h0_resolution)_$(_paper_number_token(h0))"
        cfl_token = _paper_cfl_token(entry, Tuple(selected_resolutions))
        case_out_dir = _paper_case_cfl_out_dir(out_dir, boundary_group, cfl_token)
        path = joinpath(case_out_dir,
                        "all_resolution_analytic_snapshot_$(initial_data_stem)_$(family_file_tag)_order$(order)_$(run_token)_$(resolution_token)_$(h0_token)_t$(_paper_number_token(plotted.t)).$(file_ext)")
        save(path, plotted.fig)
        outputs[entry.path] = path
        println("Saved all-resolution analytic snapshot figure: ", path)
    end

    return outputs
end

Base.@deprecate plot_all_resolution_analytic_error_snapshots_for_paper(args...; kwargs...) plot_all_resolution_analytic_snapshots_for_paper(args...; kwargs...)

"""
    plot_all_resolution_analytic_errors_for_paper(simulations=nothing; kwargs...)

Save 1x2 analytic-error-only figures containing just the bottom-row panels from
the all-resolution analytic snapshot plots. When `simulations === nothing`, this
scans `data/sims/reflective`, ignores `archive*` folders, and generates one figure
per active `.jld2` bundle. Output files are written under
`plots/wave/all_resolution_analytic_error/<boundary>/<cfl...>/`.
"""
function plot_all_resolution_analytic_errors_for_paper(simulations = nothing;
                                                       out_dir::AbstractString = PAPER_WAVE_ALL_RESOLUTION_ANALYTIC_ERROR_DIR,
                                                       sim_dir::AbstractString = joinpath(PAPER_SIM_DIR, "reflective"),
                                                       time::Real = 15.0,
                                                       reflective_time::Union{Nothing, Real} = nothing,
                                                       h0_resolution::Symbol = :run_h,
                                                       file_ext::AbstractString = "pdf")
    mkpath(out_dir)
    outputs = Dict{String, String}()

    entries = if simulations === nothing
        _paper_list_simulation_bundles(; sim_dir = sim_dir)
    elseif simulations isa AbstractDict
        [merge((key = key,), simulations[key]) for key in sort(collect(keys(simulations));
                                                              by = x -> _paper_case_sort_key(simulations[x]))]
    else
        collect(simulations)
    end

    for entry in entries
        boundary_group = entry.boundary_group
        _paper_supported_analytic_initial_data(entry.metadata.initial_data_kind) || continue

        family = entry.family
        family_file_tag = _paper_family_file_tag(family, entry.metadata)
        order = entry.order
        case_time = boundary_group === :reflective && !isnothing(reflective_time) ?
                    Float64(reflective_time) : Float64(time)
        available_resolutions = collect(_paper_available_resolutions(entry.data))
        selected_resolutions = [resolution
                                for resolution in available_resolutions
                                if resolution != :run_h2]
        required_runs = (h0_resolution, selected_resolutions...)
        missing_runs = _paper_missing_runs(entry.data, required_runs)
        if !isempty(missing_runs)
            @warn "Skipping all-resolution analytic error figure due to missing saved runs." path=entry.path missing_runs=missing_runs available_keys=sort(collect(keys(entry.data)))
            continue
        end
        isempty(selected_resolutions) && continue

        sim_h0 = _paper_run(entry.data, h0_resolution)
        h0 = _paper_grid_spacing(sim_h0)
        plotted = try
            sims = [_paper_run(entry.data, resolution)
                    for resolution in selected_resolutions]
            analytic_reference = _paper_analytic_reference(entry)
            _paper_all_resolution_analytic_error_figure(sims,
                                                        selected_resolutions;
                                                        family = family,
                                                        order = order,
                                                        time = case_time,
                                                        h0 = h0,
                                                        analytic_reference = analytic_reference)
        catch err
            @warn "Skipping all-resolution analytic error figure because the analytic reference is not valid for this case/time." path=entry.path boundary_condition=_paper_entry_boundary_condition(entry) error=sprint(showerror,
                                                                                                                                                                                                    err)
            continue
        end
        _paper_apply_typography!(plotted.fig)
        _paper_apply_analytic_error_typography!(plotted.fig)

        initial_data_stem = _paper_initial_data_stem(entry.metadata.initial_data_kind)
        run_token = _paper_energy_run_token(entry, Tuple(selected_resolutions))
        resolution_token = join(String.(selected_resolutions), "-")
        h0_token = "h0_from_$(h0_resolution)_$(_paper_number_token(h0))"
        cfl_token = _paper_cfl_token(entry, Tuple(selected_resolutions))
        case_out_dir = _paper_case_cfl_out_dir(out_dir, boundary_group, cfl_token)
        path = joinpath(case_out_dir,
                        "all_resolution_analytic_error_$(initial_data_stem)_$(family_file_tag)_order$(order)_$(run_token)_$(resolution_token)_$(h0_token)_t$(_paper_number_token(plotted.t)).$(file_ext)")
        save(path, plotted.fig)
        outputs[entry.path] = path
        println("Saved all-resolution analytic error figure: ", path)
    end

    return outputs
end

"""
    make_all_paper_plots(; kwargs...)

Load saved simulations and generate all paper plots:

1. spectra for diagonal/non-diagonal cases;
2. snapshots at `t=5` and `t=15`;
3. space-time heatmaps for `Π` and `Ψ`;
4. relative energy conservation plots;
5. Gundlach scaled-error plots at `t=15`;
6. convergence snapshot composites at `t=15`.
"""
function make_all_paper_plots(; sim_dir::AbstractString = PAPER_SIM_DIR,
                              prefer_legacy_sim_files::Bool = false,
                              out_dir::AbstractString = PAPER_PLOTS_DIR,
                              spectrum_points::Int = 31,
                              spectrum_R::Real = 30,
                              spectrum_p::Int = 2,
                              snapshot_times::Tuple{Vararg{<:Real}} = PAPER_SNAPSHOT_TIMES,
                              snapshot_resolution::Symbol = :run_h16,
                              energy_resolutions::Union{Nothing, Tuple{Vararg{Symbol}}} = nothing,
                              scaled_error_time::Real = 15.0,
                              scaled_error_resolutions::NTuple{3, Symbol} = PAPER_SCALED_ERROR_RESOLUTION_TRIPLET,
                              scaled_error_triplet_choice::Union{Nothing, Symbol} = nothing,
                              scaled_error_h0_resolution::Symbol = :run_h,
                              convergence_snapshot_time::Real = 15.0,
                              convergence_snapshot_resolutions::NTuple{3, Symbol} = PAPER_SCALED_ERROR_RESOLUTION_TRIPLET,
                              convergence_snapshot_triplet_choice::Union{Nothing, Symbol} = nothing,
                              convergence_snapshot_h0_resolution::Symbol = :run_h,
                              export_spectra_dir::Union{Nothing, AbstractString} = nothing)
    simulations = load_simulation_files(; sim_dir = sim_dir,
                                        prefer_legacy = prefer_legacy_sim_files)
    spectra = plot_spectra_for_paper(; out_dir = joinpath(out_dir, "spectra"),
                                     points = spectrum_points,
                                     R = spectrum_R,
                                     p = spectrum_p,
                                     export_plots_dir = export_spectra_dir)
    snapshots = plot_snapshots_for_paper(simulations;
                                         out_dir = joinpath(out_dir, "wave", "snapshots"),
                                         times = snapshot_times,
                                         resolution = snapshot_resolution)
    time_evolution = plot_time_evolution_for_paper(simulations;
                                                   out_dir = joinpath(dirname(out_dir),
                                                                      "time_evolution"),
                                                   resolution = snapshot_resolution)
    energy = plot_energy_for_paper(simulations;
                                   out_dir = joinpath(out_dir, "wave", "energy"),
                                   resolutions = energy_resolutions,
                                   difference_mode = :relative)
    scaled_errors = plot_scaled_errors_for_paper(simulations;
                                                 out_dir = joinpath(out_dir, "wave",
                                                                    "convergence"),
                                                 time = scaled_error_time,
                                                 resolutions = scaled_error_resolutions,
                                                 triplet_choice = scaled_error_triplet_choice,
                                                 h0_resolution = scaled_error_h0_resolution)
    convergence_snapshots = plot_convergence_snapshots_for_paper(simulations;
                                                                 out_dir = joinpath(out_dir,
                                                                                    "wave",
                                                                                    "convergence_snapshots"),
                                                                 time = convergence_snapshot_time,
                                                                 resolutions = convergence_snapshot_resolutions,
                                                                 triplet_choice = convergence_snapshot_triplet_choice,
                                                                 h0_resolution = convergence_snapshot_h0_resolution)

    return (simulations = simulations,
            spectra = spectra,
            snapshots = snapshots,
            time_evolution = time_evolution,
            energy = energy,
            scaled_errors = scaled_errors,
            convergence_snapshots = convergence_snapshots)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # make_all_paper_plots()
end
