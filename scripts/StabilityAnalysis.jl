using LinearAlgebra
using SparseArrays
using Printf
using SciMLBase: ODEProblem, solve
using OrdinaryDiffEqLowOrderRK: RK4
using CairoMakie
using LaTeXStrings

@inline function _normalize_boundary_condition(boundary_condition::Symbol)
    if boundary_condition === :radiative
        return :absorbing
    elseif boundary_condition === :reflective
        return :reflecting
    end
    return boundary_condition
end

function _sat_boundary_coefficients(BNN::Real, HNN::Real; bc::Symbol)
    bc_norm = _normalize_boundary_condition(bc)
    if bc_norm === :none
        return (
                apply = false,
                dPi_dPi = 0.0,
                dPi_dPsi = 0.0,
                dPsi_dPi = 0.0,
                dPsi_dPsi = 0.0
               )
    end

    HNN == 0 && throw(ArgumentError("`H[end,end]` must be nonzero when SAT boundary terms are enabled."))

    bnn = Float64(BNN)
    inv_hnn = 1.0 / Float64(HNN)
    c = -(bnn / 2.0) * inv_hnn

    if bc_norm === :absorbing
        return (
                apply = true,
                dPi_dPi = c,
                dPi_dPsi = c,
                dPsi_dPi = c,
                dPsi_dPsi = c
               )
    elseif bc_norm === :reflecting
        return (
                apply = true,
                dPi_dPi = 0.0,
                dPi_dPsi = -(bnn * inv_hnn),
                dPsi_dPi = 0.0,
                dPsi_dPsi = 0.0
               )
    elseif bc_norm === :dirichlet
        return (
                apply = true,
                dPi_dPi = 0.0,
                dPi_dPsi = 0.0,
                dPsi_dPi = -(bnn * inv_hnn),
                dPsi_dPsi = 0.0
               )
    end

    throw(ArgumentError("Unsupported boundary condition `$bc_norm`. Use :absorbing, :reflecting, :dirichlet, or :none."))
end

"""
    assemble_block_operator(D, G; H=nothing, B=nothing, boundary_condition=:none, enforce_origin=true)

Build the full first-order wave operator for state `[Pi; Psi]`:

    L = [0  D;
         G  0]

Optionally include outer-boundary SAT Jacobian terms and origin odd-symmetry RHS
constraint (`dPsi[1]=0`).
"""
function assemble_block_operator(D::AbstractMatrix,
                                G::AbstractMatrix;
                                H = nothing,
                                B = nothing,
                                boundary_condition::Symbol = :none,
                                enforce_origin::Bool = true)
    nD1, nD2 = size(D)
    nG1, nG2 = size(G)
    nD1 == nD2 || throw(DimensionMismatch("`D` must be square."))
    nG1 == nG2 || throw(DimensionMismatch("`G` must be square."))
    nD1 == nG1 || throw(DimensionMismatch("`D` and `G` must have the same size."))

    n = nD1
    Df = Matrix{Float64}(D)
    Gf = Matrix{Float64}(G)
    Z = zeros(Float64, n, n)
    L = [Z Df; Gf Z]

    bc_norm = _normalize_boundary_condition(boundary_condition)
    if bc_norm !== :none
        H === nothing && throw(ArgumentError("Pass `H` when SAT boundary terms are enabled."))
        B === nothing && throw(ArgumentError("Pass `B` when SAT boundary terms are enabled."))

        size(H, 1) == n == size(H, 2) || throw(DimensionMismatch("`H` must be $n x $n."))
        size(B, 1) == n == size(B, 2) || throw(DimensionMismatch("`B` must be $n x $n."))

        sat = _sat_boundary_coefficients(B[end, end], H[end, end]; bc = bc_norm)
        if sat.apply
            L[n, n] += sat.dPi_dPi
            L[n, 2 * n] += sat.dPi_dPsi
            L[2 * n, n] += sat.dPsi_dPi
            L[2 * n, 2 * n] += sat.dPsi_dPsi
        end
    end

    if enforce_origin
        L[n + 1, :] .= 0.0
    end

    return L
end

@inline rk4_stability_function(z) = 1 + z + z^2 / 2 + z^3 / 6 + z^4 / 24

function rk4_amplification_max(eigvals::AbstractVector{<:Complex}, dt::Real)
    return maximum(abs.(rk4_stability_function.(dt .* eigvals)))
end

"""
    rk4_dt_max_from_spectrum(eigvals; tol=1e-10)

Numerically estimate the largest stable fixed time step for classical RK4 for the
linear system `u_t = L u` with eigenvalues `eigvals`.
"""
function rk4_dt_max_from_spectrum(eigvals::AbstractVector{<:Complex}; tol::Float64 = 1e-10)
    isempty(eigvals) && return Inf
    all(abs.(eigvals) .== 0) && return Inf

    lo = 0.0
    hi = 1 / maximum(abs.(eigvals))

    while rk4_amplification_max(eigvals, hi) <= 1 + tol && hi < 1e9
        lo = hi
        hi *= 1.5
    end

    if hi >= 1e9 && rk4_amplification_max(eigvals, hi) <= 1 + tol
        return hi
    end

    for _ in 1:100
        mid = 0.5 * (lo + hi)
        if rk4_amplification_max(eigvals, mid) <= 1 + tol
            lo = mid
        else
            hi = mid
        end
    end

    return lo
end

"""
    spectral_analysis(D, G; H=nothing, B=nothing, boundary_condition=:none,
                      enforce_origin=true, growth_tol=1e-12, throw_on_instability=true)

Step 1:
- build `L` (including SAT if requested)
- compute full eigensystem
- assert Re(λ) <= growth_tol
- compute spectral radius and RK4 dt limits.
"""
function spectral_analysis(D::AbstractMatrix,
                           G::AbstractMatrix;
                           H = nothing,
                           B = nothing,
                           boundary_condition::Symbol = :none,
                           enforce_origin::Bool = true,
                           growth_tol::Float64 = 1e-12,
                           throw_on_instability::Bool = true)
    bc_norm = _normalize_boundary_condition(boundary_condition)
    L = assemble_block_operator(
                               D,
                               G;
                               H = H,
                               B = B,
                               boundary_condition = bc_norm,
                               enforce_origin = enforce_origin
                              )
    eig = eigen(L)
    eigvals = eig.values
    eigvecs = eig.vectors

    real_parts = real.(eigvals)
    unstable_indices = findall(real_parts .> growth_tol)
    max_real = isempty(real_parts) ? -Inf : maximum(real_parts)
    abs_eigs = abs.(eigvals)
    spectral_radius = isempty(abs_eigs) ? 0.0 : maximum(abs_eigs)
    spectral_radius_mode_index = isempty(abs_eigs) ? 0 : argmax(abs_eigs)
    spectral_radius_mode_eigenvalue = spectral_radius_mode_index == 0 ? (0.0 + 0.0im) : eigvals[spectral_radius_mode_index]

    dt_rk4_imag_axis = spectral_radius == 0 ? Inf : (2 * sqrt(2)) / spectral_radius
    dt_rk4_from_full_spectrum = rk4_dt_max_from_spectrum(eigvals)

    if throw_on_instability && !isempty(unstable_indices)
        error("Exponential instability detected: max Re(lambda) = $(max_real), unstable mode count = $(length(unstable_indices)).")
    end

    return (
            L = L,
            eigvals = eigvals,
            eigvecs = eigvecs,
            boundary_condition = bc_norm,
            max_real = max_real,
            unstable_indices = unstable_indices,
            spectral_radius = spectral_radius,
            spectral_radius_mode_index = spectral_radius_mode_index,
            spectral_radius_mode_eigenvalue = spectral_radius_mode_eigenvalue,
            rk4_dt_max_imag_axis = dt_rk4_imag_axis,
            rk4_dt_max_full_spectrum = dt_rk4_from_full_spectrum
           )
end

function zero_crossings(u::AbstractVector{<:Real}; atol::Float64 = 1e-12)
    n = length(u)
    n <= 1 && return 0

    zc = 0
    last_sign = 0
    @inbounds for i in 1:n
        ui = u[i]
        si = abs(ui) <= atol ? 0 : (ui > 0 ? 1 : -1)
        if si != 0
            if last_sign != 0 && si != last_sign
                zc += 1
            end
            last_sign = si
        end
    end

    return zc
end

function oscillation_metrics(u::AbstractVector{<:Real}; atol::Float64 = 1e-12)
    zc = zero_crossings(u; atol = atol)
    zc_ratio = length(u) <= 1 ? 0.0 : zc / (length(u) - 1)
    tv = sum(abs.(diff(u)))
    l1 = sum(abs.(u))
    tv_ratio = l1 <= atol ? 0.0 : tv / l1
    return (zero_crossings = zc, zc_ratio = zc_ratio, total_variation = tv, tv_ratio = tv_ratio)
end

function dominant_real_projection(v::AbstractVector{<:Complex})
    idx = argmax(abs.(v))
    theta = angle(v[idx])
    return real.(v .* exp(-im * theta))
end

"""
    nullspace_and_checkerboard(eigvals, eigvecs, n; ...)

Step 2:
- find near-zero eigenvalues
- extract null eigenvectors
- detect checkerboard-like high-frequency oscillations.
"""
function nullspace_and_checkerboard(eigvals::AbstractVector{<:Complex},
                                    eigvecs::AbstractMatrix{<:Complex},
                                    n::Int;
                                    null_tol::Float64 = 1e-11,
                                    checkerboard_zc_ratio::Float64 = 0.6,
                                    checkerboard_tv_ratio::Float64 = 1.0)
    null_indices = findall(abs.(eigvals) .<= null_tol)
    entries = NamedTuple[]

    for eig_idx in null_indices
        mode = dominant_real_projection(view(eigvecs, :, eig_idx))
        pi_mode = mode[1:n]
        psi_mode = mode[(n + 1):(2 * n)]

        pi_metrics = oscillation_metrics(pi_mode)
        psi_metrics = oscillation_metrics(psi_mode)

        pi_checker = (pi_metrics.zc_ratio >= checkerboard_zc_ratio) && (pi_metrics.tv_ratio >= checkerboard_tv_ratio)
        psi_checker = (psi_metrics.zc_ratio >= checkerboard_zc_ratio) && (psi_metrics.tv_ratio >= checkerboard_tv_ratio)
        is_spurious = pi_checker || psi_checker

        push!(
              entries,
              (
               eigen_index = eig_idx,
               eigenvalue = eigvals[eig_idx],
               mode = mode,
               pi_mode = pi_mode,
               psi_mode = psi_mode,
               pi_metrics = pi_metrics,
               psi_metrics = psi_metrics,
               is_spurious = is_spurious
              )
             )
    end

    spurious_mode_indices = [e.eigen_index for e in entries if e.is_spurious]
    return (
            null_indices = null_indices,
            entries = entries,
            spurious_mode_indices = spurious_mode_indices,
            has_spurious_modes = !isempty(spurious_mode_indices)
           )
end

function inward_packet_initial_data(r::AbstractVector{<:Real};
                                    amplitude::Float64 = 1.0,
                                    center = nothing,
                                    sigma = nothing,
                                    wavenumber = nothing)
    rf = Float64.(r)
    R = rf[end]
    dr = length(rf) >= 2 ? (rf[2] - rf[1]) : 1.0

    c = isnothing(center) ? 0.75 * R : Float64(center)
    s = isnothing(sigma) ? max(4dr, 0.06 * max(R, 1.0)) : Float64(sigma)
    k = isnothing(wavenumber) ? 0.35 * (pi / max(dr, 1e-12)) : Float64(wavenumber)

    envelope = exp.(-0.5 .* ((rf .- c) ./ s) .^ 2)
    carrier = cos.(k .* (rf .- c))
    pi0 = amplitude .* envelope .* carrier

    # Inward packet in this codebase convention: Pi = +Psi.
    psi0 = copy(pi0)
    psi0[1] = 0.0

    return (pi0 = pi0, psi0 = psi0, center = c, sigma = s, wavenumber = k)
end

function wave_rhs_vec!(du, u, p, t)
    _ = t
    n = p.n
    pi_vec = @view u[1:n]
    psi_vec = @view u[(n + 1):(2 * n)]
    dpi = @view du[1:n]
    dpsi = @view du[(n + 1):(2 * n)]

    mul!(dpi, p.D, psi_vec)
    mul!(dpsi, p.G, pi_vec)

    if p.sat.apply
        dpi[end] += p.sat.dPi_dPi * pi_vec[end] + p.sat.dPi_dPsi * psi_vec[end]
        dpsi[end] += p.sat.dPsi_dPi * pi_vec[end] + p.sat.dPsi_dPsi * psi_vec[end]
    end

    if p.enforce_origin
        dpsi[1] = 0.0
    end

    return nothing
end

function discrete_energy(H::AbstractMatrix, pi_vec::AbstractVector, psi_vec::AbstractVector)
    return 0.5 * (dot(pi_vec, H * pi_vec) + dot(psi_vec, H * psi_vec))
end

"""
    reflection_ibvp_test(D, G, H, r; kwargs...)

Step 3:
- inward high-frequency Gaussian packet
- RK4 time integration until origin reflection
- dispersion/phase-lag and trailing-noise diagnostics.
"""
function reflection_ibvp_test(D::AbstractMatrix,
                              G::AbstractMatrix,
                              H::AbstractMatrix,
                              r::AbstractVector{<:Real};
                              B = nothing,
                              boundary_condition::Symbol = :absorbing,
                              enforce_origin::Bool = true,
                              p::Int = 2,
                              dt::Union{Nothing, Float64} = nothing,
                              rk4_dt_limit::Union{Nothing, Float64} = nothing,
                              dt_safety::Float64 = 0.6,
                              t_final::Union{Nothing, Float64} = nothing,
                              save_every::Int = 1,
                              max_steps::Int = 200_000,
                              max_saved_steps::Int = 2_000,
                              amplitude::Float64 = 1.0,
                              center = nothing,
                              sigma = nothing,
                              wavenumber = nothing)
    n = length(r)
    n >= 3 || throw(ArgumentError("Need at least 3 grid points for reflection analysis."))

    Df = Matrix{Float64}(D)
    Gf = Matrix{Float64}(G)
    Hf = Matrix{Float64}(H)
    rf = Float64.(r)
    bc_norm = _normalize_boundary_condition(boundary_condition)

    sat = if bc_norm === :none
        _sat_boundary_coefficients(1.0, 1.0; bc = :none)
    else
        B === nothing && throw(ArgumentError("Pass `B` when `boundary_condition != :none`."))
        Bf = Matrix{Float64}(B)
        _sat_boundary_coefficients(Bf[end, end], Hf[end, end]; bc = bc_norm)
    end

    ic = inward_packet_initial_data(
                                    rf;
                                    amplitude = amplitude,
                                    center = center,
                                    sigma = sigma,
                                    wavenumber = wavenumber
                                   )

    if isnothing(rk4_dt_limit)
        spec = spectral_analysis(
                                 Df,
                                 Gf;
                                 H = Hf,
                                 B = B,
                                 boundary_condition = bc_norm,
                                 enforce_origin = enforce_origin,
                                 throw_on_instability = false
                                )
        rk4_dt_limit = spec.rk4_dt_max_full_spectrum
    end

    dt_use = isnothing(dt) ? dt_safety * rk4_dt_limit : dt
    dt_use > 0 || throw(ArgumentError("Time step must be positive."))

    R = rf[end]
    t_hit = ic.center
    t_end = isnothing(t_final) ? min(1.5 * t_hit, 0.9 * (2R - ic.center)) : t_final
    t_end > 0 || throw(ArgumentError("`t_final` must be positive."))

    nsteps = max(1, ceil(Int, t_end / dt_use))
    nsteps <= max_steps || throw(ArgumentError("Reflection test requires $nsteps steps (> max_steps=$max_steps). Increase `dt`, decrease `t_final`, or increase `max_steps`."))
    dt_use = t_end / nsteps
    save_every > 0 || throw(ArgumentError("`save_every` must be positive."))
    max_saved_steps > 0 || throw(ArgumentError("`max_saved_steps` must be positive."))
    save_stride = max(Int(save_every), ceil(Int, nsteps / max_saved_steps))

    save_indices = collect(0:save_stride:nsteps)
    if isempty(save_indices) || save_indices[end] != nsteps
        push!(save_indices, nsteps)
    end
    saveat = dt_use .* Float64.(save_indices)

    u0 = vcat(ic.pi0, ic.psi0)
    params = (D = Df, G = Gf, n = n, enforce_origin = enforce_origin, sat = sat)
    prob = ODEProblem(wave_rhs_vec!, u0, (0.0, t_end), params)

    sol = solve(prob, RK4(); adaptive = false, dt = dt_use, saveat = saveat, verbose = false)

    nsave = length(sol.t)
    pi_hist = Matrix{Float64}(undef, n, nsave)
    psi_hist = Matrix{Float64}(undef, n, nsave)
    energy = Vector{Float64}(undef, nsave)

    for (k, uk) in enumerate(sol.u)
        pi_hist[:, k] .= @view uk[1:n]
        psi_hist[:, k] .= @view uk[(n + 1):(2 * n)]
        energy[k] = discrete_energy(Hf, @view(pi_hist[:, k]), @view(psi_hist[:, k]))
    end

    t_probe = clamp(t_hit + 0.25 * R, sol.t[1], sol.t[end])
    idx_probe = argmin(abs.(sol.t .- t_probe))
    t_probe_actual = sol.t[idx_probe]

    pi_probe = @view pi_hist[:, idx_probe]
    peak_idx = argmax(abs.(pi_probe))
    r_peak = rf[peak_idx]
    r_expected = abs(ic.center - t_probe_actual)
    phase_lag = r_peak - r_expected

    core_half_width = 3 * ic.sigma
    core_mask = abs.(rf .- r_peak) .<= core_half_width
    tail_mask = .!core_mask
    trailing_noise_ratio = norm(pi_probe[tail_mask]) / max(norm(pi_probe), eps(Float64))

    energy_drift_abs = maximum(energy) - minimum(energy)
    energy_drift_rel = energy_drift_abs / max(abs(energy[1]), eps(Float64))

    return (
            t = Float64.(sol.t),
            pi = pi_hist,
            psi = psi_hist,
            energy = energy,
            dt = dt_use,
            nsteps = nsteps,
            ic = ic,
            reflection_metrics = (
                                  t_hit = t_hit,
                                  t_probe = t_probe_actual,
                                  r_peak = r_peak,
                                  r_expected = r_expected,
                                  phase_lag = phase_lag,
                                  trailing_noise_ratio = trailing_noise_ratio,
                                  energy_drift_abs = energy_drift_abs,
                                  energy_drift_rel = energy_drift_rel
                                 ),
            boundary_condition = bc_norm,
            p = p
           )
end

function save_spectrum_plot(eigvals::AbstractVector{<:Complex}, path::AbstractString)
    fig = Figure(size = (760, 560))
    ax = Axis(fig[1, 1],
              xlabel = L"\mathrm{Re}(\lambda)",
              ylabel = L"\mathrm{Im}(\lambda)",
              title = L"\mathrm{Eigenvalue\ Spectrum}")
    scatter!(ax, real.(eigvals), imag.(eigvals); markersize = 6, color = (:dodgerblue, 0.7))
    vlines!(ax, [0.0]; linestyle = :dash, color = :black, linewidth = 2)
    save(path, fig)
    return path
end

function save_nullspace_plot(null_report, path::AbstractString; max_modes::Int = 6)
    if isempty(null_report.entries)
        fig = Figure(size = (760, 240))
        ax = Axis(fig[1, 1],
                  title = L"\mathrm{Null\ Space\ Modes}",
                  xlabel = L"i",
                  ylabel = L"\mathrm{value}")
        text!(ax, 0.5, 0.5, text = L"\mathrm{No\ near\text{-}zero\ modes\ detected}", space = :relative, align = (:center, :center))
        save(path, fig)
        return path
    end

    nm = min(max_modes, length(null_report.entries))
    fig = Figure(size = (980, 220 * nm))

    for k in 1:nm
        entry = null_report.entries[k]
        mode_title = latexstring("\\mathrm{mode}\\ ", entry.eigen_index,
                                 "\\;|\\;\\mathrm{spurious}=",
                                 entry.is_spurious ? "1" : "0")
        ax = Axis(fig[k, 1],
                  xlabel = L"i_r",
                  ylabel = L"\mathrm{amplitude}",
                  title = mode_title)
        lines!(ax, entry.pi_mode; color = :royalblue, linewidth = 2, label = L"\Pi")
        lines!(ax, entry.psi_mode; color = :darkorange, linewidth = 2, linestyle = :dash, label = L"\Psi")
        axislegend(ax; position = :rt)
    end

    save(path, fig)
    return path
end

function save_reflection_heatmap(reflection_report, path::AbstractString)
    fig = Figure(size = (900, 560))
    ax = Axis(fig[1, 1], xlabel = L"t", ylabel = L"i_r", title = L"\mathrm{Wave\ Packet\ Reflection}")
    hm = heatmap!(ax, reflection_report.t, 1:size(reflection_report.pi, 1), reflection_report.pi')
    Colorbar(fig[1, 2], hm, label = L"\Pi(r,t)")
    save(path, fig)
    return path
end

function save_energy_trace(reflection_report, path::AbstractString)
    fig = Figure(size = (760, 420))
    ax = Axis(fig[1, 1], xlabel = L"t", ylabel = L"E", title = L"\mathrm{Discrete\ Energy}")
    lines!(ax, reflection_report.t, reflection_report.energy; linewidth = 2, color = :black)
    save(path, fig)
    return path
end

function save_dashboard(report, path::AbstractString)
    fig = Figure(size = (1600, 1000))

    # Spectrum
    ax1 = Axis(fig[1, 1],
               xlabel = L"\mathrm{Re}(\lambda)",
               ylabel = L"\mathrm{Im}(\lambda)",
               title = L"\mathrm{Spectrum}")
    scatter!(ax1, real.(report.spectral.eigvals), imag.(report.spectral.eigvals); markersize = 5, color = (:dodgerblue, 0.7))
    vlines!(ax1, [0.0]; linestyle = :dash, color = :black, linewidth = 2)

    # Null-space representative mode (first, if present)
    ax2 = Axis(fig[1, 2], xlabel = L"i_r", ylabel = L"\mathrm{mode\ amplitude}", title = L"\mathrm{Null\ Space\ Modes}")
    if isempty(report.nullspace.entries)
        text!(ax2, 0.5, 0.5, text = L"\mathrm{No\ near\text{-}zero\ modes}", space = :relative, align = (:center, :center))
    else
        mode = report.nullspace.entries[1]
        lines!(ax2, mode.pi_mode; color = :royalblue, linewidth = 2, label = L"\Pi")
        lines!(ax2, mode.psi_mode; color = :darkorange, linewidth = 2, linestyle = :dash, label = L"\Psi")
        axislegend(ax2; position = :rt)
    end

    # Reflection heatmap
    ax3 = Axis(fig[2, 1], xlabel = L"t", ylabel = L"i_r", title = L"\mathrm{Reflection\ Waterfall}")
    hm = heatmap!(ax3, report.reflection.t, 1:size(report.reflection.pi, 1), report.reflection.pi')
    Colorbar(fig[2, 2], hm, label = L"\Pi(r,t)")

    # Energy trace in inset axis
    ax4 = Axis(fig[2, 2], xlabel = L"t", ylabel = L"E", title = L"\mathrm{Energy}")
    lines!(ax4, report.reflection.t, report.reflection.energy; color = :black, linewidth = 2)

    save(path, fig)
    return path
end

function save_reflection_animation(reflection_report, r, path::AbstractString)
    rf = Float64.(r)
    fig = Figure(size = (900, 500))
    ax = Axis(fig[1, 1], xlabel = L"r", ylabel = L"\Pi(r,t)", title = L"\mathrm{Reflection\ Animation}")

    y = Observable(copy(reflection_report.pi[:, 1]))
    lines!(ax, rf, y; color = :royalblue, linewidth = 2)

    record(fig, path, 1:length(reflection_report.t); framerate = 20) do k
        y[] = reflection_report.pi[:, k]
        ax.title = latexstring("\\Pi(r,t),\\ t=", @sprintf("%.4f", reflection_report.t[k]))
    end

    return path
end

"""
    run_stability_suite(; kwargs...)

Full four-step analysis pipeline.

Inputs:
- either pass `ops` with fields `Geven`, `D`, `H`, `B`, `r`, `p`,
- or pass `G`, `D`, `H`, `B`, `r` directly.
"""
function run_stability_suite(; ops = nothing,
                             G = nothing,
                             D = nothing,
                             H = nothing,
                             B = nothing,
                             r = nothing,
                             p::Int = 2,
                             boundary_condition::Symbol = :absorbing,
                             enforce_origin::Bool = true,
                             growth_tol::Float64 = 1e-12,
                             null_tol::Float64 = 1e-11,
                             checkerboard_zc_ratio::Float64 = 0.6,
                             checkerboard_tv_ratio::Float64 = 1.0,
                             throw_on_instability::Bool = true,
                             reflection_save_every::Int = 1,
                             reflection_max_steps::Int = 200_000,
                             reflection_max_saved_steps::Int = 2_000,
                             make_plots::Bool = true,
                             make_animation::Bool = false,
                             output_dir::AbstractString = "plots/stability")
    if ops !== nothing
        G = getproperty(ops, :Geven)
        D = getproperty(ops, :D)
        H = getproperty(ops, :H)
        B = getproperty(ops, :B)
        r = getproperty(ops, :r)
        if hasproperty(ops, :p)
            p = Int(getproperty(ops, :p))
        end
    end

    G === nothing && throw(ArgumentError("Pass `G` or `ops`."))
    D === nothing && throw(ArgumentError("Pass `D` or `ops`."))
    H === nothing && throw(ArgumentError("Pass `H` or `ops`."))
    r === nothing && throw(ArgumentError("Pass `r` or `ops`."))

    n = size(D, 1)
    size(G, 1) == n == size(H, 1) || throw(DimensionMismatch("`D`, `G`, `H` must have matching sizes."))

    spectral = spectral_analysis(
                                 D,
                                 G;
                                 H = H,
                                 B = B,
                                 boundary_condition = boundary_condition,
                                 enforce_origin = enforce_origin,
                                 growth_tol = growth_tol,
                                 throw_on_instability = throw_on_instability
                                )

    nullspace = nullspace_and_checkerboard(
                                           spectral.eigvals,
                                           spectral.eigvecs,
                                           n;
                                           null_tol = null_tol,
                                           checkerboard_zc_ratio = checkerboard_zc_ratio,
                                           checkerboard_tv_ratio = checkerboard_tv_ratio
                                          )

    reflection = reflection_ibvp_test(
                                      D,
                                      G,
                                      H,
                                      r;
                                      B = B,
                                      boundary_condition = boundary_condition,
                                      enforce_origin = enforce_origin,
                                      p = p,
                                      rk4_dt_limit = spectral.rk4_dt_max_full_spectrum,
                                      save_every = reflection_save_every,
                                      max_steps = reflection_max_steps,
                                      max_saved_steps = reflection_max_saved_steps
                                     )

    plot_files = Dict{Symbol, String}()

    if make_plots
        mkpath(output_dir)

        spectrum_path = joinpath(output_dir, "eigen_spectrum.png")
        null_path = joinpath(output_dir, "nullspace_modes.png")
        heat_path = joinpath(output_dir, "reflection_waterfall.png")
        energy_path = joinpath(output_dir, "energy_trace.png")
        dashboard_path = joinpath(output_dir, "stability_dashboard.png")

        save_spectrum_plot(spectral.eigvals, spectrum_path)
        save_nullspace_plot(nullspace, null_path)
        save_reflection_heatmap(reflection, heat_path)
        save_energy_trace(reflection, energy_path)
        save_dashboard((; spectral, nullspace, reflection), dashboard_path)

        plot_files[:spectrum] = spectrum_path
        plot_files[:nullspace] = null_path
        plot_files[:waterfall] = heat_path
        plot_files[:energy] = energy_path
        plot_files[:dashboard] = dashboard_path

        if make_animation
            mp4_path = joinpath(output_dir, "reflection_animation.mp4")
            save_reflection_animation(reflection, r, mp4_path)
            plot_files[:animation] = mp4_path
        end
    end

    return (
            spectral = spectral,
            nullspace = nullspace,
            reflection = reflection,
            plots = plot_files
           )
end

function print_suite_summary(report)
    @printf("\n=== Stability Suite Summary ===\n")
    @printf("boundary condition in spectrum: %s\n", String(report.spectral.boundary_condition))
    @printf("max Re(lambda): %.6e\n", report.spectral.max_real)
    @printf("spectral radius rho(L): %.6e\n", report.spectral.spectral_radius)
    @printf("rho(L) mode index: %d\n", report.spectral.spectral_radius_mode_index)
    @printf("rho(L) eigenvalue: %.6e %+.6ei\n",
            real(report.spectral.spectral_radius_mode_eigenvalue),
            imag(report.spectral.spectral_radius_mode_eigenvalue))
    @printf("RK4 dt max (imag-axis estimate): %.6e\n", report.spectral.rk4_dt_max_imag_axis)
    @printf("RK4 dt max (full spectrum): %.6e\n", report.spectral.rk4_dt_max_full_spectrum)

    @printf("null modes (|lambda|<=tol): %d\n", length(report.nullspace.null_indices))
    @printf("spurious null modes flagged: %d\n", length(report.nullspace.spurious_mode_indices))

    m = report.reflection.reflection_metrics
    @printf("phase lag at reflection probe: %.6e\n", m.phase_lag)
    @printf("trailing noise ratio: %.6e\n", m.trailing_noise_ratio)
    @printf("relative energy drift: %.6e\n", m.energy_drift_rel)

    if !isempty(report.plots)
        @printf("plots written to:\n")
        for (k, v) in sort(collect(report.plots); by = x -> string(x[1]))
            @printf("  %-12s %s\n", string(k) * ":", v)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    using SphericalSBPOperators
    using SummationByPartsOperators: MattssonNordström2004, SafeMode

    source = MattssonNordström2004()
    ops = spherical_operators(
                              source;
                              accuracy_order = 8,
                              N = 24,
                              R = 1.0,
                              p = 2,
                              mode = SafeMode()
                             )

    report = run_stability_suite(
                                 ;
                                 ops = ops,
                                 boundary_condition = :absorbing,
                                 growth_tol = 1e-12,
                                 null_tol = 1e-11,
                                 throw_on_instability = false,
                                 reflection_max_saved_steps = 800,
                                 output_dir = "plots/stability"
                                )
    print_suite_summary(report)
end
