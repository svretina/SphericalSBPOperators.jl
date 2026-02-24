using Test
using SphericalSBPOperators
using SparseArrays: sparse
using SummationByPartsOperators: MattssonNordström2004, SafeMode

println("Starting tests")
ti = time()

source = MattssonNordström2004()

@testset "Float64 acceptance matrix" begin
    for acc in (2, 4, 6, 8), p in (1, 2)
        ops = spherical_operators(
                                  source;
                                  accuracy_order = acc,
                                  N = 64,
                                  R = 1.0,
                                  p = p,
                                  build_matrix = :matrix_if_square
                                 )

        @test size(ops.Geven) == (65, 65)
        @test size(ops.Godd) == (65, 65)
        @test size(ops.D) == (65, 65)
        @test size(ops.H) == (65, 65)
        @test ops.r[1] == 0.0
        @test ops.r[end] == 1.0

        report = validate(ops; max_monomial_degree = acc, verbose = false)
        @test report.diagnostics.Nh == 65
        @test report.diagnostics.M == 129
        @test report.sbp.sbp_no_origin <= 1e-10
        @test abs((ops.D * ops.r)[1] - (p + 1)) <= 1e-10
        @test haskey(report, :quadrature)
        @test haskey(report, :gradient_even)
        @test haskey(report, :divergence_odd)
        @test haskey(report, :sbp)
        @test haskey(report, :diagnostics)
    end
end

@testset "Probe and matrix extraction agree" begin
    ops_probe = spherical_operators(
                                    source;
                                    accuracy_order = 4,
                                    N = 24,
                                    R = 1.0,
                                    p = 2,
                                    build_matrix = :probe
                                   )
    ops_matrix = spherical_operators(
                                     source;
                                     accuracy_order = 4,
                                     N = 24,
                                     R = 1.0,
                                     p = 2,
                                     build_matrix = :matrix_if_square
                                    )
    @test Matrix(ops_probe.Geven) ≈ Matrix(ops_matrix.Geven)
    @test Matrix(ops_probe.Godd) ≈ Matrix(ops_matrix.Godd)
    @test Matrix(ops_probe.D) ≈ Matrix(ops_matrix.D)
    @test Matrix(ops_probe.H) ≈ Matrix(ops_matrix.H)
end

@testset "Rational exactness mode (small/medium N)" begin
    for N in (8, 16, 24), acc in (2, 4, 6, 8), p in (1, 2)
        ops = spherical_operators(
                                  source;
                                  accuracy_order = acc,
                                  N = N,
                                  R = 1 // 1,
                                  p = p,
                                  mode = SafeMode(),
                                  build_matrix = :probe
                                 )

        @test eltype(ops.r) <: Rational
        @test eltype(ops.H.nzval) <: Rational
        @test eltype(ops.Geven.nzval) <: Rational
        @test eltype(ops.D.nzval) <: Rational

        report = validate(ops; max_monomial_degree = acc, verbose = false)
        @test report.sbp.sbp_no_origin == 0
        @test (ops.D * ops.r)[1] == p + 1
        @test report.diagnostics.r0 == 0
        @test report.diagnostics.r0_ok
    end
end

@testset "Canonical build and scaling helpers" begin
    ops_canonical = spherical_operators(
                                        source;
                                        accuracy_order = 8,
                                        N = 32,
                                        R = 1.0,
                                        p = 2,
                                        mode = SafeMode(),
                                        build_matrix = :probe,
                                        return_canonical = true
                                       )
    @test eltype(ops_canonical.r) == Rational{BigInt}
    @test ops_canonical.r[2] - ops_canonical.r[1] == 1 // 1
    @test ops_canonical.r[end] == 32 // 1
    @test all(ops_canonical.Geven[i, 1] == 0 for i in 2:ops_canonical.Nh)
    @test ops_canonical.D * ops_canonical.r == fill(3 // 1, ops_canonical.Nh)

    # Raw folded matrix has four contaminated rows for acc=8.
    Dfull, xfull, Gfull, _ = SphericalSBPOperators._build_full_grid_objects(
                                                                     source;
                                                                     accuracy_order = 8,
                                                                     N = 32,
                                                                     R = big(32) // 1,
                                                                     mode = SafeMode(),
                                                                     build_matrix = :probe
                                                                    )
    r_raw, Rop, Eeven, _ = SphericalSBPOperators._build_folding_operators(
                                                                   xfull;
                                                                   atol = zero(eltype(xfull))
                                                                  )
    Gfold_even_raw = sparse(Rop * Gfull * Eeven)
    rows_to_solve = SphericalSBPOperators._rows_with_nonzero_first_column(
                                                                         Gfold_even_raw;
                                                                         atol = zero(eltype(r_raw))
                                                                        )
    cartesian_half_bandwidth = max(length(Dfull.coefficients.lower_coef), length(Dfull.coefficients.upper_coef))
    @test rows_to_solve == collect(2:(cartesian_half_bandwidth + 1))
    @test rows_to_solve == [2, 3, 4, 5]

    ops_scaled = scale_spherical_operators(ops_canonical, 1.0; target_eltype = Float64)
    @test ops_scaled.r[end] ≈ 1.0
    @test ops_scaled.r[2] - ops_scaled.r[1] ≈ 1 / 32
    @test maximum(abs.(ops_scaled.Geven[2:end, 1])) <= 1e-12
end

@testset "Parity helpers" begin
    u = [1.0, 2.0, 3.0]
    @test !check_odd(u; tol = 1e-12)
    enforce_odd!(u)
    @test check_odd(u; tol = 1e-12)
    @test u[1] == 0.0
end

@testset "Diagnostics suite" begin
    ops = spherical_operators(
                              source;
                              accuracy_order = 4,
                              N = 24,
                              R = 1.0,
                              p = 2,
                              build_matrix = :probe
                             )
    report = validate(ops; max_monomial_degree = 4, verbose = false)
    diag = diagnose(ops, report; Ktest = 6, region_points = 8, table_points = 4, verbose = false)
    g4 = first(filter(x -> x.degree == 4, report.gradient_even))
    d1 = first(filter(x -> x.degree == 1, report.divergence_odd))

    @test diag.grid.M == 49
    @test diag.grid.Nh == 25
    @test diag.grid.grid_ok
    @test diag.folding.rop_ok
    @test diag.folding.even_even_ok
    @test diag.folding.odd_signed_ok
    @test diag.folding.odd_matches_xk_for_odd_k_ok
    @test diag.operator_consistency.geven_diff_max <= 1e-10
    @test diag.operator_consistency.godd_diff_max <= 1e-10
    @test report.diagnostics.closure_width_right > 0
    @test g4.max_error_safe < 1e-10
    @test d1.max_error_safe < 1e-10
    @test g4.max_error_near_boundary ≈ g4.max_error
    @test d1.max_error_near_boundary <= d1.max_error
    @test diag.closure.safe_count > 0
    @test !diag.polynomial.skipped
    @test !isempty(diag.polynomial.gradient.entries)
    @test !isempty(diag.fullgrid_comparison.entries)
    @test diag.sbp.max_no_origin <= 1e-10
    @test !isempty(diag.interpretation.conclusions)
end

@testset "Wave SAT boundary and symmetry tests" begin
    ops = spherical_operators(
                              source;
                              accuracy_order = 4,
                              N = 32,
                              R = 1.0,
                              p = 2,
                              build_matrix = :matrix_if_square
                             )

    # Outgoing initial packet: Π0 = -Ξ0.
    ϕ = default_wave_profile(ops.r; amplitude = 1.0, center = 0.75, width = 0.06)
    Ξ0 = Vector{Float64}(ops.Geven * ϕ)
    Π0 = -copy(Ξ0)

    sol_abs = solve_wave_ode(
                             ops;
                             T_final = 0.12,
                             boundary_condition = :absorbing,
                             ϕ0 = ϕ,
                             Π0 = Π0,
                             Ξ0 = Ξ0,
                             save_every = 1,
                             verbose = false
                            )

    @test sol_abs.t[1] == 0.0
    @test sol_abs.t[end] ≈ 0.12 atol = 1e-12
    @test size(sol_abs.Π, 1) == length(ops.r)
    @test size(sol_abs.Ξ, 1) == length(ops.r)
    @test size(sol_abs.Π, 2) == length(sol_abs.t)
    @test size(sol_abs.Ξ, 2) == length(sol_abs.t)
    @test all(isfinite, sol_abs.Π)
    @test all(isfinite, sol_abs.Ξ)
    @test all(isfinite, sol_abs.energy)
    @test maximum(abs.(sol_abs.Ξ[1, :])) == 0.0
    @test minimum(sol_abs.energy) >= -1e-10
    @test sol_abs.pi === sol_abs.Π
    @test sol_abs.xi === sol_abs.Ξ
    @test sol_abs.initial_data_check.consistent
    @test sol_abs.initial_data_check.origin_ok
    @test !sol_abs.initial_data_check.require_boundary
    @test sol_abs.boundary_condition == :absorbing

    # Absorbing BC: energy should be non-increasing up to time-integration noise.
    dE_abs = diff(sol_abs.energy)
    @test maximum(dE_abs) <= 2e-6
    @test sol_abs.energy[end] <= sol_abs.energy[1] + 2e-6

    # Absorbing residual should stay small for outgoing packet.
    w_in_abs = sol_abs.Π[end, :] .+ sol_abs.Ξ[end, :]
    w_out_abs = sol_abs.Π[end, :] .- sol_abs.Ξ[end, :]
    @test maximum(abs.(w_in_abs)) <= 0.3 * maximum(abs.(w_out_abs)) + 5e-4

    # Reflecting SAT BC (energy-conserving characteristic reflection).
    sol_ref = solve_wave_ode(
                             ops;
                             T_final = 0.12,
                             boundary_condition = :reflecting,
                             ϕ0 = ϕ,
                             Π0 = Π0,
                             Ξ0 = Ξ0,
                             save_every = 1,
                             verbose = false
                            )
    rel_energy_drift = maximum(abs.(sol_ref.energy .- sol_ref.energy[1])) / max(abs(sol_ref.energy[1]), 1e-14)
    @test rel_energy_drift <= 2e-4
    w_reflect_residual = sol_ref.Π[end, :] .+ sol_ref.Ξ[end, :] .- (sol_ref.Π[end, :] .- sol_ref.Ξ[end, :])
    @test all(isfinite, w_reflect_residual)
    @test maximum(abs.(w_reflect_residual)) <= 1.0

    # Parity invariance check: disabling origin enforcement permits drift/nonzero origin.
    Ξ0_bad = copy(Ξ0)
    Ξ0_bad[1] = 1e-4
    sol_no_enforce = solve_wave_ode(
                                    ops;
                                    T_final = 0.03,
                                    boundary_condition = :none,
                                    ϕ0 = ϕ,
                                    Π0 = Π0,
                                    Ξ0 = Ξ0_bad,
                                    enforce_origin = false,
                                    save_every = 1,
                                    verbose = false
                                   )
    @test maximum(abs.(sol_no_enforce.Ξ[1, :])) >= 1e-8
end

@testset "Wave initial-data modes and potential consistency" begin
    ops = spherical_operators(
                              source;
                              accuracy_order = 4,
                              N = 32,
                              R = 1.0,
                              p = 2,
                              build_matrix = :matrix_if_square
                             )
    r = Float64.(ops.r)

    ϕ_bump = bumpb_profile(r; amplitude = 1.0, center = 0.0, radius = 0.35)
    Π_zero = zeros(Float64, length(r))
    sol_pot = solve_wave_ode(
                             ops;
                             T_final = 0.02,
                             dt = 0.01,
                             boundary_condition = :none,
                             initial_data_mode = :potential,
                             ϕ0 = ϕ_bump,
                             Π0 = Π_zero,
                             verbose = false
                            )
    Ξ_expected = Vector{Float64}(ops.Geven * ϕ_bump)
    Ξ_expected[1] = 0.0
    @test sol_pot.initial_data_check.mode == :potential
    @test sol_pot.initial_data_check.potential.residual_ok
    @test maximum(abs.(sol_pot.Ξ[:, 1] .- Ξ_expected)) <= 1e-10

    Π_bump = bumpb_profile(r; amplitude = 1.0, center = 0.0, radius = 0.35)
    Ξ_zero = zeros(Float64, length(r))
    pcheck = check_potential_consistency(ops, Π_bump, Ξ_zero; tol = 1e-10)
    @test pcheck.xi_growth_expected_from_pi
    @test pcheck.growth_explanation == "Ξ appears immediately if Π0 varies because Ξ_t = G Π."
    @test any(contains("This initial state is inconsistent with Ξ=φ_r for any scalar φ"), pcheck.warnings)

    w_in0 = ones(Float64, length(r))
    w_out0 = -ones(Float64, length(r))
    char = characteristic_initial_data(r; w_in0 = w_in0, w_out0 = w_out0)
    @test maximum(abs.(char.Π0[2:end])) <= 1e-12
    @test maximum(abs.(char.Ξ0[2:end] .- 1.0)) <= 1e-12
    @test char.Ξ0[1] == 0.0
end

@testset "Provided dt stability guard" begin
    ops = spherical_operators(
                              source;
                              accuracy_order = 4,
                              N = 24,
                              R = 1.0,
                              p = 2,
                              build_matrix = :matrix_if_square
                             )

    @test_throws ArgumentError solve_wave_ode(
                                             ops;
                                             T_final = 1.0,
                                             dt = 1.0,
                                             boundary_condition = :absorbing,
                                             save_every = 1,
                                             verbose = false
                                            )
end

@testset "Wave interior convergence sanity" begin
    function run_case(N)
        opsN = spherical_operators(
                                  source;
                                  accuracy_order = 4,
                                  N = N,
                                  R = 1.0,
                                  p = 2,
                                  build_matrix = :matrix_if_square
                                 )
        ϕN = default_wave_profile(opsN.r; amplitude = 1.0, center = 0.35, width = 0.08)
        ΞN = Vector{Float64}(opsN.Geven * ϕN)
        ΠN = zeros(Float64, length(opsN.r))
        solN = solve_wave_ode(
                              opsN;
                              T_final = 0.03,
                              boundary_condition = :none,
                              ϕ0 = ϕN,
                              Π0 = ΠN,
                              Ξ0 = ΞN,
                              safety_factor = 0.5,
                              save_every = 10,
                              verbose = false
                             )
        return opsN, solN
    end

    _, sol16 = run_case(16)
    _, sol32 = run_case(32)
    _, sol64 = run_case(64)

    Π16 = sol16.Π[:, end]
    Π32 = sol32.Π[1:2:end, end]
    Π32f = sol32.Π[:, end]
    Π64 = sol64.Π[1:2:end, end]

    n16 = length(Π16)
    n32 = length(Π32f)
    idx16 = 2:(n16 - 2)
    idx32 = 2:(n32 - 2)

    err16_32 = maximum(abs.(Π16[idx16] .- Π32[idx16]))
    err32_64 = maximum(abs.(Π32f[idx32] .- Π64[idx32]))

    @test err32_64 < 0.8 * err16_32
end

@testset "ImplicitMidpoint vector-state path" begin
    ops = spherical_operators(
                              source;
                              accuracy_order = 6,
                              N = 24,
                              R = 2.4,
                              p = 2,
                              build_matrix = :matrix_if_square
                             )
    n = length(ops.r)
    Π0 = exp.(-0.5 .* (Float64.(ops.r) .^ 2))
    Ξ0 = zeros(Float64, n)

    p = WaveODEParams(ops; boundary_condition = :reflecting, enforce_origin = true)
    U = vcat(Π0, Ξ0)
    dU = similar(U)
    wave_system_ode_vec!(dU, U, p, 0.0)
    @test length(dU) == 2 * n
    @test dU[n + 1] == 0.0

    Jproto = wave_system_jac_prototype(ops; boundary_condition = :reflecting)
    wave_system_jac!(Jproto, U, p, 0.0)
    @test size(Jproto) == (2 * n, 2 * n)
    @test maximum(abs.(Array(@view Jproto[n + 1, :]))) == 0.0

    sol = solve_wave_ode(
                         ops;
                         T_final = 0.1,
                         dt = 0.01,
                         boundary_condition = :reflecting,
                         alg = ImplicitMidpoint(),
                         Π0 = Π0,
                         Ξ0 = Ξ0,
                         save_every = 1,
                         verbose = false
                        )
    @test size(sol.Π, 1) == n
    @test size(sol.Ξ, 1) == n
    @test all(isfinite, sol.energy)
end

@testset "Integrator microbenchmark helper" begin
    ops = spherical_operators(
                              source;
                              accuracy_order = 4,
                              N = 20,
                              R = 2.0,
                              p = 2,
                              build_matrix = :matrix_if_square
                             )
    Π0 = exp.(-0.5 .* (Float64.(ops.r) .^ 2))
    Ξ0 = zeros(Float64, length(ops.r))
    bench = benchmark_wave_integrators(
                                        ops;
                                        T_final = 0.1,
                                        dt = 0.01,
                                        boundary_condition = :reflecting,
                                        Π0 = Π0,
                                        Ξ0 = Ξ0,
                                        warmup = false,
                                        verbose = false
                                       )
    @test haskey(bench, :rk4)
    @test haskey(bench, :implicit_midpoint)
    @test bench.rk4.nsteps == 10
    @test bench.implicit_midpoint.nsteps == 10
end

@testset "Reflecting energy bump diagnostics" begin
    rep = diagnose_reflecting_energy_bump(
                                          source = source,
                                          accuracy_order = 6,
                                          N = 16,
                                          R = 2.0,
                                          p = 2,
                                          dt = 0.02,
                                          T_final = 0.4,
                                          K = 10,
                                          verbose = false
                                         )
    @test haskey(rep, :integrator)
    @test haskey(rep, :early_energy)
    @test haskey(rep, :scaling)
    @test haskey(rep, :skew_adjointness)
    @test haskey(rep, :boundary_timing)
    @test haskey(rep, :parity)
    @test rep.parity.xi1_max == 0.0
    @test rep.parity.dxi1_max == 0.0
    @test rep.skew_adjointness.reflecting_sat.maxabs_no_origin <= 1e-10
    @test !isempty(rep.conclusions)
end

@testset "Reflecting SAT energy-drift focused diagnostics" begin
    rep = diagnose_reflecting_sat_energy_drift(
                                                source = source,
                                                accuracy_order = 4,
                                                N = 16,
                                                R = 1.6,
                                                dr = 0.1,
                                                p = 2,
                                                dt = 0.02,
                                                T_final = 0.24,
                                                K = 8,
                                                run_dt_scaling = false,
                                                verbose = false
                                               )

    @test haskey(rep, :config)
    @test haskey(rep, :cases)
    @test haskey(rep, :characteristic_constraint)
    @test haskey(rep, :operator_consistency)
    @test haskey(rep, :conclusions)

    @test rep.characteristic_constraint.kind == :w_in_minus_w_out
    @test rep.operator_consistency.runtime_vs_jacobian_maxabs <= 1e-10
    @test rep.operator_consistency.skew_runtime.maxabs_no_origin <= 1e-10
    @test haskey(rep.cases.gaussian, :potential_consistency)
    @test haskey(rep.cases.bump, :potential_consistency)
    @test rep.cases.gaussian.xi_growth_expected_from_pi0
    @test rep.cases.bump.xi_growth_expected_from_pi0
    @test rep.cases.gaussian.max_abs_interior_minus_flux_minus_sbp <= 1e-10
    @test rep.cases.bump.max_abs_interior_minus_flux_minus_sbp <= 1e-10
    @test !isempty(rep.cases.gaussian.early_rows)
    @test !isempty(rep.cases.bump.early_rows)
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti / 60, digits = 3), " minutes")
