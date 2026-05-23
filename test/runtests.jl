using Test
using SphericalSBPOperators
using LinearAlgebra: dot, mul!
using MultiFloats: Float64x2, Float64x4
using SparseArrays: sparse
using SummationByPartsOperators: MattssonNordström2004, SafeMode

println("Starting tests")
ti = time()

source = MattssonNordström2004()

@testset "Staggered operator API" begin
    ops = Staggered.spherical_operators(
        source;
        accuracy_order = 4,
        N = 32,
        R = 1.0,
        p = 2
    )

    @test size(ops.Geven) == (32, 32)
    @test size(ops.Godd) == (32, 32)
    @test size(ops.D) == (32, 32)
    @test size(ops.H) == (32, 32)
    @test ops.r[1] > 0
    @test ops.r[end] ≈ 1.0
    @test all(ops.H[i, i] > 0 for i in 1:ops.Nh)
    @test all(ops.H[i, i] != 0 for i in 1:ops.Nh)

    # Verify D is assembled from full-row mass inversion, with no origin special case.
    Hdiag = [ops.H[i, i] for i in 1:ops.Nh]
    RHS = ops.B - transpose(ops.Geven) * ops.H
    D_expected = zeros(Float64, ops.Nh, ops.Nh)
    for i in 1:ops.Nh, j in 1:ops.Nh

        D_expected[i, j] = RHS[i, j] / Hdiag[i]
    end
    @test Matrix(ops.D) ≈ D_expected atol = 1.0e-12 rtol = 1.0e-12

    report = Staggered.validate(ops; max_monomial_degree = 4, verbose = false)
    @test report.sbp.sbp_no_origin <= 1.0e-10
    @test !report.diagnostics.has_origin_node
    @test report.diagnostics.r0_ok

    # Top-level convenience wrappers should mirror the staggered submodule API.
    ops_wrap = staggered_spherical_operators(
        source;
        accuracy_order = 4,
        N = 32,
        R = 1.0,
        p = 2
    )
    @test Matrix(ops_wrap.Geven) ≈ Matrix(ops.Geven)
    @test Matrix(ops_wrap.D) ≈ Matrix(ops.D)
    report_wrap = validate_staggered(ops_wrap; max_monomial_degree = 4, verbose = false)
    @test report_wrap.sbp.sbp_no_origin <= 1.0e-10
end

@testset "Staggered wave evolution path" begin
    ops = staggered_spherical_operators(
        source;
        accuracy_order = 4,
        N = 32,
        R = 1.0,
        p = 2
    )
    phi0 = default_wave_profile(ops.r; amplitude = 1.0, center = 0.75, width = 0.06)
    psi0 = Vector{Float64}(ops.Geven * phi0)
    pi0 = -copy(psi0)

    sol = solve_wave_ode(
        ops;
        T_final = 0.06,
        boundary_condition = :none,
        phi0 = phi0,
        pi0 = pi0,
        psi0 = psi0,
        save_every = 1,
        verbose = false
    )

    @test sol.t[1] == 0.0
    @test sol.t[end] ≈ 0.06 atol = 1.0e-12
    @test size(sol.Π, 1) == length(ops.r)
    @test size(sol.Ψ, 1) == length(ops.r)
    @test all(isfinite, sol.Π)
    @test all(isfinite, sol.Ψ)
    @test all(isfinite, sol.energy)
    @test !sol.initial_data_check.has_origin_node
    @test !sol.initial_data_check.enforce_origin
end

@testset "Non-diagonal wave evolution path" begin
    ops = non_diagonal_spherical_operators(
        source;
        accuracy_order = 4,
        N = 16,
        R = 1.0,
        p = 2,
        mode = SafeMode()
    )
    phi0 = default_wave_profile(ops.r; amplitude = 1.0, center = 0.75, width = 0.06)
    psi0 = Vector{Float64}(ops.Geven * phi0)
    pi0 = -copy(psi0)

    sol = solve_wave_ode(
        ops;
        T_final = 0.02,
        boundary_condition = :none,
        phi0 = phi0,
        pi0 = pi0,
        psi0 = psi0,
        save_every = 1,
        verbose = false
    )

    @test sol.t[1] == 0.0
    @test sol.t[end] ≈ 0.02 atol = 1.0e-12
    @test size(sol.Π, 1) == length(ops.r)
    @test size(sol.Ψ, 1) == length(ops.r)
    @test all(isfinite, sol.Π)
    @test all(isfinite, sol.Ψ)
    @test all(isfinite, sol.energy)
    @test sol.initial_data_check.has_origin_node
    @test sol.initial_data_check.enforce_origin
end

@testset "Wave energy uses geometric norms" begin
    ops_diag = diagonal_spherical_operators(
        source;
        accuracy_order = 4,
        N = 24,
        R = 1.0,
        p = 2
    )
    Πd = collect(range(0.1, 1.0; length = ops_diag.Nh))
    Ψd = collect(range(-0.3, 0.4; length = ops_diag.Nh))

    E_diag = wave_energy(ops_diag, Πd, Ψd)
    E_diag_expected = 0.5 * (
        sum(Πd .* Vector(ops_diag.S * Πd)) +
        sum(Ψd .* Vector(ops_diag.V * Ψd))
    )
    @test E_diag ≈ E_diag_expected

    dΠd = Vector(ops_diag.D * Ψd)
    dΨd = Vector(ops_diag.Geven * Πd)
    dE_diag = energy_rate(ops_diag, Πd, Ψd, dΠd, dΨd)
    dE_diag_expected = sum(Πd .* Vector(ops_diag.S * dΠd)) +
        sum(Ψd .* Vector(ops_diag.V * dΨd))
    @test dE_diag ≈ dE_diag_expected

    ops_stag = staggered_spherical_operators(
        source;
        accuracy_order = 4,
        N = 24,
        R = 1.0,
        p = 2
    )
    Πs = collect(range(0.2, 0.9; length = ops_stag.Nh))
    Ψs = collect(range(-0.25, 0.35; length = ops_stag.Nh))

    E_stag = wave_energy(ops_stag, Πs, Ψs)
    E_stag_expected = 0.5 * (
        sum(Πs .* Vector(ops_stag.H * Πs)) +
        sum(Ψs .* Vector(ops_stag.H * Ψs))
    )
    @test E_stag ≈ E_stag_expected
end

@testset "Non-Diagonal Unified API" begin
    ops4 = non_diagonal_spherical_operators(
        source;
        accuracy_order = 4,
        N = 20,
        R = 20,
        p = 2,
        mode = SafeMode()
    )
    ref4 = sbp4_operators(
        source,
        21;
        h = 1,
        accuracy_order = 4,
        p = 2,
        mode = SafeMode(),
        verbose = false
    )
    seed4 = sbp4_scalar_mass_gradient(
        source;
        points = 21,
        h = 1,
        accuracy_order = 4,
        p = 2,
        mode = SafeMode()
    )

    @test ops4.G === ops4.Geven
    @test ops4.Nh == 21
    @test ops4.M_full == 41
    @test Matrix(ops4.H) == Matrix(seed4.Hcart_half)
    @test Matrix(ops4.H) != Matrix(ops4.S)
    @test Matrix(ops4.D) == Matrix(ref4.D)
    @test Matrix(ops4.Geven) == Matrix(ref4.G)
    @test Matrix(ops4.S) == Matrix(ref4.S)
    @test Matrix(ops4.V) == Matrix(ref4.V)
    @test Matrix(ops4.B) == Matrix(ref4.B)

    ops4_points = non_diagonal_spherical_operators(
        source;
        accuracy_order = 4,
        points = 21,
        R = 20,
        p = 2,
        mode = SafeMode()
    )
    @test ops4_points.Nh == ops4.Nh
    @test Matrix(ops4_points.D) == Matrix(ops4.D)

    ops6 = non_diagonal_spherical_operators(
        source;
        accuracy_order = 6,
        N = 20,
        R = 20,
        p = 2,
        mode = SafeMode()
    )
    ref6 = sbp6_operators(
        source,
        21;
        h = 1,
        accuracy_order = 6,
        p = 2,
        mode = SafeMode(),
        verbose = false
    )
    seed6 = sbp6_scalar_mass_gradient(
        source;
        points = 21,
        h = 1,
        accuracy_order = 6,
        p = 2,
        mode = SafeMode()
    )

    @test ops6.Nh == 21
    @test ops6.M_full == 41
    @test Matrix(ops6.H) == Matrix(seed6.Hcart_half)
    @test Matrix(ops6.H) != Matrix(ops6.S)
    @test Matrix(ops6.D) == Matrix(ref6.D)
    @test Matrix(ops6.Geven) == Matrix(ref6.G)
    @test Matrix(ops6.S) == Matrix(ref6.S)
    @test Matrix(ops6.V) == Matrix(ref6.V)
    @test Matrix(ops6.B) == Matrix(ref6.B)
end

@testset "Float64 acceptance matrix" begin
    for acc in (2, 4, 6, 8), p in (1, 2)

        ops = diagonal_spherical_operators(
            source;
            accuracy_order = acc,
            N = 64,
            R = 1.0,
            p = p
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
        @test report.sbp.sbp_no_origin <= 1.0e-10
        @test abs((ops.D * ops.r)[1] - (p + 1)) <= 1.0e-10
        @test haskey(report, :quadrature)
        @test haskey(report, :gradient_even)
        @test haskey(report, :divergence_odd)
        @test haskey(report, :sbp)
        @test haskey(report, :diagnostics)
    end
end

@testset "Mixed-order diagonal-mass operators" begin
    acc = 4
    p = 2
    N = 32
    R = 1.0

    ops_std = diagonal_spherical_operators(
        source;
        accuracy_order = acc,
        N = N,
        R = R,
        p = p,
        mode = SafeMode()
    )
    ops_mix = mixed_order_diagonal_spherical_operators(
        source;
        accuracy_order = acc,
        N = N,
        R = R,
        p = p,
        mode = SafeMode()
    )

    @test size(ops_mix.Geven) == size(ops_std.Geven)
    @test size(ops_mix.Godd) == size(ops_std.Godd)
    @test size(ops_mix.D) == size(ops_std.D)
    @test Matrix(ops_mix.Geven) ≈ Matrix(ops_std.Geven) atol = 1.0e-12 rtol = 1.0e-12
    @test maximum(abs.(Matrix(ops_mix.Godd) .- Matrix(ops_std.Godd))) > 0.0
    @test maximum(abs.(Matrix(ops_mix.D) .- Matrix(ops_std.D))) > 0.0

    report_mix = validate(ops_mix; max_monomial_degree = acc, verbose = false)
    @test report_mix.sbp.sbp_no_origin <= 1.0e-10
    @test abs((ops_mix.D * ops_mix.r)[1] - (p + 1)) <= 1.0e-10
end

@testset "Experimental diagonal repair widens the constrained block" begin
    ops_std = diagonal_spherical_operators(
        source;
        accuracy_order = 4,
        N = 30,
        R = 30 // 1,
        p = 2,
        mode = SafeMode(),
        return_canonical = true
    )
    ops_exp = diagonal_exp_spherical_operators(
        source;
        accuracy_order = 4,
        N = 30,
        R = 30 // 1,
        p = 2,
        mode = SafeMode(),
        return_canonical = true
    )
    exp_build = diagonal_exp_spherical_operators(
        source;
        accuracy_order = 4,
        N = 30,
        R = 30 // 1,
        p = 2,
        mode = SafeMode(),
        return_canonical = true,
        return_repair_info = true
    )

    err_std = ops_std.D * (ops_std.r .^ 3) .- 5 .* (ops_std.r .^ 2)
    err_exp = ops_exp.D * (ops_exp.r .^ 3) .- 5 .* (ops_exp.r .^ 2)

    @test Matrix(exp_build.ops.Geven) == Matrix(ops_exp.Geven)
    @test Matrix(exp_build.ops.D) == Matrix(ops_exp.D)
    @test exp_build.repair_info.constraint_system_consistent
    @test exp_build.repair_info.constraint_rank == exp_build.repair_info.augmented_constraint_rank
    @test exp_build.repair_info.affected_divergence_rows == exp_build.repair_info.stencil_cols
    @test exp_build.repair_info.optimized_target_rows == [9, 10, 11]

    @test err_std[4] != 0
    @test err_std[5] != 0
    @test err_exp[4] == 0
    @test err_exp[5] == 0
    @test maximum(abs.(err_exp[6:8])) < maximum(abs.(err_std[6:8]))
end

@testset "Experimental SBP6 without outer boundary closure help" begin
    solved = NonDiagonalMass.sbp6_exp_solve_accuracy_constraints(
        source;
        points = 31,
        h = 1,
        N = 30,
        R = 30,
        p = 2,
        mode = SafeMode(),
        outer_boundary_closure_help = false,
        exact_solve = true,
        verbose = false
    )

    V = Matrix(solved.V)
    Sdiag = [solved.S[i, i] for i in 1:31]

    @test solved.outer_boundary_closure_help == false
    @test solved.row_sets.rows_r3 == collect(1:25)
    @test isempty(solved.template.right_diag_indices)
    @test solved.template.v_offdiag_pairs == [(2, 3), (2, 4), (3, 4), (3, 5), (4, 5),
                                              (5, 6), (6, 7)]

    @test Sdiag[1] == 1430827147971 // 4664636476456
    @test Sdiag[2] == 42904697232509 // 51311001241016
    @test Sdiag[3] == 1539566352701 // 424057861496
    @test Sdiag[4] == 474102956851443 // 51311001241016
    @test Sdiag[5] == 37409168459741 // 2332318238228
    @test Sdiag[6] == 1279079894309069 // 51311001241016
    @test Sdiag[7] == 168033579790977 // 4664636476456
    @test Sdiag[8] == 1257018716149729 // 25655500620508
    @test Sdiag[9] == 298538403436579 // 4664636476456
    @test Sdiag[10] == 4156190224075161 // 51311001241016
    @test Sdiag[11] == 100
    @test Sdiag[12] == 121
    @test Sdiag[13] == 144
    @test Sdiag[14] == 169
    @test Sdiag[15] == 196
    @test Sdiag[16] == 225
    @test Sdiag[17] == 256
    @test Sdiag[18] == 289
    @test Sdiag[19] == 324
    @test Sdiag[20] == 361
    @test Sdiag[21] == 400
    @test Sdiag[22] == 441
    @test Sdiag[23] == 484
    @test Sdiag[24] == 529
    @test Sdiag[25] == 576
    @test Sdiag[26] == 1095025 // 1728
    @test Sdiag[27] == 1331213 // 2160
    @test Sdiag[28] == 144693 // 160
    @test Sdiag[29] == 132839 // 270
    @test Sdiag[30] == 10102933 // 8640
    @test Sdiag[31] == 13649 // 48

    @test V[2, 2] == 69612410558321 // 51311001241016
    @test V[2, 3] == 4996740529431 // 6413875155127
    @test V[2, 4] == -13306004610507 // 51311001241016
    @test V[3, 3] == 3923059854503 // 1509147095324
    @test V[3, 4] == 949724456067 // 1166159119114
    @test V[3, 5] == -11638692514107 // 51311001241016
    @test V[4, 4] == 3014861355653139 // 359177008687112
    @test V[4, 5] == 11797110150741 // 359177008687112
    @test V[5, 5] == 11554098239601139 // 718354017374224
    @test V[5, 6] == 239755863585 // 4664636476456
    @test V[6, 6] == 29105204758165 // 1166159119114
    @test V[6, 7] == -70992217935 // 12827750310254
    @test V[7, 7] == 461864744704269 // 12827750310254
    @test V[8, 8] == 49
    @test V[9, 9] == 64
    @test V[10, 10] == 81
    @test V[26, 26] == 1095025 // 1728
    @test V[27, 27] == 1331213 // 2160
    @test V[28, 28] == 144693 // 160
    @test V[29, 29] == 132839 // 270
    @test V[30, 30] == 10102933 // 8640
    @test V[31, 31] == 13649 // 48
end

@testset "Direct matrix extraction path" begin
    Dfull, xfull, Gfull, Hfull = SphericalSBPOperators._build_full_grid_objects(
        source;
        accuracy_order = 4,
        N = 24,
        R = 24.0,
        mode = SafeMode()
    )

    @test size(Matrix(Dfull)) == size(Gfull)
    @test Matrix(Gfull) == Matrix(Dfull)
    @test size(Hfull) == (length(xfull), length(xfull))
end

@testset "Rational exactness mode (small/medium N)" begin
    for N in (8, 16, 24), acc in (2, 4, 6, 8), p in (1, 2)
        ops = diagonal_spherical_operators(
            source;
            accuracy_order = acc,
            N = N,
            R = 1 // 1,
            p = p,
            mode = SafeMode()
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
    ops_canonical = diagonal_spherical_operators(
        source;
        accuracy_order = 8,
        N = 32,
        R = 1.0,
        p = 2,
        mode = SafeMode(),
        return_canonical = true
    )
    @test eltype(ops_canonical.r) == Rational{BigInt}
    @test ops_canonical.r[2] - ops_canonical.r[1] == 1 // 1
    @test ops_canonical.r[end] == 32 // 1
    @test all(ops_canonical.Geven[i, 1] == 0 for i in 2:ops_canonical.Nh)
    @test ops_canonical.D * ops_canonical.r == fill(3 // 1, ops_canonical.Nh)

    # Raw folded matrix has four contaminated rows for acc=8.
    Dfull, xfull,
        Gfull,
        _ = SphericalSBPOperators._build_full_grid_objects(
        source;
        accuracy_order = 8,
        N = 32,
        R = big(32) // 1,
        mode = SafeMode()
    )
    r_raw, Rop,
        Eeven,
        _ = SphericalSBPOperators._build_folding_operators(
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
    @test maximum(abs.(ops_scaled.Geven[2:end, 1])) <= 1.0e-12
end

@testset "Parity helpers" begin
    u = [1.0, 2.0, 3.0]
    @test !check_odd(u; tol = 1.0e-12)
    enforce_odd!(u)
    @test check_odd(u; tol = 1.0e-12)
    @test u[1] == 0.0
end

@testset "Diagnostics suite" begin
    ops = diagonal_spherical_operators(
        source;
        accuracy_order = 4,
        N = 24,
        R = 1.0,
        p = 2
    )
    report = validate(ops; max_monomial_degree = 4, verbose = false)
    diag = diagnose(
        ops, report; Ktest = 6, region_points = 8, table_points = 4, verbose = false
    )
    g4 = nothing
    for entry in report.gradient_even
        if entry.degree == 4
            g4 = entry
            break
        end
    end
    d1 = nothing
    for entry in report.divergence_odd
        if entry.degree == 1
            d1 = entry
            break
        end
    end

    @test diag.grid.M == 49
    @test diag.grid.Nh == 25
    @test diag.grid.grid_ok
    @test diag.folding.rop_ok
    @test diag.folding.even_even_ok
    @test diag.folding.odd_signed_ok
    @test diag.folding.odd_matches_xk_for_odd_k_ok
    @test diag.operator_consistency.geven_diff_max <= 1.0e-10
    @test diag.operator_consistency.godd_diff_max <= 1.0e-10
    @test report.diagnostics.closure_width_right > 0
    @test g4.max_error_safe < 1.0e-10
    @test d1.max_error_safe < 1.0e-10
    @test g4.max_error_near_boundary ≈ g4.max_error
    @test d1.max_error_near_boundary <= d1.max_error
    @test diag.closure.safe_count > 0
    @test !diag.polynomial.skipped
    @test !isempty(diag.polynomial.gradient.entries)
    @test !isempty(diag.fullgrid_comparison.entries)
    @test diag.sbp.max_no_origin <= 1.0e-10
    @test !isempty(diag.interpretation.conclusions)
end

@testset "Wave SAT boundary and symmetry tests" begin
    ops = diagonal_spherical_operators(
        source;
        accuracy_order = 4,
        N = 32,
        R = 1.0,
        p = 2
    )

    # Outgoing initial packet: pi0 = -psi0.
    phi0 = default_wave_profile(ops.r; amplitude = 1.0, center = 0.75, width = 0.06)
    psi0 = Vector{Float64}(ops.Geven * phi0)
    pi0 = -copy(psi0)

    sol_abs = solve_wave_ode(
        ops;
        T_final = 0.12,
        boundary_condition = :absorbing,
        phi0 = phi0,
        pi0 = pi0,
        psi0 = psi0,
        save_every = 1,
        verbose = false
    )

    @test sol_abs.t[1] == 0.0
    @test sol_abs.t[end] ≈ 0.12 atol = 1.0e-12
    @test size(sol_abs.Π, 1) == length(ops.r)
    @test size(sol_abs.Ψ, 1) == length(ops.r)
    @test size(sol_abs.Π, 2) == length(sol_abs.t)
    @test size(sol_abs.Ψ, 2) == length(sol_abs.t)
    @test all(isfinite, sol_abs.Π)
    @test all(isfinite, sol_abs.Ψ)
    @test all(isfinite, sol_abs.energy)
    @test maximum(abs.(sol_abs.Ψ[1, :])) == 0.0
    @test minimum(sol_abs.energy) >= -1.0e-10
    @test sol_abs.pi === sol_abs.Π
    @test sol_abs.psi === sol_abs.Ψ
    @test sol_abs.initial_data_check.consistent
    @test sol_abs.initial_data_check.origin_ok
    @test !sol_abs.initial_data_check.require_boundary
    @test sol_abs.boundary_condition == :absorbing

    # Absorbing BC: energy should be non-increasing up to time-integration noise.
    dE_abs = diff(sol_abs.energy)
    @test maximum(dE_abs) <= 2.0e-6
    @test sol_abs.energy[end] <= sol_abs.energy[1] + 2.0e-6

    # Absorbing residual should stay small for outgoing packet.
    w_in_abs = sol_abs.Π[end, :] .+ sol_abs.Ψ[end, :]
    w_out_abs = sol_abs.Π[end, :] .- sol_abs.Ψ[end, :]
    @test maximum(abs.(w_in_abs)) <= 0.3 * maximum(abs.(w_out_abs)) + 5.0e-4

    # Reflecting SAT BC (energy-conserving characteristic reflection).
    sol_ref = solve_wave_ode(
        ops;
        T_final = 0.12,
        boundary_condition = :reflecting,
        phi0 = phi0,
        pi0 = pi0,
        psi0 = psi0,
        save_every = 1,
        verbose = false
    )
    rel_energy_drift = maximum(abs.(sol_ref.energy .- sol_ref.energy[1])) /
        max(abs(sol_ref.energy[1]), 1.0e-14)
    @test rel_energy_drift <= 2.0e-4
    w_reflect_residual = sol_ref.Π[end, :] .+ sol_ref.Ψ[end, :] .-
        (sol_ref.Π[end, :] .- sol_ref.Ψ[end, :])
    @test all(isfinite, w_reflect_residual)
    @test maximum(abs.(w_reflect_residual)) <= 1.0

    # Parity invariance check: disabling origin enforcement permits drift/nonzero origin.
    psi0_bad = copy(psi0)
    psi0_bad[1] = 1.0e-4
    sol_no_enforce = solve_wave_ode(
        ops;
        T_final = 0.03,
        boundary_condition = :none,
        phi0 = phi0,
        pi0 = pi0,
        psi0 = psi0_bad,
        enforce_origin = false,
        save_every = 1,
        verbose = false
    )
    @test maximum(abs.(sol_no_enforce.Ψ[1, :])) >= 1.0e-8
end

@testset "Wave initial-data modes and potential consistency" begin
    ops = diagonal_spherical_operators(
        source;
        accuracy_order = 4,
        N = 32,
        R = 1.0,
        p = 2
    )
    r = Float64.(ops.r)

    phi_bump = bumpb_profile(r; amplitude = 1.0, center = 0.0, radius = 0.35)
    pi_zero = zeros(Float64, length(r))
    sol_pot = solve_wave_ode(
        ops;
        T_final = 0.02,
        dt = 0.01,
        boundary_condition = :none,
        initial_data_mode = :potential,
        phi0 = phi_bump,
        pi0 = pi_zero,
        verbose = false
    )
    psi_expected = Vector{Float64}(ops.Geven * phi_bump)
    psi_expected[1] = 0.0
    @test sol_pot.initial_data_check.mode == :potential
    @test sol_pot.initial_data_check.potential.residual_ok
    @test maximum(abs.(sol_pot.Ψ[:, 1] .- psi_expected)) <= 1.0e-10

    pi_bump = bumpb_profile(r; amplitude = 1.0, center = 0.0, radius = 0.35)
    psi_zero = zeros(Float64, length(r))
    pcheck = check_potential_consistency(ops, pi_bump, psi_zero; tol = 1.0e-10)
    @test pcheck.psi_growth_expected_from_pi
    @test pcheck.growth_explanation ==
        "Ψ appears immediately if Π0 varies because Ψ_t = G Π."
    @test any(contains("This initial state is inconsistent with Ψ=φ_r for any scalar φ"), pcheck.warnings)

    w_in0 = ones(Float64, length(r))
    w_out0 = -ones(Float64, length(r))
    char = characteristic_initial_data(r; w_in0 = w_in0, w_out0 = w_out0)
    @test maximum(abs.(char.Π0[2:end])) <= 1.0e-12
    @test maximum(abs.(char.Ψ0[2:end] .- 1.0)) <= 1.0e-12
    @test char.Ψ0[1] == 0.0
end

@testset "Regular left-moving spherical Gaussian analytic data" begin
    params = (amplitude = 1.25, R = 6.0, d = 0.8)
    rpos = 0.37
    rneg = -rpos
    t = 0.21

    @test regular_left_moving_spherical_gaussian_phi_exact(t, rneg; params...) ≈
          regular_left_moving_spherical_gaussian_phi_exact(t, rpos; params...) atol = 1.0e-12
    @test regular_left_moving_spherical_gaussian_pi_exact(t, rneg; params...) ≈
          regular_left_moving_spherical_gaussian_pi_exact(t, rpos; params...) atol = 1.0e-12
    @test regular_left_moving_spherical_gaussian_psi_exact(t, rneg; params...) ≈
          -regular_left_moving_spherical_gaussian_psi_exact(t, rpos; params...) atol = 1.0e-12

    @test regular_left_moving_spherical_gaussian_phi_exact(t, 0.0; params...) ≈
          2.0 * regular_left_moving_spherical_gaussian_profile_prime(t; params...) atol = 1.0e-12
    @test regular_left_moving_spherical_gaussian_pi_exact(t, 0.0; params...) ≈
          2.0 * regular_left_moving_spherical_gaussian_profile_second(t; params...) atol = 1.0e-12
    @test regular_left_moving_spherical_gaussian_psi_exact(t, 0.0; params...) == 0.0

    r = collect(range(0.0, params.R; length = 33))
    exact0 = regular_left_moving_spherical_gaussian_solution(0.0, r; params...)
    @test exact0.Φ[1] ≈ regular_left_moving_spherical_gaussian_phi_exact(0.0, 0.0; params...) atol = 1.0e-12
    @test exact0.Π[1] ≈ 2.0 * regular_left_moving_spherical_gaussian_profile_second(0.0; params...) atol = 1.0e-12
    @test exact0.Ψ[1] == 0.0
    @test all(isfinite, exact0.Φ)
    @test all(isfinite, exact0.Π)
    @test all(isfinite, exact0.Ψ)
end

@testset "Boundary-aware analytic wave references" begin
    r = collect(range(0.0, 6.0; length = 33))

    exact_none = analytic_wave_solution(0.0,
                                        r;
                                        initial_data_kind = :regular_left_moving_spherical_gaussian,
                                        boundary_condition = :none,
                                        amplitude = 1.25,
                                        R = 6.0,
                                        d = 0.8,
                                        rc = 3.0)
    @test exact_none.boundary_condition == :none
    @test exact_none.boundary_ok
    @test hasproperty(exact_none, :Φ)
    @test hasproperty(exact_none, :Π)
    @test hasproperty(exact_none, :Ψ)

    @test_throws ArgumentError analytic_wave_solution(0.0,
                                                      r;
                                                      initial_data_kind = :regular_left_moving_spherical_gaussian,
                                                      boundary_condition = :reflecting,
                                                      amplitude = 1.25,
                                                      R = 6.0,
                                                      d = 0.8,
                                                      rc = 3.0)

    exact_unchecked = analytic_wave_solution(0.0,
                                             r;
                                             initial_data_kind = :regular_left_moving_spherical_gaussian,
                                             boundary_condition = :reflecting,
                                             amplitude = 1.25,
                                             R = 6.0,
                                             d = 0.8,
                                             rc = 3.0,
                                             validate_boundary = false)
    @test exact_unchecked.boundary_condition == :reflecting
    @test !exact_unchecked.boundary_ok
    @test exact_unchecked.boundary_residual > exact_unchecked.boundary_tolerance
end

@testset "Pointwise analytic-reference scaled errors" begin
    q = 2
    R = 6.0
    amplitude = 1.25
    d = 0.8
    rc = 3.0
    t = [0.0]
    coarse_r = collect(range(0.0, R; length = 9))
    medium_r = collect(range(0.0, R; length = 17))
    fine_r = collect(range(0.0, R; length = 33))

    exact_coarse = regular_left_moving_spherical_gaussian_solution(0.0, coarse_r;
                                                                   amplitude = amplitude,
                                                                   R = R,
                                                                   d = d,
                                                                   rc = rc)
    exact_medium = regular_left_moving_spherical_gaussian_solution(0.0, medium_r;
                                                                   amplitude = amplitude,
                                                                   R = R,
                                                                   d = d,
                                                                   rc = rc)
    exact_fine = regular_left_moving_spherical_gaussian_solution(0.0, fine_r;
                                                                 amplitude = amplitude,
                                                                 R = R,
                                                                 d = d,
                                                                 rc = rc)

    h = coarse_r[2] - coarse_r[1]
    h2 = medium_r[2] - medium_r[1]
    h4 = fine_r[2] - fine_r[1]
    err_profile_coarse = sin.(coarse_r)
    err_profile_medium = sin.(medium_r)
    err_profile_fine = sin.(fine_r)

    sim_h = (sol = (t = t,
                    r = coarse_r,
                    Π = reshape(exact_coarse.Π .+ h^q .* err_profile_coarse, :, 1),
                    Ψ = reshape(exact_coarse.Ψ .- h^q .* err_profile_coarse, :, 1),
                    boundary_condition = :none),
             wave_config = (boundary_condition = :none,),
             initial_data = (kind = :regular_left_moving_spherical_gaussian,
                             summary = (amplitude = amplitude, R = R, d = d, rc = rc)))
    sim_h2 = (sol = (t = t,
                     r = medium_r,
                     Π = reshape(exact_medium.Π .+ h2^q .* err_profile_medium, :, 1),
                     Ψ = reshape(exact_medium.Ψ .- h2^q .* err_profile_medium, :, 1),
                     boundary_condition = :none),
              wave_config = (boundary_condition = :none,),
              initial_data = (kind = :regular_left_moving_spherical_gaussian,
                              summary = (amplitude = amplitude, R = R, d = d, rc = rc)))
    sim_h4 = (sol = (t = t,
                     r = fine_r,
                     Π = reshape(exact_fine.Π .+ h4^q .* err_profile_fine, :, 1),
                     Ψ = reshape(exact_fine.Ψ .- h4^q .* err_profile_fine, :, 1),
                     boundary_condition = :none),
              wave_config = (boundary_condition = :none,),
              initial_data = (kind = :regular_left_moving_spherical_gaussian,
                              summary = (amplitude = amplitude, R = R, d = d, rc = rc)))

    pi_data = SphericalSBPOperators._pointwise_analytic_error_history(sim_h, sim_h2, sim_h4;
                                                                      field = :Π,
                                                                      expected_order = q)
    psi_data = SphericalSBPOperators._pointwise_analytic_error_history(sim_h, sim_h2, sim_h4;
                                                                       field = :Ψ,
                                                                       expected_order = q)

    @test pi_data.scaled_err_h[:, 1] ≈ pi_data.scaled_err_h2[:, 1] atol = 1.0e-12 rtol = 1.0e-12
    @test pi_data.scaled_err_h[:, 1] ≈ pi_data.scaled_err_h4[:, 1] atol = 1.0e-12 rtol = 1.0e-12
    @test psi_data.scaled_err_h[:, 1] ≈ psi_data.scaled_err_h2[:, 1] atol = 1.0e-12 rtol = 1.0e-12
    @test psi_data.scaled_err_h[:, 1] ≈ psi_data.scaled_err_h4[:, 1] atol = 1.0e-12 rtol = 1.0e-12
    @test pi_data.scale_h == 1.0
    @test pi_data.scale_h2 ≈ 2.0^q atol = 1.0e-12
    @test pi_data.scale_h4 ≈ 4.0^q atol = 1.0e-12
end

@testset "Provided dt stability guard" begin
    ops = diagonal_spherical_operators(
        source;
        accuracy_order = 4,
        N = 24,
        R = 1.0,
        p = 2
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
        opsN = diagonal_spherical_operators(
            source;
            accuracy_order = 4,
            N = N,
            R = 1.0,
            p = 2
        )
        phiN = default_wave_profile(opsN.r; amplitude = 1.0, center = 0.35, width = 0.08)
        xiN = Vector{Float64}(opsN.Geven * phiN)
        piN = zeros(Float64, length(opsN.r))
        solN = solve_wave_ode(
            opsN;
            T_final = 0.03,
            boundary_condition = :none,
            phi0 = phiN,
            pi0 = piN,
            psi0 = xiN,
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
    ops = diagonal_spherical_operators(
        source;
        accuracy_order = 6,
        N = 24,
        R = 2.4,
        p = 2
    )
    n = length(ops.r)
    pi0 = exp.(-0.5 .* (Float64.(ops.r) .^ 2))
    psi0 = zeros(Float64, n)

    p = WaveODEParams(ops; boundary_condition = :reflecting, enforce_origin = true)
    U = vcat(pi0, psi0)
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
        pi0 = pi0,
        psi0 = psi0,
        save_every = 1,
        verbose = false
    )
    @test size(sol.Π, 1) == n
    @test size(sol.Ψ, 1) == n
    @test all(isfinite, sol.energy)
end

@testset "Integrator microbenchmark helper" begin
    ops = diagonal_spherical_operators(
        source;
        accuracy_order = 4,
        N = 20,
        R = 2.0,
        p = 2
    )
    pi0 = exp.(-0.5 .* (Float64.(ops.r) .^ 2))
    psi0 = zeros(Float64, length(ops.r))
    bench = benchmark_wave_integrators(
        ops;
        T_final = 0.1,
        dt = 0.01,
        boundary_condition = :reflecting,
        pi0 = pi0,
        psi0 = psi0,
        warmup = false,
        verbose = false
    )
    @test haskey(bench, :tsitpap8)
    @test haskey(bench, :implicit_midpoint)
    @test bench.tsitpap8.nsteps == 10
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
    @test rep.skew_adjointness.reflecting_sat.maxabs_no_origin <= 1.0e-10
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
    @test rep.operator_consistency.runtime_vs_jacobian_maxabs <= 1.0e-10
    @test rep.operator_consistency.skew_runtime.maxabs_no_origin <= 1.0e-10
    @test haskey(rep.cases.gaussian, :potential_consistency)
    @test haskey(rep.cases.bump, :potential_consistency)
    @test rep.cases.gaussian.psi_growth_expected_from_pi0
    @test rep.cases.bump.psi_growth_expected_from_pi0
    @test rep.cases.gaussian.max_abs_interior_minus_flux_minus_sbp <= 1.0e-10
    @test rep.cases.bump.max_abs_interior_minus_flux_minus_sbp <= 1.0e-10
    @test !isempty(rep.cases.gaussian.early_rows)
    @test !isempty(rep.cases.bump.early_rows)
end

@testset "Gundlach error diagnostics" begin
    u_h = [1.0, 2.0, 4.0]
    u_ref = [1.0, 1.0, 1.0]
    H = [2.0 0.0 0.0;
         0.0 3.0 0.0;
         0.0 0.0 4.0]

    err = gundlach_error(u_h, u_ref; h = 0.2, k = 2, h0 = 0.1)
    @test err ≈ [0.0, 0.25, 0.75]

    norm_val = gundlach_error_norm(u_h, u_ref, H; h = 0.2, k = 2, R = 2.0, h0 = 0.1)
    @test norm_val ≈ sqrt(dot(err, H * err) / 2.0) atol = 1.0e-12

    err_h0_a = gundlach_error(u_h, u_ref; h = 0.4, k = 2, h0 = 0.1)
    err_h0_b = gundlach_error(u_h, u_ref; h = 0.4, k = 2, h0 = 0.2)
    @test err_h0_b ≈ 4.0 .* err_h0_a

    coarse_r = [0.0, 1.0, 2.0]
    ref_r = [0.0, 0.5, 1.0, 1.5, 2.0]
    coarse_pi = reshape([1.0, 3.0, 5.0], :, 1)
    coarse_psi = reshape([2.0, 5.0, 8.0], :, 1)
    ref_pi = reshape([0.0, 1.0, 2.0, 3.0, 4.0], :, 1)
    ref_psi = reshape([2.0, 3.0, 4.0, 5.0, 6.0], :, 1)
    coarse_sol = WaveEvolutionResult([0.0],
                                     coarse_pi,
                                     coarse_psi,
                                     [0.0],
                                     coarse_r,
                                     0.1,
                                     0,
                                     :none,
                                     nothing)
    ref_sol = WaveEvolutionResult([0.0],
                                  ref_pi,
                                  ref_psi,
                                  [0.0],
                                  ref_r,
                                  0.05,
                                  0,
                                  :none,
                                  nothing)
    ops = (S = [2.0 0.0 0.0;
                0.0 2.0 0.0;
                0.0 0.0 2.0],
           V = [5.0 0.0 0.0;
                0.0 5.0 0.0;
                0.0 0.0 5.0],
           R = 2.0)

    err_pi = gundlach_error(coarse_sol, ref_sol, ops, Val(:Π), 1; k = 0, h0 = 1.0)
    @test err_pi ≈ [1.0, 1.0, 1.0]

    err_psi = gundlach_error(coarse_sol, ref_sol, ops, :Psi, 1; k = 0, h0 = 1.0)
    @test err_psi ≈ [0.0, 1.0, 2.0]

    norm_pi = gundlach_error_norm(coarse_sol, ref_sol, ops, Val(:Π), 1; k = 0, h0 = 1.0)
    @test norm_pi ≈ sqrt(3.0) atol = 1.0e-12

    norm_psi = gundlach_error_norm(coarse_sol, ref_sol, ops, Val(:Ψ), 1; k = 0, h0 = 1.0)
    @test norm_psi ≈ sqrt(12.5) atol = 1.0e-12

    @test_throws DimensionMismatch gundlach_error([1.0, 2.0], [1.0]; h = 0.1, k = 1)
    @test_throws DimensionMismatch gundlach_error_norm([1.0, 2.0], [0.0, 0.0], ones(3, 3);
                                                       h = 0.1,
                                                       k = 1,
                                                       R = 1.0)
    @test_throws ArgumentError gundlach_error([1.0], [0.0]; h = 0.0, k = 1)
    @test_throws ArgumentError gundlach_error_norm([1.0], [0.0], ones(1, 1);
                                                   h = 0.1,
                                                   k = 1,
                                                   R = -1.0)
end

@testset "Gundlach plotting data helpers" begin
    q = 2

    coarse_r = [0.0, 1.0, 2.0]
    medium_r = collect(0.0:0.5:2.0)
    fine_r = collect(0.0:0.25:2.0)

    coarse_pi = reshape(fill(1.0^q, length(coarse_r)), :, 1)
    medium_pi = reshape(fill(0.5^q, length(medium_r)), :, 1)
    fine_pi = reshape(fill(0.25^q, length(fine_r)), :, 1)
    ops_h = (S = [i == j ? 2.0 / length(coarse_r) : 0.0 for i in 1:length(coarse_r), j in 1:length(coarse_r)],
             V = [i == j ? 2.0 / length(coarse_r) : 0.0 for i in 1:length(coarse_r), j in 1:length(coarse_r)],
             R = 2.0)
    ops_h2 = (S = [i == j ? 2.0 / length(medium_r) : 0.0 for i in 1:length(medium_r), j in 1:length(medium_r)],
              V = [i == j ? 2.0 / length(medium_r) : 0.0 for i in 1:length(medium_r), j in 1:length(medium_r)],
              R = 2.0)
    ops_h4 = (S = [i == j ? 2.0 / length(fine_r) : 0.0 for i in 1:length(fine_r), j in 1:length(fine_r)],
              V = [i == j ? 2.0 / length(fine_r) : 0.0 for i in 1:length(fine_r), j in 1:length(fine_r)],
              R = 2.0)

    sol_h = WaveEvolutionResult([0.0],
                                coarse_pi,
                                copy(coarse_pi),
                                [0.0],
                                coarse_r,
                                0.1,
                                0,
                                :none,
                                nothing)
    sol_h2 = WaveEvolutionResult([0.0],
                                 medium_pi,
                                 copy(medium_pi),
                                 [0.0],
                                 medium_r,
                                 0.05,
                                 0,
                                 :none,
                                 nothing)
    sol_h4 = WaveEvolutionResult([0.0],
                                 fine_pi,
                                 copy(fine_pi),
                                 [0.0],
                                 fine_r,
                                 0.025,
                                 0,
                                 :none,
                                 nothing)
    sim_h = (sol = sol_h, ops = ops_h)
    sim_h2 = (sol = sol_h2, ops = ops_h2)
    sim_h4 = (sol = sol_h4, ops = ops_h4)

    err_data = SphericalSBPOperators._gundlach_pointwise_error_history(sim_h, sim_h2, sim_h4;
                                                                       field = :Π,
                                                                       expected_order = q,
                                                                       h0 = 1.0)
    @test err_data.err_h_h2[:, 1] ≈ err_data.err_h2_h4[:, 1]
    @test err_data.norm_h_h2[1] ≈ err_data.norm_h2_h4[1]

    order_data = SphericalSBPOperators._gundlach_pointwise_convergence_order_history(sim_h, sim_h2, sim_h4;
                                                                                      field = :Π,
                                                                                      expected_order = q,
                                                                                      h0 = 1.0)
    @test order_data.order[:, 1] ≈ fill(2.0, length(coarse_r))
    @test order_data.norm_order[1] ≈ 2.0 atol = 1.0e-12

    coarse_r_interp = [0.0, 1.0, 2.0]
    medium_r_interp = collect(0.0:0.4:2.0)
    fine_r_interp = collect(0.0:0.2:2.0)
    coarse_pi_interp = reshape([1.0, 2.0, 3.0], :, 1)
    medium_pi_interp = reshape(1.0 .+ medium_r_interp, :, 1)
    fine_pi_interp = reshape(1.0 .+ fine_r_interp, :, 1)

    sol_h_interp = WaveEvolutionResult([0.0],
                                       coarse_pi_interp,
                                       copy(coarse_pi_interp),
                                       [0.0],
                                       coarse_r_interp,
                                       0.1,
                                       0,
                                       :none,
                                       nothing)
    sol_h2_interp = WaveEvolutionResult([0.0],
                                        medium_pi_interp,
                                        copy(medium_pi_interp),
                                        [0.0],
                                        medium_r_interp,
                                        0.04,
                                        0,
                                        :none,
                                        nothing)
    sol_h4_interp = WaveEvolutionResult([0.0],
                                        fine_pi_interp,
                                        copy(fine_pi_interp),
                                        [0.0],
                                        fine_r_interp,
                                        0.02,
                                        0,
                                        :none,
                                        nothing)
    ops_h_interp = (S = [i == j ? 2.0 / length(coarse_r_interp) : 0.0 for i in 1:length(coarse_r_interp), j in 1:length(coarse_r_interp)],
                    V = [i == j ? 2.0 / length(coarse_r_interp) : 0.0 for i in 1:length(coarse_r_interp), j in 1:length(coarse_r_interp)],
                    R = 2.0)
    ops_h2_interp = (S = [i == j ? 2.0 / length(medium_r_interp) : 0.0 for i in 1:length(medium_r_interp), j in 1:length(medium_r_interp)],
                     V = [i == j ? 2.0 / length(medium_r_interp) : 0.0 for i in 1:length(medium_r_interp), j in 1:length(medium_r_interp)],
                     R = 2.0)
    ops_h4_interp = (S = [i == j ? 2.0 / length(fine_r_interp) : 0.0 for i in 1:length(fine_r_interp), j in 1:length(fine_r_interp)],
                     V = [i == j ? 2.0 / length(fine_r_interp) : 0.0 for i in 1:length(fine_r_interp), j in 1:length(fine_r_interp)],
                     R = 2.0)
    sim_h_interp = (sol = sol_h_interp, ops = ops_h_interp)
    sim_h2_interp = (sol = sol_h2_interp, ops = ops_h2_interp)
    sim_h4_interp = (sol = sol_h4_interp, ops = ops_h4_interp)

    interp_data = SphericalSBPOperators._gundlach_pointwise_error_history(sim_h_interp,
                                                                          sim_h2_interp,
                                                                          sim_h4_interp;
                                                                          field = :Π,
                                                                          expected_order = 1,
                                                                          h0 = 1.0)
    @test interp_data.r ≈ coarse_r_interp
    @test interp_data.err_h_h2[:, 1] ≈ zeros(length(coarse_r_interp))
    @test interp_data.err_h2_h4[:, 1] ≈ zeros(length(coarse_r_interp))
end

@testset "Wave matrix kernels agree with sparse mul!" begin
    function check_wave_D_kernel(A, K, x; rtol, atol)
        T = eltype(x)
        y_sparse = similar(x)
        y_kernel = similar(x)
        y_beta_initial = T.(1:length(x)) ./ T(length(x) + 1)
        y_beta_sparse = copy(y_beta_initial)
        y_beta_kernel = copy(y_beta_initial)

        mul!(y_sparse, A, x)
        wave_D_mul!(y_kernel, K, x)
        wave_D_mul!(y_kernel, K, x)
        @test y_kernel ≈ y_sparse rtol = rtol atol = atol
        @test (@allocated wave_D_mul!(y_kernel, K, x)) == 0

        y_beta_sparse .= y_beta_initial
        y_beta_kernel .= y_beta_sparse
        mul!(y_beta_sparse, A, x, T(2), T(-0.25))
        wave_D_mul!(y_beta_kernel, K, x, T(2), T(-0.25))
        @test y_beta_kernel ≈ y_beta_sparse rtol = rtol atol = atol
        y_beta_kernel .= y_beta_initial
        @test (@allocated wave_D_mul!(y_beta_kernel, K, x, T(2), T(-0.25))) == 0
    end

    function check_wave_Geven_kernel(A, K, x; rtol, atol)
        T = eltype(x)
        y_sparse = similar(x)
        y_kernel = similar(x)
        y_beta_initial = T.(1:length(x)) ./ T(length(x) + 1)
        y_beta_sparse = copy(y_beta_initial)
        y_beta_kernel = copy(y_beta_initial)

        mul!(y_sparse, A, x)
        wave_Geven_mul!(y_kernel, K, x)
        wave_Geven_mul!(y_kernel, K, x)
        @test y_kernel ≈ y_sparse rtol = rtol atol = atol
        @test (@allocated wave_Geven_mul!(y_kernel, K, x)) == 0

        y_beta_sparse .= y_beta_initial
        y_beta_kernel .= y_beta_sparse
        mul!(y_beta_sparse, A, x, T(2), T(-0.25))
        wave_Geven_mul!(y_beta_kernel, K, x, T(2), T(-0.25))
        @test y_beta_kernel ≈ y_beta_sparse rtol = rtol atol = atol
        y_beta_kernel .= y_beta_initial
        @test (@allocated wave_Geven_mul!(y_beta_kernel, K, x, T(2), T(-0.25))) == 0
    end

    function check_wave_kernel_pair(ops, ::Type{T}; rtol, atol) where {T}
        x = T.(sin.(Float64.(ops.r)))
        check_wave_D_kernel(ops.D,
                            wave_D_kernel(ops; target_eltype = T),
                            x;
                            rtol = rtol,
                            atol = atol)
        check_wave_Geven_kernel(ops.Geven,
                                wave_Geven_kernel(ops; target_eltype = T),
                                x;
                                rtol = rtol,
                                atol = atol)
    end

    for acc in (4, 6, 8)
        ops = diagonal_spherical_operators(
            source;
            accuracy_order = acc,
            N = 32,
            R = 2.0,
            p = 2
        )
        check_wave_kernel_pair(ops, Float64; rtol = 1.0e-12, atol = 1.0e-12)
        check_wave_kernel_pair(ops, Float64x2; rtol = 1.0e-24, atol = 1.0e-24)
        check_wave_kernel_pair(ops, Float64x4; rtol = 1.0e-48, atol = 1.0e-48)

        rhs_ops = wave_kernel_operators(ops)
        pi_state = sin.(Float64.(ops.r))
        psi_state = cos.(Float64.(ops.r))
        dpi_sparse = similar(pi_state)
        dpsi_sparse = similar(pi_state)
        dpi_kernel = similar(pi_state)
        dpsi_kernel = similar(pi_state)

        wave_rhs!(dpi_sparse, dpsi_sparse, pi_state, psi_state, ops)
        wave_rhs!(dpi_kernel, dpsi_kernel, pi_state, psi_state, rhs_ops)
        @test dpi_kernel ≈ dpi_sparse rtol = 1.0e-12 atol = 1.0e-12
        @test dpsi_kernel ≈ dpsi_sparse rtol = 1.0e-12 atol = 1.0e-12
        wave_rhs!(dpi_kernel, dpsi_kernel, pi_state, psi_state, rhs_ops)
        @test (@allocated wave_rhs!(dpi_kernel,
                                     dpsi_kernel,
                                     pi_state,
                                     psi_state,
                                     rhs_ops)) == 0
    end

    for acc in (4, 6)
        ops = non_diagonal_spherical_operators(
            source;
            accuracy_order = acc,
            N = 32,
            R = 2.0,
            p = 2,
            mode = SafeMode(),
            target_eltype = Float64
        )
        check_wave_kernel_pair(ops, Float64; rtol = 1.0e-12, atol = 1.0e-12)
        check_wave_kernel_pair(ops, Float64x2; rtol = 1.0e-24, atol = 1.0e-24)
        check_wave_kernel_pair(ops, Float64x4; rtol = 1.0e-48, atol = 1.0e-48)
    end
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti / 60, digits = 3), " minutes")
