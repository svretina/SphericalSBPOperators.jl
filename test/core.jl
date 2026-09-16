const CORE = SphericalSBPOperators

@testset "core: tolerances and sparse cleanup" begin
    @test CORE._default_atol(Float64) == 1.0e-12
    @test CORE._default_atol(Rational{BigInt}) == 0 // 1
    @test CORE._resolve_atol(Float64, nothing) == 1.0e-12
    @test CORE._resolve_atol(Float64, 1 // 10) == 0.1
    @test CORE._resolve_atol(Rational{BigInt}, 1 // 10) == 1 // 10
    @test_throws ArgumentError CORE._resolve_atol(Rational{BigInt}, 0.1)

    @test CORE._maxabs(Int[]) == 0
    @test CORE._maxabs([-2, 1, 3]) == 3
    @test CORE._maxabs_sparse(sparse(Int[], Int[], Int[], 3, 3)) == 0
    @test CORE._maxabs_sparse(sparse([1, 3], [2, 1], [-2, 5], 3, 3)) == 5

    A = sparse([1, 1, 2], [1, 2, 2], [1.0, eps(Float64), 2.0], 2, 2)
    @test CORE.snap_sparse!(A) === A
    @test A[1, 2] == 0.0
    @test nnz(A) == 2
    exact = sparse([1, 1], [1, 2], Rational{BigInt}[1 // 1, 1 // 10^30], 2, 2)
    CORE.snap_sparse!(exact)
    @test exact[1, 2] == 1 // 10^30
end

@testset "core: Cartesian grid and folding maps" begin
    D, x, G, H = CORE._build_full_grid_objects(
        SOURCE; accuracy_order = 4, N = 8, R = 8 // 1, mode = SafeMode())
    @test length(x) == 17
    @test x[9] == 0 // 1
    @test size(G) == size(H) == size(D) == (17, 17)
    @test CORE._boundary_closure_width_from_operator(D) == 4

    _, xs, Gs, Hs = CORE._build_full_grid_objects(
        SOURCE; accuracy_order = 4, N = 8, R = 15 // 2, mode = SafeMode(),
        include_origin = false)
    @test length(xs) == 16
    @test all(!iszero, xs)
    @test size(Gs) == size(Hs) == (16, 16)
    @test_throws ArgumentError CORE._build_full_grid_objects(
        SOURCE; accuracy_order = 4, N = 0, R = 1, mode = SafeMode())

    xfull = Rational{BigInt}[-2, -1, 0, 1, 2]
    r, Rop, Eeven, Eodd = CORE._build_folding_operators(xfull; atol = zero(eltype(xfull)))
    @test r == Rational{BigInt}[0, 1, 2]
    @test Rop * xfull == r
    even_data = Rational{BigInt}[3, 5, 7]
    odd_data = Rational{BigInt}[0, 5, 7]
    @test Eeven * even_data == [7, 5, 3, 5, 7]
    @test Eodd * odd_data == [-7, -5, 0, 5, 7]
    @test_throws ArgumentError CORE._build_folding_operators(
        Rational{BigInt}[-2, -1, 1, 2]; atol = zero(Rational{BigInt}))

    staggered_r, _, staggered_even, staggered_odd = CORE._build_folding_operators(
        Rational{BigInt}[-3, -1, 1, 3]; atol = zero(Rational{BigInt}), require_origin = false)
    @test staggered_r == Rational{BigInt}[1, 3]
    @test staggered_even * [2 // 1, 4 // 1] == [4, 2, 2, 4]
    @test staggered_odd * [2 // 1, 4 // 1] == [-4, -2, 2, 4]

    lookup = CORE._build_half_lookup([0.0, 1.0, 2.0], 1.0e-12)
    @test CORE._lookup_half_index(1.0 + 1.0e-14, lookup, [0.0, 1.0, 2.0]) == 2
    @test_throws ArgumentError CORE._lookup_half_index(3.0, lookup, [0.0, 1.0, 2.0])
end

@testset "core: scaling, closure diagnostics, and exact solves" begin
    @test CORE._uniform_spacing([0, 2, 4]; atol = 0) == 2
    @test CORE._uniform_spacing([0.0, 0.1, 0.2 + 1.0e-15]; atol = 1.0e-12) ≈ 0.1
    @test_throws ArgumentError CORE._uniform_spacing([0, 2, 5]; atol = 0)
    @test_throws ArgumentError CORE._uniform_spacing([1]; atol = 0)
    @test CORE._default_scale_eltype(1.0) === Float64
    @test CORE._default_scale_eltype(1 // 1) === Rational{BigInt}

    scaled = CORE._scale_sparse_matrix(sparse([1, 2], [1, 2], [1, 3], 2, 2),
                                        1 // 2, Rational{BigInt}; snap_factor = 64.0)
    @test Matrix(scaled) == Rational{BigInt}[1 // 2 0; 0 3 // 2]

    G = sparse([1, 2, 2, 3, 3, 4, 4, 5], [1, 1, 2, 2, 3, 3, 4, 5],
               ones(Int, 8), 5, 5)
    closure = CORE._closure_diagnostics(G)
    @test closure.closure_width_left == 1
    @test closure.closure_width_right == 1
    @test CORE._closure_diagnostics(spzeros(Int, 0, 0)).closure_width_right == 0

    Q = Rational{BigInt}
    A = Q[1 2; 3 4]
    @test CORE._solve_exact_linear_system(A, Q[5, 11]) == Q[1, 2]
    @test CORE._solve_exact_linear_system(Q[1 1], Q[3]) == Q[3, 0]
    @test CORE._solve_exact_linear_system(zeros(Q, 0, 2), Q[]) == Q[0, 0]
    @test_throws ArgumentError CORE._solve_exact_linear_system(Q[1 1; 1 1], Q[1, 2])
    @test_throws DimensionMismatch CORE._solve_exact_linear_system(Q[1 1], Q[1, 2])
end

@testset "core: exact SBP6 support" begin
    Q = Rational{BigInt}
    @test CORE._as_big_rational(2) == 2 // 1
    @test CORE._as_big_rational(3 // 7) == 3 // 7
    @test CORE._as_big_rational(0.5) == 1 // 2
    @test_throws ArgumentError CORE._as_big_rational(Inf)

    diagonal = sparse([1, 2], [1, 2], Q[2 // 1, 3 // 1], 2, 2)
    @test CORE._extract_diagonal(diagonal) == Q[2, 3]
    @test_throws ArgumentError CORE._extract_diagonal(sparse([1, 1], [1, 2], Q[1, 1], 2, 2))
    @test CORE._is_symmetric(sparse(Q[2 1; 1 2]))
    @test !CORE._is_symmetric(sparse(Q[2 1; 0 2]))
    @test CORE._is_positive_definite(sparse(Q[2 1; 1 2]))
    @test !CORE._is_positive_definite(sparse(Q[1 2; 2 1]))

    r = Q[0, 1]
    S = sparse(Q[1 0; 0 2])
    V = sparse(Q[1 0; 0 3])
    Geven = sparse(Q[0 1; 1 0])
    built = CORE.sbp6_construct_divergence(S, V, Geven, r; p = 2)
    @test built.B == sparse(Q[0 0; 0 1])
    @test built.RHS == built.B - Geven' * V
    @test S * built.D + Geven' * V == built.B
    @test_throws ArgumentError CORE.sbp6_construct_divergence(
        sparse(Q[0 0; 0 1]), V, Geven, r; p = 2)

    setup = CORE.sbp6_scalar_mass_gradient(SOURCE;
        accuracy_order = 6, points = 21, N = 20, R = 20 // 1, p = 2,
        mode = SafeMode())
    @test setup.r[1] == 0 // 1
    @test length(setup.r) == 21
    @test setup.Sdiag == CORE._extract_diagonal(setup.S)
    @test setup.Hcart_half == transpose(setup.Hcart_half)
    @test_throws ArgumentError CORE.sbp6_scalar_mass_gradient(SOURCE;
        points = 1, N = 0, R = 1, mode = SafeMode())
end
