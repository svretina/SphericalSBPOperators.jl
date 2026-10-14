using Test
using SparseArrays: sparse, spzeros, nnz
using SummationByPartsOperators: MattssonNordström2004, SafeMode
using SphericalSBPOperators

const SOURCE = MattssonNordström2004()

include("core.jl")

@testset "public spherical operator API" begin
    collocated = spherical_operators(SOURCE; accuracy_order = 4, N = 20,
                                     R = 20, p = 2, mode = SafeMode())
    @test collocated isa NonDiagonalMassSphericalOperators
    @test has_origin_node(collocated)
    @test scalar_mass(collocated) === collocated.S
    @test vector_mass(collocated) === collocated.V
    @test size(collocated.D) == (21, 21)

    paper_sbp6 = spherical_operators(SOURCE; accuracy_order = 6, N = 30,
                                     R = 30, p = 2, mode = SafeMode())
    @test paper_sbp6.S[1, 1] == 1430827147971 // 4664636476456
    @test paper_sbp6.V[2, 3] == 4996740529431 // 6413875155127

    staggered = spherical_operators(SOURCE; accuracy_order = 4, N = 20,
                                    R = 20, p = 2, grid = :staggered,
                                    mode = SafeMode())
    @test staggered isa StaggeredSphericalOperators
    @test !has_origin_node(staggered)
    @test scalar_mass(staggered) === staggered.H
    @test staggered.divergence_method === :standard
    @test size(staggered.D) == (20, 20)
    @test_throws ArgumentError spherical_operators(SOURCE; accuracy_order = 4,
                                                   N = 20, R = 20, grid = :invalid)
end

@testset "collocated SBP4 cylindrical operators" begin
    R = 20 // 1
    ops = spherical_operators(SOURCE; accuracy_order = 4, N = 20, R, p = 1,
                              mode = SafeMode())
    r = ops.r
    D = ops.D
    G = ops.Geven

    # The defining SBP identity and the cylindrical quadrature are exact.
    @test ops.S * D + G' * ops.V == ops.B
    @test sum(ops.S) == R^2 / 2

    # The folded gradient is a derivative on even fields.
    @test G * (r .^ 2) == 2 .* r
    @test (G * (r .^ 4))[1:(end - 4)] == 4 .* r[1:(end - 4)] .^ 3

    # The fourth-order closure imposes these covariant-derivative identities.
    Dr = D * r
    Dr3 = D * (r .^ 3)
    @test Dr[1:(end - 4)] == 2 .* ones(eltype(r), length(r) - 4)
    @test Dr3[1:5] == 4 .* r[1:5] .^ 2
    @test ops.accuracy_order == 4

    gradient_errors = Float64[]
    divergence_errors = Float64[]
    for N in (20, 40, 80)
        refined = spherical_operators(SOURCE; accuracy_order = 4, N, R = 1.0,
                                      p = 1, mode = SafeMode())
        r = refined.r
        interior = 10:(length(r) - 9)
        push!(gradient_errors,
              maximum(abs.(refined.Geven * cos.(r) .+ sin.(r))[interior]))
        flux = r .* exp.(-r .^ 2)
        exact_divergence = (2 .- 2 .* r .^ 2) .* exp.(-r .^ 2)
        push!(divergence_errors,
              maximum(abs.(refined.D * flux .- exact_divergence)[interior]))
    end
    @test all(gradient_errors[1:(end - 1)] ./ gradient_errors[2:end] .> 8)
    @test all(divergence_errors[1:(end - 1)] ./ divergence_errors[2:end] .> 8)
end

@testset "core folding utilities" begin
    Dfull, xfull, Gfull, Hfull = SphericalSBPOperators._build_full_grid_objects(SOURCE;
                                                                                accuracy_order = 4,
                                                                                N = 24,
                                                                                R = 24,
                                                                                mode = SafeMode())
    r, Rop, Eeven, Eodd = SphericalSBPOperators._build_folding_operators(xfull;
                                                                         atol = zero(eltype(xfull)))
    @test size(Gfull) == size(Dfull)
    @test size(Hfull) == (length(xfull), length(xfull))
    @test r[1] == zero(eltype(r))
    @test size(sparse(Rop * Gfull * Eeven)) == (length(r), length(r))
    @test size(sparse(Rop * Gfull * Eodd)) == (length(r), length(r))
end
