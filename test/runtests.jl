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

@testset "core folding utilities" begin
    Dfull, xfull, Gfull, Hfull = SphericalSBPOperators._build_full_grid_objects(
        SOURCE; accuracy_order = 4, N = 24, R = 24, mode = SafeMode())
    r, Rop, Eeven, Eodd = SphericalSBPOperators._build_folding_operators(
        xfull; atol = zero(eltype(xfull)))
    @test size(Gfull) == size(Dfull)
    @test size(Hfull) == (length(xfull), length(xfull))
    @test r[1] == zero(eltype(r))
    @test size(sparse(Rop * Gfull * Eeven)) == (length(r), length(r))
    @test size(sparse(Rop * Gfull * Eodd)) == (length(r), length(r))
end
