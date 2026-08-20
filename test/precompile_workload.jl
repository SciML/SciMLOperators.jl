using LinearAlgebra
using SciMLOperators
using Test

@testset "Precompile workload APIs" begin
    v = [1.0, 2.0]
    A = MatrixOperator([2.0 1.0; 1.0 3.0])
    D = DiagonalOperator([2.0, 3.0])

    @test A * v == [4.0, 7.0]
    w = zeros(2)
    @test mul!(w, A, v) === w
    @test w == [4.0, 7.0]

    @test ScalarOperator(2.0) * v == [2.0, 4.0]
    F = FunctionOperator(
        (v, u, p, t) -> 2 .* v, v;
        ifcache = false, isconstant = true, islinear = true
    )
    @test F * v == [2.0, 4.0]

    @test (A + D) * v == [6.0, 13.0]
    @test (A * D) * v == [10.0, 20.0]
    @test (A ∘ D) * v == [10.0, 20.0]
    @test (2.0 * A) * v == [8.0, 14.0]

    B = BlockDiagonalOperator(A, D)
    @test B * [1.0, 2.0, 3.0, 4.0] == [4.0, 7.0, 6.0, 12.0]

    T = kron(IdentityOperator(2), D)
    @test T * [1.0, 2.0, 3.0, 4.0] == [2.0, 6.0, 6.0, 12.0]

    cached = cache_operator(A * D, v)
    @test cached * v == [10.0, 20.0]
    w = zeros(2)
    @test mul!(w, cached, v) === w
    @test w == [10.0, 20.0]
end
