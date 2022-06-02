using SciMLOperators, LinearAlgebra
using Random

Random.seed!(0)
N = 8

@testset "SciMLMatrixOperator" begin
    u = rand(N)
    p = nothing
    t = 0

    A  = rand(N,N)
    At = A'

    AA  = SciMLMatrixOperator(A)
    AAt = AA'

    @test AA  isa SciMLMatrixOperator
    @test AAt isa SciMLMatrixOperator

    FF  = factorize(AA)
    FFt = FF'

    @test FF  isa SciMLFactorizedOperator
    @test FFt isa SciMLFactorizedOperator

    @test eachindex(A)  === eachindex(AA)
    @test eachindex(A') === eachindex(AAt) === eachindex(SciMLMatrixOperator(At))

    @test A  ≈ convert(AbstractMatrix, AA ) ≈ convert(AbstractMatrix, FF )
    @test At ≈ convert(AbstractMatrix, AAt) ≈ convert(AbstractMatrix, FFt)

    @test A  ≈ Matrix(AA ) ≈ Matrix(FF )
    @test At ≈ Matrix(AAt) ≈ Matrix(FFt)

    @test A  * u ≈ AA(u,p,t)  ≈ FF(u,p,t)
    @test At * u ≈ AAt(u,p,t) ≈ FFt(u,p,t)

    @test A  \ u ≈ AA  \ u ≈ FF  \ u
    @test At \ u ≈ AAt \ u ≈ FFt \ u
end

@testset "SciMLFunctionOperator" begin
end

@testset "Operator Algebra" begin
    # try out array arithmatic
end
#
