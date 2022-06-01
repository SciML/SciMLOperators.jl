using SciMLOperators, LinearAlgebra
using Random

Random.seed!(0)
N = 8

@testset "DiffEqArrayOperator" begin
    u = rand(N)
    p = nothing
    t = 0

    A  = rand(N,N)
    At = A'

    AA  = DiffEqArrayOperator(A)
    AAt = AA'

    @test AA  isa DiffEqArrayOperator
    @test AAt isa DiffEqArrayOperator

    FF  = factorize(AA)
    FFt = FF'

    @test FF  isa FactorizedDiffEqArrayOperator
    @test FFt isa FactorizedDiffEqArrayOperator

    @test eachindex(A)  === eachindex(AA)
    @test eachindex(A') === eachindex(AAt) === eachindex(DiffEqArrayOperator(At))

    @test A  ≈ convert(AbstractMatrix, AA ) ≈ convert(AbstractMatrix, FF )
    @test At ≈ convert(AbstractMatrix, AAt) ≈ convert(AbstractMatrix, FFt)

    @test A  ≈ Matrix(AA ) ≈ Matrix(FF )
    @test At ≈ Matrix(AAt) ≈ Matrix(FFt)

    @test A  * u ≈ AA(u,p,t)  ≈ FF(u,p,t)
    @test At * u ≈ AAt(u,p,t) ≈ FFt(u,p,t)

    @test A  \ u ≈ AA  \ u ≈ FF  \ u
    @test At \ u ≈ AAt \ u ≈ FFt \ u
end
#
