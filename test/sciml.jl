using SciMLOperators, LinearAlgebra
using Random

Random.seed!(0)
N = 8

@testset "MatrixOperator" begin
    u = rand(N)
    p = nothing
    t = 0
    α = rand()
    β = rand()

    A  = rand(N,N)
    At = A'

    AA  = MatrixOperator(A)
    AAt = AA'

    @test AA  isa MatrixOperator
    @test AAt isa MatrixOperator

    FF  = factorize(AA)
    FFt = FF'

    @test FF  isa FactorizedOperator
    @test FFt isa FactorizedOperator

    @test eachindex(A)  === eachindex(AA)
    @test eachindex(A') === eachindex(AAt) === eachindex(MatrixOperator(At))

    @test A  ≈ convert(AbstractMatrix, AA ) ≈ convert(AbstractMatrix, FF )
    @test At ≈ convert(AbstractMatrix, AAt) ≈ convert(AbstractMatrix, FFt)

    @test A  ≈ Matrix(AA ) ≈ Matrix(FF )
    @test At ≈ Matrix(AAt) ≈ Matrix(FFt)

    @test A  * u ≈ AA(u,p,t)  ≈ FF(u,p,t)
    @test At * u ≈ AAt(u,p,t) ≈ FFt(u,p,t)

    @test A  \ u ≈ AA  \ u ≈ FF  \ u
    @test At \ u ≈ AAt \ u ≈ FFt \ u


    v=rand(N); @test mul!(v, AA, u) ≈ A * u
    v=rand(N); w=copy(v); @test mul!(v, AA, u, α, β) ≈ α*A*u + β*w
end

@testset "SciMLFunctionOperator" begin
end

@testset "Operator Algebra" begin
    # try out array arithmatic
end
#
