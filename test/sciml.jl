using SciMLOperators, LinearAlgebra
using Random

Random.seed!(0)
N = 8

@testset "MatrixOperator" begin
    u = rand(N)
    p = nothing
    t = 0

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
end

@testset "AffineOperator" begin
    u = rand(N)
    A = rand(N,N)
    b = rand(N)
    α = rand()
    β = rand()

    L = AffineOperator(MatrixOperator(A), b)

    @test L * u ≈ A * u + b
    v=rand(N); @test mul!(v, L, u) ≈ A * u + b
    v=rand(N); @test mul!(v, L, u, α, β) ≈ α*(A*u + b) + β*v

    @test L \ u ≈ A \ (u - b)
#   v=rand(N); @test ldiv!(v, L, u) ≈ A \ (u-b)
#   v=rand(N); @test ldiv!(L, u) ≈ A \ (u-b)
end

@testset "SciMLFunctionOperator" begin
end

@testset "Operator Algebra" begin
    # try out array arithmatic
end
#
