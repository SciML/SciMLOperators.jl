using SciMLOperators, LinearAlgebra
using Random
using Test

Random.seed!(0)
@testset "WOperator" begin
    J = rand(12, 12)
    u = rand(12)
    M = I(12)
    gamma = 1/123
    W = WOperator{true}(M, gamma, J, u)
    
    @test convert(AbstractMatrix, W) ≈ J - M/gamma
end
