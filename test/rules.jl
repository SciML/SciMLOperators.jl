#
using SciMLOperators, LinearAlgebra, ChainRules
using Test, ChainRulesTestUtils, Random

Random.seed!(0)
N = 8
K = 12

@testset "MatrixOperator" begin
    A = MatrixOperator(rand(N,N))
    u = rand(N)
    
    A = Diagonal(rand(N))
    test_frule(*, A, u)

    A = MatrixOperator(rand(N,N))
    test_frule(*, A, u)

#   test_rrule(*, A, u)
end

