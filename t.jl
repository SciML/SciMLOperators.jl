#
using SciMLOperators, LinearAlgebra
using SciMLOperators: InvertibleOperator, ⊗
using Test
@testset "TensorProductOperator" begin #for K in [1, 12]
    K = 2
    m1, n1 = 3 , 5
    m2, n2 = 7 , 11
    m3, n3 = 13, 17
    
    A = rand(m1, n1)
    B = rand(m2, n2)
    C = rand(m3, n3)
    α = rand()
    β = rand()
    
    AB  = kron(A, B)
    ABC = kron(A, B, C)
    
    # Inputs
    u2 = rand(n1*n2, K)
    u3 = rand(n1*n2*n3, K)
    
    # Outputs
    v2 = rand(m1*m2, K)
    v3 = rand(m1*m2*m3, K)
    
    opAB  = TensorProductOperator(A, B)
    opABC = TensorProductOperator(A, B, C)

    @test AB \ v2 ≈ opAB \ v2
    @test ABC \ v3 ≈ opABC \ v3

    opAB  = cache_operator(opAB,  u2)
    opABC = cache_operator(opABC, u3)
    
    M3 = m1*m2*m3

    v3=rand(M3,K); w3=copy(v3);
    
    (K == 1) && @test_broken mul!(v3, opABC, u3, α, β) ≈ α*ABC*u3 + β*w3
end #end
