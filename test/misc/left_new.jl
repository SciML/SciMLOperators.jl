using SciMLOperators, LinearAlgebra, SparseArrays
using Random
using Test

using SciMLOperators: AdjointOperator, TransposedOperator

Random.seed!(0)
N = 8
K = 4

@testset "AdjointOperator" begin
    # Create a test matrix to use as an operator
    A = rand(ComplexF64, N, N)
    L = MatrixOperator(A)
    L_adj = L'
    
    @test (L_adj isa AdjointOperator) || (L_adj isa MatrixOperator && L_adj.A == A')
    @test size(L_adj) == (N, N)
    @test islinear(L_adj)
    
    # Test vectors
    v = rand(ComplexF64, N)
    u = rand(ComplexF64, N)
    w = zeros(ComplexF64, N)
    
    # Parameters
    p = nothing
    t = 0.0
    α = 2.0
    β = 0.5
    
    # Test standard adjoint multiplication
    @test L_adj * v ≈ A' * v
    
    # Test out-of-place new interface
    @test L_adj(v, u, p, t) ≈ A' * v
    
    # Test in-place new interface
    L_adj(w, v, u, p, t)
    @test w ≈ A' * v
    
    # Test in-place with scaling new interface
    w_orig = rand(ComplexF64, N)
    w .= w_orig
    L_adj(w, v, u, p, t, α, β)
    @test w ≈ α * (A' * v) + β * w_orig
    
    # Test with matrix input
    V = rand(ComplexF64, N, K)
    W = zeros(ComplexF64, N, K)
    
    # Test out-of-place with matrix
    @test L_adj(V, u, p, t) ≈ A' * V
    
    # Test in-place with matrix
    L_adj(W, V, u, p, t)
    @test W ≈ A' * V
    
    # Test in-place with scaling with matrix
    W_orig = rand(ComplexF64, N, K)
    W .= W_orig
    L_adj(W, V, u, p, t, α, β)
    @test W ≈ α * (A' * V) + β * W_orig
end

@testset "TransposedOperator" begin
    # Create a test matrix to use as an operator
    A = rand(ComplexF64, N, N)
    L = MatrixOperator(A)
    L_trans = transpose(L)
    
    @test (L_trans isa TransposedOperator) || (L_trans isa MatrixOperator && L_trans.A == transpose(A))
    @test size(L_trans) == (N, N)
    @test islinear(L_trans)
    
    # Test vectors
    v = rand(ComplexF64, N)
    u = rand(ComplexF64, N)
    w = zeros(ComplexF64, N)
    
    # Parameters
    p = nothing
    t = 0.0
    α = 2.0
    β = 0.5
    
    # Test standard transpose multiplication
    @test L_trans * v ≈ transpose(A) * v
    
    # Test out-of-place new interface
    @test L_trans(v, u, p, t) ≈ transpose(A) * v
    
    # Test in-place new interface
    L_trans(w, v, u, p, t)
    @test w ≈ transpose(A) * v
    
    # Test in-place with scaling new interface
    w_orig = rand(ComplexF64, N)
    w .= w_orig
    L_trans(w, v, u, p, t, α, β)
    @test w ≈ α * (transpose(A) * v) + β * w_orig
    
    # Test with matrix input
    V = rand(ComplexF64, N, K)
    W = zeros(ComplexF64, N, K)
    
    # Test out-of-place with matrix
    @test L_trans(V, u, p, t) ≈ transpose(A) * V
    
    # Test in-place with matrix
    L_trans(W, V, u, p, t)
    @test W ≈ transpose(A) * V
    
    # Test in-place with scaling with matrix
    W_orig = rand(ComplexF64, N, K)
    W .= W_orig
    L_trans(W, V, u, p, t, α, β)
    @test W ≈ α * (transpose(A) * V) + β * W_orig
end

@testset "AdjointOperator vs TransposedOperator Real Case" begin
    # For real matrices, adjoint and transpose should be equivalent
    A_real = rand(N, N)
    L = MatrixOperator(A_real)
    L_adj = L'
    L_trans = transpose(L)
    
    v = rand(N)
    u = rand(N)
    p = nothing
    t = 0.0
    
    # Test that they give the same results
    @test L_adj(v, u, p, t) ≈ L_trans(v, u, p, t)
    
    w_adj = zeros(N)
    w_trans = zeros(N)
    
    L_adj(w_adj, v, u, p, t)
    L_trans(w_trans, v, u, p, t)
    @test w_adj ≈ w_trans
    
    α, β = 2.0, 0.5
    w_orig = rand(N)
    w_adj .= w_orig
    w_trans .= w_orig
    
    L_adj(w_adj, v, u, p, t, α, β)
    L_trans(w_trans, v, u, p, t, α, β)
    @test w_adj ≈ w_trans
end

@testset "AdjointOperator vs TransposedOperator Complex Case" begin
    # For complex matrices, adjoint and transpose should be different
    A_complex = rand(ComplexF64, N, N)
    L = MatrixOperator(A_complex)
    L_adj = L'
    L_trans = transpose(L)
    
    v = rand(ComplexF64, N)
    u = rand(ComplexF64, N)
    p = nothing
    t = 0.0
    
    # Verify that they give different results
    @test !(L_adj(v, u, p, t) ≈ L_trans(v, u, p, t)) # Should be different
    
    # Verify against expected results
    @test L_adj(v, u, p, t) ≈ A_complex' * v
    @test L_trans(v, u, p, t) ≈ transpose(A_complex) * v
    
    # Test with in-place operations
    w_adj = zeros(ComplexF64, N)
    w_trans = zeros(ComplexF64, N)
    
    L_adj(w_adj, v, u, p, t)
    L_trans(w_trans, v, u, p, t)
    
    @test w_adj ≈ A_complex' * v
    @test w_trans ≈ transpose(A_complex) * v
    @test !(w_adj ≈ w_trans) # Should be different
end

@testset "Left Multiplication" begin
    A = rand(N, N)
    L = MatrixOperator(A)
    
    v = rand(N)
    u = v' * L
    
    # Test that left multiplication works
    @test u ≈ v' * A
end


@testset "Nested Adjoint/Transpose Operations" begin
    A = rand(ComplexF64, N, N)
    L = MatrixOperator(A)
    
    # Double adjoint should get back the original
    L_adj_adj = L''
    
    v = rand(ComplexF64, N)
    u = rand(ComplexF64, N)
    p = nothing
    t = 0.0
    
    @test L_adj_adj(v, u, p, t) ≈ L(v, u, p, t)
    
    # Adjoint of transpose
    L_trans_adj = (transpose(L))'
    @test L_trans_adj(v, u, p, t) ≈ conj(A) * v
    
    # Transpose of adjoint
    L_adj_trans = transpose(L')
    @test L_adj_trans(v, u, p, t) ≈ conj(A) * v
end

@testset "Rectangular Operators" begin
    # Test with non-square matrices
    A_rect = rand(ComplexF64, N, 2*N)
    L_rect = MatrixOperator(A_rect)
    
    L_adj = L_rect'
    L_trans = transpose(L_rect)
    
    @test size(L_adj) == (2*N, N)
    @test size(L_trans) == (2*N, N)
    
    v = rand(ComplexF64, N)
    u = rand(ComplexF64, 2*N)
    w_adj = zeros(ComplexF64, 2*N)
    w_trans = zeros(ComplexF64, 2*N)
    
    p = nothing
    t = 0.0
    
    # Test out-of-place
    @test L_adj(v, u, p, t) ≈ A_rect' * v
    @test L_trans(v, u, p, t) ≈ transpose(A_rect) * v
    
    # Test in-place
    L_adj(w_adj, v, u, p, t)
    L_trans(w_trans, v, u, p, t)
    
    @test w_adj ≈ A_rect' * v
    @test w_trans ≈ transpose(A_rect) * v
    
    # Test vector sizes match correctly
    v_big = rand(ComplexF64, 2*N)
    w_small = zeros(ComplexF64, N)
    
    # Test adjoint of adjoint (should match original operation)
    L_rect(w_small, v_big, u, p, t)
    expected = A_rect * v_big
    @test w_small ≈ expected
end