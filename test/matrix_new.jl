using SciMLOperators, LinearAlgebra
using Random
using Test

using SciMLOperators: InvertibleOperator, InvertedOperator, ⊗

Random.seed!(0)
N = 8
K = 12

@testset "MatrixOperator New Interface" begin
    # Test vectors
    u = rand(N, K)  # Update vector
    v = rand(N, K)  # Action vector
    w = zeros(N, K)  # Result vector
    
    # Test parameters
    p = nothing
    t = 0
    α = rand()
    β = rand()

    # Create test matrix
    A = rand(N, N)
    AA = MatrixOperator(A)

    # Test out-of-place operation with same update and action vectors
    @test AA(u, u, p, t) ≈ A * u
    
    # Test out-of-place operation with different update and action vectors
    @test AA(v, u, p, t) ≈ A * v
    
    # Test in-place operation
    copy!(w, zeros(N, K))
    AA(w, v, u, p, t)
    @test w ≈ A * v
    
    # Test in-place with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    AA(w, v, u, p, t, α, β)
    @test w ≈ α * (A * v) + β * orig_w
end

@testset "MatrixOperator with updates New Interface" begin
    # Test vectors
    u = rand(N, K)  # Update vector
    v = rand(N, K)  # Action vector
    w = zeros(N, K)  # Result vector
    
    # Test parameters
    p = rand(N)
    t = rand()
    α = rand()
    β = rand()

    # Create updating matrix operator
    L = MatrixOperator(zeros(N, N);
        update_func = (A, u, p, t) -> p * p',
        update_func! = (A, u, p, t) -> A .= p * p')

    @test !isconstant(L)

    # Expected updated matrix
    A_expected = p * p'
    
    # Test out-of-place operation with different update and action vectors
    @test L(v, u, p, t) ≈ A_expected * v
    
    # Test in-place operation
    copy!(w, zeros(N, K))
    L(w, v, u, p, t)
    @test w ≈ A_expected * v
    
    # Test in-place with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    L(w, v, u, p, t, α, β)
    @test w ≈ α * (A_expected * v) + β * orig_w
end

@testset "InvertibleOperator New Interface" begin
    # Test vectors
    u = rand(N, K)  # Update vector
    v = rand(N, K)  # Action vector
    w = zeros(N, K)  # Result vector
    
    # Test parameters
    p = nothing
    t = 0
    α = rand()
    β = rand()
    
    # Create test matrix
    A = rand(N, N)
    AA = MatrixOperator(A)
    FF = factorize(AA)
    
    # Test out-of-place operation
    @test FF(v, u, p, t) ≈ A * v
    
    # Test in-place operation
    copy!(w, zeros(N, K))
    FF(w, v, u, p, t)
    @test w ≈ A * v
    
    # Test in-place with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    FF(w, v, u, p, t, α, β)
    @test w ≈ α * (A * v) + β * orig_w
end

@testset "DiagonalOperator New Interface" begin
    # Test vectors
    u = rand(N, K)  # Update vector
    v = rand(N, K)  # Action vector
    w = zeros(N, K)  # Result vector
    
    # Test parameters
    p = rand(N)
    t = rand()
    α = rand()
    β = rand()
    
    # Create diagonal operator
    D = DiagonalOperator(zeros(N);
        update_func = (diag, u, p, t) -> p * t,
        update_func! = (diag, u, p, t) -> diag .= p * t)
    
    # Expected diagonal matrix
    D_expected = Diagonal(p * t)
    
    # Test out-of-place operation
    @test D(v, u, p, t) ≈ D_expected * v
    
    # Test in-place operation
    copy!(w, zeros(N, K))
    D(w, v, u, p, t)
    @test w ≈ D_expected * v
    
    # Test in-place with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    D(w, v, u, p, t, α, β)
    @test w ≈ α * (D_expected * v) + β * orig_w
end

@testset "AffineOperator New Interface" begin
    # Test vectors
    u = rand(N, K)  # Update vector
    v = rand(N, K)  # Action vector
    w = zeros(N, K)  # Result vector
    
    # Test parameters
    p = rand(N, K)
    t = rand()
    α = rand()
    β = rand()
    
    # Create matrices
    A = rand(N, N)
    B = rand(N, N)
    
    # Create affine operator with updating bias term
    L = AffineOperator(A, B, zeros(N, K);
        update_func = (b, u, p, t) -> p * t,
        update_func! = (b, u, p, t) -> b .= p * t)
    
    @test !isconstant(L)
    
    # Expected bias term
    b_expected = p * t
    # Expected result
    result_expected = A * v + B * b_expected
    
    # Test out-of-place operation
    @test L(v, u, p, t) ≈ result_expected
    
    # Test in-place operation
    copy!(w, zeros(N, K))
    L(w, v, u, p, t)
    @test w ≈ result_expected
    
    # Test in-place with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    L(w, v, u, p, t, α, β)
    @test w ≈ α * result_expected + β * orig_w
end

@testset "AddVector New Interface" begin
    # Test vectors
    u = rand(N, K)  # Update vector
    v = rand(N, K)  # Action vector
    w = zeros(N, K)  # Result vector
    
    # Test parameters
    p = rand(N, K)
    t = rand()
    α = rand()
    β = rand()
    
    # Create AddVector operator with updating bias term
    L = AddVector(zeros(N, K);
        update_func = (b, u, p, t) -> p * t,
        update_func! = (b, u, p, t) -> b .= p * t)
    
    # Expected bias term
    b_expected = p * t
    # Expected result
    result_expected = v + b_expected
    
    # Test out-of-place operation
    @test L(v, u, p, t) ≈ result_expected
    
    # Test in-place operation
    copy!(w, zeros(N, K))
    L(w, v, u, p, t)
    @test w ≈ result_expected
    
    # Test in-place with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    L(w, v, u, p, t, α, β)
    @test w ≈ α * result_expected + β * orig_w
end