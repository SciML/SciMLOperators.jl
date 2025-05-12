using SciMLOperators, LinearAlgebra
using Random
using Test

Random.seed!(0)
N = 8
K = 12

@testset "BatchedDiagonalOperator Basic" begin
    # Test vectors
    u = rand(N, K)   # Update vector
    v = rand(N, K)   # Action vector
    w = zeros(N, K)  # Output vector
    
    d = rand(N, K)   # Diagonal values
    p = nothing
    t = 0.0
    α = rand()
    β = rand()

    # Create operator
    L = DiagonalOperator(d)
    
    # Test properties
    @test isconstant(L)
    @test issquare(L)
    @test islinear(L)
    
    # Test original interface
    @test L * v ≈ d .* v
    
    # Test with new interface - same vector for update and action
    @test L(u, u, p, t) ≈ d .* u
    
    # Test with different vectors for update and action
    @test L(v, u, p, t) ≈ d .* v
    
    # Test in-place operation
    copy!(w, zeros(N, K))
    L(w, v, u, p, t)
    @test w ≈ d .* v
    
    # Test in-place with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    L(w, v, u, p, t, α, β)
    @test w ≈ α * (d .* v) + β * orig_w
    
    # Test division operations
    @test L \ u ≈ d .\ u
end

@testset "BatchedDiagonalOperator Update" begin
    # Test vectors
    u = rand(N, K)    # Update vector
    v = rand(N, K)    # Action vector
    w = zeros(N, K)   # Output vector
    
    d = zeros(N, K)   # Initial diagonal values
    p = rand(N, K)    # Parameters
    t = rand()        # Time
    
    # Create operator with update functions
    D = DiagonalOperator(d;
        update_func = (diag, u, p, t) -> p * t,
        update_func! = (diag, u, p, t) -> diag .= p * t)

    # Test properties
    @test !isconstant(D)
    @test issquare(D)
    @test islinear(D)

    # Expected result after update
    expected = (p * t) .* v
    
    # Test with new interface
    @test D(v, u, p, t) ≈ expected
    
    # Test in-place operation
    copy!(w, zeros(N, K))
    D(w, v, u, p, t)
    @test w ≈ expected
    
    # Test in-place with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    α = rand()
    β = rand()
    D(w, v, u, p, t, α, β)
    @test w ≈ α * expected + β * orig_w
end

@testset "BatchedDiagonalOperator Special Cases" begin
    # Test edge cases
    d_zero = zeros(N, K)
    u = rand(N, K)     # Update vector
    v = rand(N, K)     # Action vector
    w = rand(N, K)     # Output vector
    p = nothing
    t = 0.0
    
    # Test with zero diagonal
    Z = DiagonalOperator(d_zero)
    
    # Test multiplication with zero
    @test Z(v, u, p, t) ≈ zeros(N, K)
    
    # Test in-place application with zero
    copy!(w, rand(N, K))
    Z(w, v, u, p, t)
    @test w ≈ zeros(N, K)
    
    # Test in-place with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    α = rand()
    β = rand()
    Z(w, v, u, p, t, α, β)
    @test w ≈ β * orig_w  # Since α * zero = 0
end