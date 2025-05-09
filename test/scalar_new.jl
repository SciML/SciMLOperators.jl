using SciMLOperators
using SciMLOperators: AbstractSciMLScalarOperator,
                      ComposedScalarOperator,
                      AddedScalarOperator,
                      InvertedScalarOperator,
                      IdentityOperator,
                      AddedOperator,
                      ScaledOperator

using LinearAlgebra, Random, Test

Random.seed!(0)
N = 8
K = 12

@testset "ScalarOperator New Interface" begin
    a = rand()
    b = rand()
    x = rand()
    α = ScalarOperator(x)
    
    # Test vectors
    u = rand(N, K)  # Update vector
    v = rand(N, K)  # Action vector
    w = zeros(N, K)  # Result vector

    # Scalar Number tests
    su = rand()  # Scalar update value
    sv = rand()  # Scalar action value
    
    # Test operator traits
    @test α isa ScalarOperator
    @test iscached(α)
    @test issquare(α)
    @test islinear(α)

    # Test with same vector for update and action
    @test α(u, u, nothing, 0.0) ≈ u * x
    
    # Test with different vectors for update and action
    @test α(v, u, nothing, 0.0) ≈ v * x
    
    # Test in-place operation
    copy!(w, zeros(N, K))
    α(w, v, u, nothing, 0.0)
    @test w ≈ v * x
    
    # Test in-place with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    α(w, v, u, nothing, 0.0, a, b)
    @test w ≈ a * (x * v) + b * orig_w
    
    # Test with updating ScalarOperator
    α_update = ScalarOperator(0.0; update_func = (a, u, p, t) -> p)
    p_val = 2.0
    t_val = 1.0
    
    @test !isconstant(α_update)
    
    # Test update then act - out of place
    @test α_update(v, u, p_val, t_val) ≈ v * p_val
    
    # Test update then act - in place
    copy!(w, zeros(N, K))
    α_update(w, v, u, p_val, t_val)
    @test w ≈ v * p_val
    
    # Test with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    α_update(w, v, u, p_val, t_val, a, b)
    @test w ≈ a * (p_val * v) + b * orig_w
    
    # Test operator combinations
    β = α_update + α
    @test β(v, u, p_val, t_val) ≈ v * (p_val + x)
    
    β = α_update * α
    @test β(v, u, p_val, t_val) ≈ v * (p_val * x)
    
    β = inv(α)
    @test β(v, u, nothing, 0.0) ≈ v * (1/x)
    
    # Test with matrix operators
    M = MatrixOperator(rand(N, N))
    L = α + M
    @test L(v, u, nothing, 0.0) ≈ x * v + M * v
    
    L = α * M
    @test L(v, u, nothing, 0.0) ≈ x * (M * v)
end

@testset "ScalarOperator with Number arguments" begin
    x = rand()
    α = ScalarOperator(x)
    
    u = rand()  # Scalar update value
    v = rand()  # Scalar action value
    p = nothing
    t = 0.0
    
    # Test with scalar values
    @test α(v, u, p, t) ≈ v * x
    
    # Test with updating ScalarOperator
    α_update = ScalarOperator(0.0; update_func = (a, u, p, t) -> p)
    p_val = 2.0
    t_val = 1.0
    
    @test α_update(v, u, p_val, t_val) ≈ v * p_val
    
    # Ensure we still get error for in-place operations with Numbers
    @test_throws ArgumentError α(0.0, v, u, p, t)
    @test_throws ArgumentError α(0.0, v, u, p, t, 1.0, 2.0)
    
    # Test operator combinations with scalars
    β = α_update + α
    @test β(v, u, p_val, t_val) ≈ v * (p_val + x)
    
    β = α_update * α
    @test β(v, u, p_val, t_val) ≈ v * (p_val * x)
    
    β = inv(α)
    @test β(v, u, p, t) ≈ v * (1/x)
end

@testset "ScalarOperator with kwargs" begin
    # Test scalar operator which expects keyword argument to update
    γ = ScalarOperator(0.0; update_func = (args...; dtgamma) -> dtgamma,
                      accepted_kwargs = (:dtgamma,))
    
    u = rand(N, K)
    v = rand(N, K)
    p = nothing
    t = 0.0
    dtgamma = rand()
    
    # Test with keyword arguments
    @test γ(v, u, p, t; dtgamma) ≈ v * dtgamma
    
    # Test with combined operators
    γ_added = γ + ScalarOperator(1.0)
    @test γ_added(v, u, p, t; dtgamma) ≈ v * (dtgamma + 1.0)
end

@testset "ScalarOperator with AbstractArrays and kwargs" begin
    # Simple scalar operator
    a = rand()
    α = ScalarOperator(a)
    
    # Arrays to test with
    u = rand(N, K)  # Update vector
    v = rand(N, K)  # Action vector
    w = zeros(N, K)  # Output vector
    
    # Test basic operations with abstract arrays
    @test α(v, u, nothing, 0.0) ≈ v * a
    
    # Test in-place operation with abstract arrays
    copy!(w, zeros(N, K))
    α(w, v, u, nothing, 0.0)
    @test w ≈ v * a
    
    # Test in-place with scaling
    b = rand()
    c = rand()
    w_orig = rand(N, K)
    copy!(w, w_orig)
    α(w, v, u, nothing, 0.0, b, c)
    @test w ≈ b * (a * v) + c * w_orig
    
    # Test with keyword arguments
    γ = ScalarOperator(0.0; update_func = (args...; dtgamma) -> dtgamma,
                      accepted_kwargs = (:dtgamma,))
    
    dtgamma = rand()
    
    # Test with basic keyword argument
    @test γ(v, u, nothing, 0.0; dtgamma) ≈ v * dtgamma
    
    # Test in-place with keyword argument
    copy!(w, zeros(N, K))
    γ(w, v, u, nothing, 0.0; dtgamma)
    @test w ≈ v * dtgamma
    
    # Test composite operators with keyword arguments
    γ_added = γ + ScalarOperator(1.0)
    @test γ_added(v, u, nothing, 0.0; dtgamma) ≈ v * (dtgamma + 1.0)
    
    γ_mul = γ * ScalarOperator(2.0)
    @test γ_mul(v, u, nothing, 0.0; dtgamma) ≈ v * (dtgamma * 2.0)
    
    γ_inv = inv(γ)
    @test γ_inv(v, u, nothing, 0.0; dtgamma) ≈ v * (1.0/dtgamma)
end