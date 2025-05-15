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

@testset "ScalarOperator Basic Operations" begin
    a = rand()
    b = rand()
    x = rand()
    α = ScalarOperator(x)
    u = rand(N, K)

    @test α isa ScalarOperator
    @test iscached(α)
    @test issquare(α)
    @test islinear(α)

    @test convert(Float32, α) isa Float32
    @test convert(ScalarOperator, a) isa ScalarOperator

    @test size(α) == ()
    @test isconstant(α)

    # Tests with the new interface
    v = copy(u)  # Action vector
    w = zeros(N, K)  # Output vector
    
    # lmul!/rmul! test equivalents
    result = α(v, u, nothing, 0.0)
    @test result ≈ v * x
    
    # Test in-place operations
    α(w, v, u, nothing, 0.0)
    @test w ≈ v * x
    
    # Test in-place operations with scaling
    orig_w = rand(N, K)
    copy!(w, orig_w)
    α(w, v, u, nothing, 0.0, a, b)
    @test w ≈ a * (x * v) + b * orig_w

    # Test scalar operations
    X = rand(N, K)  # Action vector
    Y = rand(N, K)  # Update vector
    Z = zeros(N, K)  # Output vector
    a_scalar = rand()
    aa = ScalarOperator(a_scalar)
    
    # Test axpy! equivalent
    result = aa(X, Y, nothing, 0.0)
    @test result ≈ a_scalar * X
end

@testset "ScalarOperator Combinations" begin
    x = rand()
    α = ScalarOperator(x)
    u = rand(N, K)  # Update vector
    v = rand(N, K)  # Action vector
    
    # Test scalar operator combinations
    β = α + α
    @test β isa AddedScalarOperator
    @test β(v, u, nothing, 0.0) ≈ x * v + x * v
    @test convert(Number, β) ≈ x + x

    β = α * α
    @test β isa ComposedScalarOperator
    @test β(v, u, nothing, 0.0) ≈ x * x * v
    @test convert(Number, β) ≈ x * x

    β = inv(α)
    @test β isa InvertedScalarOperator
    @test β(v, u, nothing, 0.0) ≈ (1 / x) * v
    @test convert(Number, β) ≈ 1 / x
    
    # Test combination with other operators
    for op in (MatrixOperator(rand(N, N)), SciMLOperators.IdentityOperator(N))
        L = α + op
        @test L isa SciMLOperators.AddedOperator
        @test L(v, u, nothing, 0.0) ≈ x * v + op * v
        
        L = α * op
        @test L isa SciMLOperators.ScaledOperator
        @test L(v, u, nothing, 0.0) ≈ x * (op * v)
    end
end

@testset "ScalarOperator scalar argument test" begin
    a = rand()
    u = rand()  # Update scalar
    v = rand()  # Action scalar
    p = nothing
    t = 0.0

    α = ScalarOperator(a)
    @test α(v, u, p, t) ≈ v * a
end

@testset "ScalarOperator update test" begin
    u = ones(N, K)  # Update vector
    v = ones(N, K)  # Action vector
    w = zeros(N, K)  # Output vector
    p = 2.0
    t = 4.0
    
    c = rand()
    d = rand()

    α = ScalarOperator(0.0; update_func = (a, u, p, t) -> p)
    β = ScalarOperator(0.0; update_func = (a, u, p, t) -> t)

    @test !isconstant(α)
    @test !isconstant(β)

    # Test update_coefficients
    update_coefficients!(α, u, p, t)
    update_coefficients!(β, u, p, t)

    @test convert(Number, α) ≈ p
    @test convert(Number, β) ≈ t

    # Test updated values with operators
    @test α(v, u, p, t) ≈ p * v
    @test β(v, u, p, t) ≈ t * v
    
    # Test in-place with scaling
    orig_w = rand(N, K)
    copy!(w, orig_w)
    α(w, v, u, p, t, c, d)
    @test w ≈ c * p * v + d * orig_w

    # Test operator combinations
    num = α + 2 / β * 3 - 4
    val = p + 2 / t * 3 - 4
    
    @test convert(Number, num) ≈ val

    # Test with keyword arguments
    γ = ScalarOperator(0.0; update_func = (args...; dtgamma) -> dtgamma,
                     accepted_kwargs = (:dtgamma,))
    
    dtgamma = rand()
    @test γ(v, u, p, t; dtgamma) ≈ dtgamma * v
    
    γ_added = γ + α
    @test γ_added(v, u, p, t; dtgamma) ≈ (dtgamma + p) * v
end