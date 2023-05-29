#
using SciMLOperators
using SciMLOperators: AbstractSciMLScalarOperator,
                      ComposedScalarOperator,
                      AddedScalarOperator,
                      InvertedScalarOperator,
                      IdentityOperator,
                      AddedOperator,
                      ScaledOperator

using LinearAlgebra, Random

Random.seed!(0)
N = 8
K = 12

@testset "ScalarOperator" begin
    a = rand()
    b = rand()
    x = rand()
    α = ScalarOperator(x)
    u = rand(N,K)

    @test α isa ScalarOperator
    @test iscached(α)
    @test issquare(α)
    @test islinear(α)

    @test convert(Float32, α) isa Float32
    @test convert(ScalarOperator, a) isa ScalarOperator

    @test size(α) == ()
    @test isconstant(α)

    v=copy(u); @test lmul!(α, u) ≈ v * x
    v=copy(u); @test rmul!(u, α) ≈ x * v

    v=rand(N,K); @test mul!(v, α, u) ≈ u * x
    v=rand(N,K); w=copy(v); @test mul!(v, α, u, a, b) ≈ a*(x*u) + b*w

    v=rand(N,K); @test ldiv!(v, α, u) ≈ u / x
    w=copy(u);   @test ldiv!(α, u) ≈ w / x

    X=rand(N,K); Y=rand(N,K); Z=copy(Y); a=rand(); aa=ScalarOperator(a);
    @test axpy!(aa,X,Y) ≈ a*X+Z

    # Test that ScalarOperator's remain AbstractSciMLScalarOperator's under common ops
    β = α + α
    @test β isa AddedScalarOperator
    @test β * u ≈ x * u + x * u
    @inferred convert(Float32, β)
    @test convert(Number, β) ≈ x + x

    β = α * α
    @test β isa ComposedScalarOperator
    @test β * u ≈ x * x * u
    @inferred convert(Float32, β)
    @test convert(Number, β) ≈ x * x

    β = inv(α)
    @test β isa InvertedScalarOperator
    @test β * u ≈ 1 / x * u
    @inferred convert(Float32, β)
    @test convert(Number, β) ≈ 1 / x

    β = α * inv(α)
    @test β isa ComposedScalarOperator
    @test β * u ≈ u
    @inferred convert(Float32, β)
    @test convert(Number, β) ≈ true

    β = α / α
    @test β isa ComposedScalarOperator
    @test β * u ≈ u
    @inferred convert(Float32, β)
    @test convert(Number, β) ≈ true

    # Test combination with other operators
    for op in (MatrixOperator(rand(N, N)), SciMLOperators.IdentityOperator(N))
        @test α + op isa SciMLOperators.AddedOperator
        @test (α + op) * u ≈ x * u + op * u
        @test α * op isa SciMLOperators.ScaledOperator
        @test (α * op) * u ≈ x * (op * u)
        @test all(map(T -> (T isa SciMLOperators.ScaledOperator), (α / op, op / α, op \ α, α \ op)))
        @test (α / op) * u ≈ (op \ α) * u ≈ α * (op \ u)
        @test (op / α) * u ≈ (α \ op) * u ≈ 1/α * op * u
    end

    # ensure composedscalaroperators doesn't nest
    α = ScalarOperator(rand())
    L = α * (α * α) * α
    @test L isa ComposedScalarOperator
    for op in L.ops
        @test !isa(op, ComposedScalarOperator)
    end

end

@testset "ScalarOperator scalar argument test" begin
    a = rand()
    u = rand()
    v = rand()
    p = nothing
    t = 0.0

    α = ScalarOperator(a)
    @test α(u, p, t) ≈ u * a
    @test_throws ArgumentError α(v, u, p, t)
    @test_throws ArgumentError α(v, u, p, t, 1, 2)
end

@testset "ScalarOperator update test" begin
    u = ones(N,K)
    v = zeros(N,K)
    p = 2.0
    t = 4.0
    a = rand()
    b = rand()

    α = ScalarOperator(0.0; update_func=(a,u,p,t) -> p)
    β = ScalarOperator(0.0; update_func=(a,u,p,t) -> t)

    @test !isconstant(α)
    @test !isconstant(β)

    @test convert(Float32, α) isa Float32
    @test convert(Float32, β) isa Float32

    @test convert(Number, α) ≈ 0.0
    @test convert(Number, β) ≈ 0.0

    update_coefficients!(α, u, p, t)
    update_coefficients!(β, u, p, t)

    @test convert(Number, α) ≈ p
    @test convert(Number, β) ≈ t

    @test α(u, p, t) ≈ p * u
    v=rand(N,K); @test α(v, u, p, t) ≈ p * u
    v=rand(N,K); w=copy(v); @test α(v, u, p, t, a, b) ≈ a*p*u + b*w

    @test β(u, p, t) ≈ t * u
    v=rand(N,K); @test β(v, u, p, t) ≈ t * u
    v=rand(N,K); w=copy(v); @test β(v, u, p, t, a, b) ≈ a*t*u + b*w

    num = α + 2 / β * 3 - 4
    val = p + 2 / t * 3 - 4

    @test convert(Number, num) ≈ val

    # Test scalar operator which expects keyword argument to update,
    # modeled in the style of a DiffEq W-operator.
    γ = ScalarOperator(0.0; update_func = (args...; dtgamma) -> dtgamma,
                       accepted_kwargs = (:dtgamma,))

    dtgamma = rand()
    @test γ(u,p,t; dtgamma) ≈ dtgamma * u
    @test γ(v,u,p,t; dtgamma) ≈ dtgamma * u
 
    γ_added = γ + α
    @test γ_added(u,p,t; dtgamma) ≈ (dtgamma + p) * u
    @test γ_added(v,u,p,t; dtgamma) ≈ (dtgamma + p) * u
end
#
