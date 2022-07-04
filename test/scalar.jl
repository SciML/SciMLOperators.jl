#
using SciMLOperators, LinearAlgebra
using Random

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
    @test convert(Number, α) isa Number
    @test convert(ScalarOperator, a) isa ScalarOperator

    @test size(α) == ()

    v=copy(u); @test lmul!(α, u) ≈ v * x
    v=copy(u); @test rmul!(u, α) ≈ x * v

    v=rand(N,K); @test mul!(v, α, u) ≈ u * x
    v=rand(N,K); w=copy(v); @test mul!(v, α, u, a, b) ≈ a*(x*u) + b*w

    v=rand(N,K); @test ldiv!(v, α, u) ≈ u / x
    w=copy(u);   @test ldiv!(α, u) ≈ w / x

    X=rand(N,K); Y=rand(N,K); Z=copy(Y); a=rand(); aa=ScalarOperator(a);
    @test axpy!(aa,X,Y) ≈ a*X+Z
end

@testset "ScalarOperator update test" begin
    u = rand(N,K)
    p = rand(N)
    t = 0.0

    α = ScalarOperator(zero(Float64);
                       update_func=(a,u,p,t) -> sum(p)
                      )

    ans = sum(p) * u
    @test α(u,p,t) ≈ ans
    v=copy(u); @test α(v,u,p,t) ≈ ans
end
#
