#
using SciMLOperators, LinearAlgebra
using Random

using SciMLOperators: InvertibleOperator, ⊗

Random.seed!(0)
N = 8
K = 12

@testset "FunctionOperator" begin
    u = rand(N,K)
    p = nothing
    t = 0.0
    α = rand()
    β = rand()

    A = rand(N,N) |> Symmetric
    F = lu(A)

    f1(u, p, t)  = A * u
    f1i(u, p, t) = A \ u

    f2(du, u, p, t)  = mul!(du, A, u)
    f2i(du, u, p, t) = ldiv!(du, F, u)

    # out of place
    op1 = FunctionOperator(f1, u, A*u;

                           op_inverse=f1i,

                           ifcache = false,

                           islinear=true,
                           opnorm=true,
                           issymmetric=true,
                           ishermitian=true,
                           isposdef=true,
                          )

    # in place
    op2 = FunctionOperator(f2, u, A*u;

                           op_inverse=f2i,

                           ifcache = false,

                           islinear=true,
                           opnorm=true,
                           issymmetric=true,
                           ishermitian=true,
                           isposdef=true,
                          )
    @test issquare(op1)
    @test issquare(op2)

    @test islinear(op1)
    @test islinear(op2)

    @test op1' === op1

    @test size(op1) == (N,N)
    @test has_adjoint(op1)
    @test has_mul(op1)
    @test !has_mul!(op1)
    @test has_ldiv(op1)
    @test !has_ldiv!(op1)

    @test size(op2) == (N,N)
    @test has_adjoint(op2)
    @test has_mul(op2)
    @test has_mul!(op2)
    @test has_ldiv(op2)
    @test has_ldiv!(op2)

    @test !iscached(op1)
    @test !iscached(op2)

    op1 = cache_operator(op1, u, A * u)
    op2 = cache_operator(op2, u, A * u)

    @test iscached(op1)
    @test iscached(op2)

    v = rand(N,K); @test A * u ≈ op1 * u ≈ mul!(v, op2, u)
    v = rand(N,K); @test A * u ≈ op1(u,p,t) ≈ op2(v,u,p,t)
    v = rand(N,K); w=copy(v); @test α*(A*u)+ β*w ≈ mul!(v, op2, u, α, β)

    v = rand(N,K); @test A \ u ≈ op1 \ u ≈ ldiv!(v, op2, u)
    v = copy(u);   @test A \ v ≈ ldiv!(op2, u)
end

@testset "FunctionOperator update test" begin
    u = rand(N,K)
    p = rand(N)
    t = rand()

    f(u, p, t) = Diagonal(p * t) * u
    f(du,u,p,t) = mul!(du, Diagonal(p*t), u)

    L = FunctionOperator(f, u, u; p=zero(p), t=zero(t))

    ans = @. u * p * t
    @test L(u,p,t) ≈ ans
    v=copy(u); @test L(v,u,p,t) ≈ ans

    # test that output isn't accidentally mutated by passing an internal cache.

    A = Diagonal(p * t)
    u1 = rand(N, K)
    u2 = rand(N, K)

    v1 = L * u1; @test v1 ≈ A * u1
    v2 = L * u2; @test v2 ≈ A * u2; @test v1 ≈ A * u1
    @test v1 + v2 ≈ A * (u1 + u2)

    v1 .= 0.0
    v2 .= 0.0

    mul!(v1, L, u1); @test v1 ≈ A * u1
    mul!(v2, L, u2); @test v2 ≈ A * u2; @test v1 ≈ A * u1
    @test v1 + v2 ≈ A * (u1 + u2)

    v1 = rand(N, K); w1 = copy(v1)
    v2 = rand(N, K); w2 = copy(v2)
    a1, a2, b1, b2 = rand(4)

    mul!(v1, L, u1, a1, b1); @test v1 ≈ a1*A*u1 + b1*w1
    mul!(v2, L, u2, a2, b2); @test v2 ≈ a2*A*u2 + b2*w2; @test v1 ≈ a1*A*u1 + b1*w1
    @test v1 + v2 ≈ (a1*A*u1 + b1*w1) + (a2*A*u2 + b2*w2)
end
#
