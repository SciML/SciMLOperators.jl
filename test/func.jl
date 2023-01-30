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

                           opnorm=true,
                           issymmetric=true,
                           ishermitian=true,
                           isposdef=true,
                          )

    # in place
    op2 = FunctionOperator(f2, u, A*u;

                           op_inverse=f2i,

                           opnorm=true,
                           issymmetric=true,
                           ishermitian=true,
                           isposdef=true,
                          )

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

    f(du,u,p,t) = mul!(du, Diagonal(p*t), u)

    op = FunctionOperator(f, u, u; p=zero(p), t=zero(t))

    ans = @. u * p * t
    @test op(u,p,t) ≈ ans
    v=copy(u); @test op(v,u,p,t) ≈ ans
end
#
