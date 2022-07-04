#
using SciMLOperators, LinearAlgebra
using Random

using SciMLOperators: InvertibleOperator, ⊗
using FFTW

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
    op1 = FunctionOperator(
                           f1;

                           isinplace=false,
                           T=Float64,
                           size=(N,N),

                           input_prototype=u,
                           output_prototype=A*u,

                           op_inverse=f1i,

                           opnorm=true,
                           issymmetric=true,
                           ishermitian=true,
                           isposdef=true,
                          )

    # in place
    op2 = FunctionOperator(
                           f2;

                           isinplace=true,
                           T=Float64,
                           size=(N,N),

                           input_prototype=u,
                           output_prototype=A*u,

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

    op2 = cache_operator(op2, u)

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

    op = FunctionOperator(
                          f;

                          isinplace=true,
                          T=Float64,
                          size=(N,N),

                          input_prototype=u,
                          output_prototype=u,

                          p=p*0.0,
                          t=0.0,
                         )

    @test op(u, p, t) ≈ @. u * p * t
end

@testset "FunctionOperator FFTW test" begin
    n = 256
    L = 2π

    dx = L / n
    x  = range(start=-L/2, stop=L/2-dx, length=n) |> Array

    k  = rfftfreq(n, 2π*n/L) |> Array
    m  = length(k)
    tr = plan_rfft(x)

    ftr = FunctionOperator(
                           (du,u,p,t) -> mul!(du, tr, u);
                           isinplace=true,
                           T=ComplexF64,
                           size=(m,n),

                           input_prototype=x,
                           output_prototype=im*k,

                           op_adjoint = (du,u,p,t) -> ldiv!(du, tr, u),
                           op_inverse = (du,u,p,t) -> ldiv!(du, tr, u),
                           op_adjoint_inverse = (du,u,p,t) -> ldiv!(du, tr, u),
                          )

    # derivative test
    ik = im * DiagonalOperator(k)
    Dx = ftr \ ik * ftr
    Dx = cache_operator(Dx, x)

    u  = @. sin(5x)cos(7x);
    du = @. 5cos(5x)cos(7x) - 7sin(5x)sin(7x);

    @test ≈(Dx * u, du; atol=1e-8)
    v = copy(u); @test ≈(mul!(v, Dx, u), du; atol=1e-8)

    itr = inv(ftr)
    ftt = ftr'
    itt = itr'

    @test itr isa FunctionOperator
    @test ftt isa FunctionOperator
    @test itt isa FunctionOperator

    @test size(ftr) == (m, n)
    @test size(itr) == (n, m)
    @test size(ftt) == (n, m)
    @test size(itt) == (m, n)

    @test ftt.op == ftr.op_adjoint
    @test ftt.op_adjoint == ftr.op
    @test ftt.op_inverse == ftr.op_adjoint_inverse
    @test ftt.op_adjoint_inverse == ftr.op_inverse

    @test itr.op == ftr.op_inverse
    @test itr.op_adjoint == ftr.op_adjoint_inverse
    @test itr.op_inverse == ftr.op
    @test itr.op_adjoint_inverse == ftr.op_adjoint

    @test itt.op == ftr.op_adjoint_inverse
    @test itt.op_adjoint == ftr.op_inverse
    @test itt.op_inverse == ftr.op_adjoint
    @test itt.op_adjoint_inverse == ftr.op
end
#
