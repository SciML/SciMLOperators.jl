#
using SciMLOperators, LinearAlgebra
using Random

using SciMLOperators: InvertibleOperator, ⊗
using FFTW

Random.seed!(0)
N = 8
K = 12

@testset "FunctionOperator FFTW test" begin
    n = 256
    L = 2π

    dx = L / n
    x  = range(start=-L/2, stop=L/2-dx, length=n) |> Array

    k  = rfftfreq(n, 2π*n/L) |> Array
    m  = length(k)
    tr = plan_rfft(x)

    ftr = FunctionOperator((du,u,p,t) -> mul!(du, tr, u), x, im*k;
                           isinplace=true,
                           T=ComplexF64,

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

@testset "Operator Algebra" begin
    N2 = N*N
    A = rand(N,N)
    B = rand(N,N)
    C = rand(N,N)
    D = rand(N,N)

    u = rand(N2,K)
    α = rand()
    β = rand()

    T1 = ⊗(A, B)
    T2 = ⊗(C, D)

    D1  = DiagonalOperator(rand(N2))
    D2  = DiagonalOperator(rand(N2))

    TT = [T1, T2]
    DD = Diagonal([D1, D2])

    op = TT' * DD * TT
    op = cache_operator(op, u)

    v=rand(N2,K); @test mul!(v, op, u) ≈ op * u
    v=rand(N2,K); w=copy(v); @test mul!(v, op, u, α, β) ≈ α*(op * u) + β * w
end
#
