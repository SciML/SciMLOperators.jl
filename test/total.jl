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
    x  = -L/2:dx:L/2-dx |> Array

    k  = rfftfreq(n, 2π*n/L) |> Array
    m  = length(k)
    tr = plan_rfft(x)

    ftr = FunctionOperator((du,u,p,t) -> mul!(du, tr, u), x, im*k;
                           isinplace=true,
                           T=ComplexF64,

                           op_adjoint = (du,u,p,t) -> ldiv!(du, tr, u),
                           op_inverse = (du,u,p,t) -> ldiv!(du, tr, u),
                           op_adjoint_inverse = (du,u,p,t) -> ldiv!(du, tr, u),

                           islinear=true,
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
    # Introduce update function for B
    B = MatrixOperator(zeros(N,N); update_func=(A, u, p, t) -> (A .= p))
    C = rand(N,N)
    # Introduce update function for D dependent on kwarg "matrix"
    D = MatrixOperator(zeros(N,N); update_func=(A, u, p, t; matrix) -> (A .= p*t*matrix), 
                       accepted_kwargs=(:matrix,))

    u = rand(N2,K)
    p = rand()
    t = rand()
    matrix = rand(N, N)
    diag = rand(N2)
    α = rand()
    β = rand()

    T1 = ⊗(A, B)
    T2 = ⊗(C, D)

    # Introduce update function for D1
    D1  = DiagonalOperator(zeros(N2); update_func=(d, u, p, t) -> (d .= p))
    # Introduce update funcion for D2 dependent on kwarg "diag" 
    D2  = DiagonalOperator(zeros(N2); update_func=(d, u, p, t; diag) -> (d .= p*t*diag),
                           accepted_kwargs=(:diag,))

    TT = [T1, T2]
    DD = Diagonal([D1, D2])

    op = TT' * DD * TT
    op = cache_operator(op, u)

    # Update operator
    @test_nowarn update_coefficients!(op, u, p, t; diag, matrix)
    # Form dense operator manually 
    dense_T1 = kron(A, p * ones(N, N))
    dense_T2 = kron(C, (p*t) .* matrix)
    dense_DD = Diagonal(vcat(p * ones(N2), p*t*diag))
    dense_op = hcat(dense_T1', dense_T2') * dense_DD * vcat(dense_T1, dense_T2)
    # Test correctness of op
    @test op * u ≈ dense_op * u
    @test convert(AbstractMatrix, op) ≈ dense_op 
    # Test consistency with three-arg mul!
    v=rand(N2,K); @test mul!(v, op, u) ≈ op * u
    # Test consistency with in-place five-arg mul!
    v=rand(N2,K); w=copy(v); @test mul!(v, op, u, α, β) ≈ α*(op * u) + β * w
    # Test consistency with operator application form
    @test op(u, p, t; diag, matrix) ≈ op * u 
    v=rand(N2,K); @test op(v, u, p, t; diag, matrix) ≈ op * u 
end
#