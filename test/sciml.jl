using SciMLOperators, LinearAlgebra
using Random

using SciMLOperators: InvertibleOperator, ⊗
using FFTW

Random.seed!(0)
N = 8
K = 12

@testset "MatrixOperator, InvertibleOperator" begin
    u = rand(N,K)
    p = nothing
    t = 0
    α = rand()
    β = rand()

    A  = rand(N,N)
    At = A'

    AA  = MatrixOperator(A)
    AAt = AA'

    @test AA  isa MatrixOperator
    @test AAt isa MatrixOperator

    FF  = factorize(AA)
    FFt = FF'

    @test FF  isa InvertibleOperator
    @test FFt isa InvertibleOperator

    @test eachindex(A)  === eachindex(AA)
    @test eachindex(A') === eachindex(AAt) === eachindex(MatrixOperator(At))

    @test A  ≈ convert(AbstractMatrix, AA ) ≈ convert(AbstractMatrix, FF )
    @test At ≈ convert(AbstractMatrix, AAt) ≈ convert(AbstractMatrix, FFt)

    @test A  ≈ Matrix(AA ) ≈ Matrix(FF )
    @test At ≈ Matrix(AAt) ≈ Matrix(FFt)

    @test A  * u ≈ AA(u,p,t)
    @test At * u ≈ AAt(u,p,t)

    @test A  \ u ≈ AA  \ u ≈ FF  \ u
    @test At \ u ≈ AAt \ u ≈ FFt \ u

    v=rand(N,K); @test mul!(v, AA, u) ≈ A * u
    v=rand(N,K); w=copy(v); @test mul!(v, AA, u, α, β) ≈ α*A*u + β*w
end

@testset "AffineOperator" begin
    u = rand(N,K)
    A = rand(N,N)
    B = rand(N,N)
    D = Diagonal(A)
    b = rand(N,K)
    α = rand()
    β = rand()

    L = AffineOperator(MatrixOperator(A), MatrixOperator(B), b)

    @test L * u ≈ A * u + B*b
    v=rand(N,K); @test mul!(v, L, u) ≈ A*u + B*b
    v=rand(N,K); w=copy(v); @test mul!(v, L, u, α, β) ≈ α*(A*u + B*b) + β*w

    L = AffineOperator(MatrixOperator(D), MatrixOperator(B), b)
    @test L \ u ≈ D \ (u - B * b)
    v=rand(N,K); @test ldiv!(v, L, u) ≈ D \ (u-B*b)
    v=copy(u); @test ldiv!(L, u) ≈ D \ (v-B*b)
end

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

    # nonallocating
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
    @test !has_mul(op2)
    @test has_mul!(op2)
    @test !has_ldiv(op2)
    @test has_ldiv!(op2)

    op2 = cache_operator(op2, u)

    v = rand(N,K); @test A * u ≈ op1 * u ≈ mul!(v, op2, u)
    v = rand(N,K); @test A * u ≈ op1(u,p,t) ≈ op2(v,u,p,t)
    v = rand(N,K); w=copy(v); @test α*(A*u)+ β*w ≈ mul!(v, op2, u, α, β)

    v = rand(N,K); @test A \ u ≈ op1 \ u ≈ ldiv!(v, op2, u)
    v = copy(u);   @test A \ v ≈ ldiv!(op2, u)
end

@testset "FunctionOperator FFTW test" begin
    n = 256
    L = 2π
    
    dx = L / n
    x  = range(start=-L/2, stop=L/2-dx, length=n) |> Array
    
    k  = rfftfreq(n, 2π*n/L) |> Array
    tr = plan_rfft(x)
    
    transform = FunctionOperator(
                                 (du,u,p,t) -> mul!(du, tr, u);
                                 isinplace=true,
                                 T=ComplexF64,
                                 size=(length(k),n),
    
                                 input_prototype=x,
                                 output_prototype=im*k,
    
                                 op_inverse = (du,u,p,t) -> ldiv!(du, tr, u)
                                )
    
    ik = im * DiagonalOperator(k)
    Dx = transform \ ik * transform
    
    Dx = cache_operator(Dx, x)
    
    u  = @. sin(5x)cos(7x);
    du = @. 5cos(5x)cos(7x) - 7sin(5x)sin(7x);
    
    @test ≈(Dx * u, du; atol=1e-8)
    v = copy(u); @test ≈(mul!(v, Dx, u), du; atol=1e-8)
end

@testset "TensorProductOperator" begin
    m1, n1 = 3 , 5
    m2, n2 = 7 , 11
    m3, n3 = 13, 17

    A = rand(m1, n1)
    B = rand(m2, n2)
    C = rand(m3, n3)
    α = rand()
    β = rand()

    AB  = kron(A, B)
    ABC = kron(A, B, C)

    u2 = rand(n1*n2, K)
    u3 = rand(n1*n2*n3, K)

    opAB  = TensorProductOperator(A, B)
    opABC = TensorProductOperator(A, B, C)

    @test opAB  isa TensorProductOperator
    @test opABC isa TensorProductOperator

    @test convert(AbstractMatrix, opAB)  ≈ AB
    @test convert(AbstractMatrix, opABC) ≈ ABC

    @test opAB  * u2 ≈ AB  * u2
    @test opABC * u3 ≈ ABC * u3

    opAB  = cache_operator(opAB,  u2)
    opABC = cache_operator(opABC, u3)

    N2 = n1*n2
    N3 = n1*n2*n3
    M2 = m1*m2
    M3 = m1*m2*m3
    v2=rand(M2,K); @test mul!(v2, opAB , u2) ≈ AB  * u2
    v3=rand(M3,K); @test mul!(v3, opABC, u3) ≈ ABC * u3

    v2=rand(M2,K); w2=copy(v2); @test mul!(v2, opAB , u2, α, β) ≈ α*AB *u2 + β*w2
    v3=rand(M3,K); w3=copy(v3); @test mul!(v3, opABC, u3, α, β) ≈ α*ABC*u3 + β*w3

    N1 = 8
    N2 = 12
    A = Bidiagonal(rand(N1,N1), :L) # invertible
    B = Bidiagonal(rand(N2,N2), :L)
    u = rand(N1*N2)

    AB = kron(A, B)
    op = ⊗(A, B)
    op = cache_operator(op, u)
    v=rand(N1*N2); @test ldiv!(v, op, u) ≈ AB \ u
    v=copy(u);     @test ldiv!(op, u)    ≈ AB \ v
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
