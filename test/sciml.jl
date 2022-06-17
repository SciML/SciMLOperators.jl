using SciMLOperators, LinearAlgebra
using Random

using SciMLOperators: AbstractSciMLOperator, InvertibleOperator, ⊗

Random.seed!(0)
N = 8

@testset "MatrixOperator, InvertibleOperator" begin
    u = rand(N)
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

    v=rand(N); @test mul!(v, AA, u) ≈ A * u
    v=rand(N); w=copy(v); @test mul!(v, AA, u, α, β) ≈ α*A*u + β*w
end

@testset "AffineOperator" begin
    u = rand(N)
    A = rand(N,N)
    D = Diagonal(A)
    b = rand(N)
    α = rand()
    β = rand()

    L = AffineOperator(MatrixOperator(A), b)

    @test L * u ≈ A * u + b
    v=rand(N); @test mul!(v, L, u) ≈ A * u + b
    v=rand(N); w=copy(v); @test mul!(v, L, u, α, β) ≈ α*(A*u + b) + β*w

    L = AffineOperator(MatrixOperator(D), b)
    @test L \ u ≈ D \ (u - b)
    v=rand(N); @test ldiv!(v, L, u) ≈ D \ (u-b)
    v=copy(u); @test ldiv!(L, u) ≈ D \ (v-b)
end

@testset "FunctionOperator" begin

    u = rand(N)
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

    v = rand(N); @test A * u ≈ op1 * u ≈ mul!(v, op2, u)
    v = rand(N); @test A * u ≈ op1(u,p,t) ≈ op2(v,u,p,t)
    v = rand(N); w=copy(v); @test α*(A*u)+ β*w ≈ mul!(v, op2, u, α, β)

    v = rand(N); @test A \ u ≈ op1 \ u ≈ ldiv!(v, op2, u)
    v = copy(u); @test A \ v ≈ ldiv!(op2, u)
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

    u2 = rand(n1*n2)
    u3 = rand(n1*n2*n3)

    opAB  = TensorProductOperator(A, B)
    opABC = TensorProductOperator(A, B, C)

    @test opAB  isa TensorProductOperator
    @test opABC isa TensorProductOperator

    @test convert(AbstractMatrix, opAB)  ≈ AB
    @test convert(AbstractMatrix, opABC) ≈ ABC

    @test opAB * u2 ≈ AB * u2
    # TODO - figure out a way to nest TensorProductOperators
    # allow SciMLOperators to act on AbstractArrays
#   @test opABC * u3 ≈ ABCmulu

    opAB  = cache_operator(opAB,  u2)
#   opABC = cache_operator(opABC, u3)

    N2 = n1*n2
    N3 = n1*n2*n3
    M2 = m1*m2
    M3 = m1*m2*m3
    v2=rand(M2); @test mul!(v2, opAB , u2) ≈ AB  * u2
#   v=rand(M3); @test mul!(v, opABC, u) ≈ ABC * u3

    v2=rand(M2); w2=copy(v2); @test mul!(v2, opAB , u2, α, β) ≈ α*AB *u2 + β*w2
#   v3=rand(M3); w3=copy(v3); @test mul!(v3, opABC, u3, α, β) ≈ α*ABC*u3 + β*w3

    N1 = 8
    N2 = 12
    A = Bidiagonal(rand(N1,N1), :L)
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

    u = rand(N2)
    α = rand()
    β = rand()

    T1 = ⊗(A, B)
    T2 = ⊗(C, D)

    D1  = DiagonalOperator(rand(N2))
    D2  = DiagonalOperator(rand(N2))

    TT = AbstractSciMLOperator[T1, T2]
    DD = Diagonal(AbstractSciMLOperator[D1, D2])

    op = TT' * DD * TT
    op = cache_operator(op, u)

    v=rand(N2); @test mul!(v, op, u) ≈ op * u
    v=rand(N2); w=copy(v); @test mul!(v, op, u, α, β) ≈ α*(op * u) + β * w
end
#
