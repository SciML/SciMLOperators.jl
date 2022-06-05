using SciMLOperators, LinearAlgebra
using Random

using SciMLOperators: InvertibleOperator

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
                           isreal=true,
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
                           isreal=true,
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

@testset "Operator Algebra" begin
    # try out array arithmatic
end
#
