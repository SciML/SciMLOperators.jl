using SciMLOperators, LinearAlgebra
using Random

Random.seed!(0)
N = 8

@testset "MatrixOperator, FactorizedOperator" begin
    u = rand(N)
    p = nothing
    t = 0

    A  = rand(N,N)
    At = A'

    AA  = MatrixOperator(A)
    AAt = AA'

    @test AA  isa MatrixOperator
    @test AAt isa MatrixOperator

    FF  = factorize(AA)
    FFt = FF'

    @test FF  isa FactorizedOperator
    @test FFt isa FactorizedOperator

    @test eachindex(A)  === eachindex(AA)
    @test eachindex(A') === eachindex(AAt) === eachindex(MatrixOperator(At))

    @test A  ≈ convert(AbstractMatrix, AA ) ≈ convert(AbstractMatrix, FF )
    @test At ≈ convert(AbstractMatrix, AAt) ≈ convert(AbstractMatrix, FFt)

    @test A  ≈ Matrix(AA ) ≈ Matrix(FF )
    @test At ≈ Matrix(AAt) ≈ Matrix(FFt)

    @test A  * u ≈ AA(u,p,t)  ≈ FF(u,p,t)
    @test At * u ≈ AAt(u,p,t) ≈ FFt(u,p,t)

    @test A  \ u ≈ AA  \ u ≈ FF  \ u
    @test At \ u ≈ AAt \ u ≈ FFt \ u
end

@testset "FunctionOperator" begin
    A = rand(N,N) |> Symmetric
    F = lu(A)

    u = rand(N)
    p = nothing
    t = nothing

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

    v = zero(u); @test A * u ≈ op1 * u ≈ mul!(v, op2, u)
    v = zero(u); @test A * u ≈ op1(u,p,t) ≈ op2(v,u,p,t)

    v = zero(u); @test A \ u ≈ op1 \ u ≈ ldiv!(v, op2, u)
end

@testset "Operator Algebra" begin
    # try out array arithmatic
end
#
