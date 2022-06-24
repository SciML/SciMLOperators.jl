#
using SciMLOperators, LinearAlgebra, ChainRules
using Test, ChainRulesTestUtils, Random
using Zygote

Random.seed!(0)
N = 8

f = (M, u) -> sum(M * u);

@testset "Matrix Operators" begin
    M = MatrixOperator(rand(N,N))
    D = DiagonalOperator(rand(N))
    A = AffineOperator(rand(N,N), rand(N,N), rand(N))
    u = rand(N)


    @test Zygote.gradient(f, M, u)[1].A isa Matrix
    @test Zygote.gradient(f, D, u)[1].A isa Diagonal
    Zygote.gradient(f, A, u)
end

@testset "Function Operator" begin
    u = rand(N)

    A = rand(N,N) |> Symmetric
    F = lu(A)

    fwd(du, u, p, t) = mul!(du, A, u)
    bwd(du, u, p, t) = ldiv!(du, F, u)

    # in place
    F = FunctionOperator(
                         fwd;

                         isinplace=true,
                         T=Float64,
                         size=(N,N),

                         input_prototype=u,
                         output_prototype=A*u,

                         op_inverse=bwd,

                         opnorm=true,
                         issymmetric=true,
                         ishermitian=true,
                         isposdef=true,
                        )

    grad = Zygote.gradient(f, F, u)
#   test_rrule(*, F, u)
end
#
