#
using SciMLOperators, LinearAlgebra, ChainRules
using Test, ChainRulesTestUtils, Random
using Zygote

Random.seed!(0)
N = 8

@testset "MatrixOperator" begin
    A = MatrixOperator(rand(N,N))
    u = rand(N)
end

@testset "Function Operator" begin
    u = rand(N)

    A = rand(N,N) |> Symmetric
    F = lu(A)

    f1(u, p, t)  = A * u
    f1i(u, p, t) = A \ u

    f2(du, u, p, t)  = (du .= A * u; du)
    f2i(du, u, p, t) = (du .= A \ u; du)

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

    f = (A, u) -> sum(A * u)

    grad = Zygote.gradient(f, op1, u)
    grad = Zygote.gradient(f, op2, u) # error?
end
#
