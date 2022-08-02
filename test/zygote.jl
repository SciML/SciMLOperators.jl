#
using SciMLOperators, Zygote, LinearAlgebra
using Random

using SciMLOperators: IdentityOperator, NullOperator, AdjointOperator, TransposedOperator,
                      InvertedOperator, InvertibleOperator

Random.seed!(0)
n = 3
N = n*n
K = 12

u0 = rand(N, K)
ps = rand(N)

M = rand(N,N)

for (op_type, A) in
    (
     ("IdentityOperator", IdentityOperator{N}()),
     ("NullOperator", NullOperator{N}()),
     ("MatrixOperator", MatrixOperator(rand(N,N))),
     ("AffineOperator", AffineOperator(rand(N,N), rand(N,N), rand(N,K))),
     ("ScaledOperator", rand() * MatrixOperator(rand(N,N))),
     ("InvertedOperator", InvertedOperator(rand(N,N) |> MatrixOperator)),
     ("InvertibleOperator", InvertibleOperator(rand(N,N) |> MatrixOperator)),
     ("BatchDiagonalOperator", DiagonalOperator(rand(N,K))),
     ("AddedOperator", rand(N,N) + rand(N,N)),
     #("ComposedOperator", rand(N,N) * rand(N,N)),
     ("TensorProdutOperator", TensorProductOperator(rand(n,n), rand(n,n))),
     ("AdjointOperator", AdjointOperator(rand(N,N) |> MatrixOperator) |> adjoint),
     ("TransposedOperator", TransposedOperator(rand(N,N) |> MatrixOperator) |> transpose),
     ("ScalarOperator", ScalarOperator(rand())),
     #("AddedScalarOperator", ScalarOperator(rand()) + ScalarOperator(rand())),
     ("ComposedScalarOperator", ScalarOperator(rand()) * ScalarOperator(rand())),
     ("FunctionOperator", FunctionOperator((u,p,t)->M*u, op_inverse=(u,p,t)->M\u,
                                           T=Float64, isinplace=false, size=(N,N),
                                           input_prototype=u0, output_prototype=u0))
    )

    loss_mul = function(p)
    
        v = Diagonal(p) * u0

        w = A * v

        l = sum(w)
    end

    loss_div = function(p)

        v = Diagonal(p) * u0

        w = A \ v

        l = sum(w)
    end

    if A isa NullOperator
        @testset "$op_type" begin
            l_mul = loss_mul(ps)
            g_mul = Zygote.gradient(loss_mul, ps)[1]

            @test isa(g_mul, Nothing)
        end
    else
        @testset "$op_type" begin
            l_mul = loss_mul(ps)
            g_mul = Zygote.gradient(loss_mul, ps)[1]

            l_div = loss_div(ps)
            g_div = Zygote.gradient(loss_div, ps)[1]

            @test !isa(g_mul, Nothing)
            @test !isa(g_div, Nothing)
        end
    end
end

