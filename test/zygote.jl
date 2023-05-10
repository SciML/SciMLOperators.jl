#
using SciMLOperators, Zygote, LinearAlgebra
using Random

using SciMLOperators
using SciMLOperators: AbstractSciMLOperator,
                      IdentityOperator, NullOperator,
                      AdjointOperator, TransposedOperator,
                      InvertedOperator, InvertibleOperator,
                      BatchedDiagonalOperator, AddedOperator, ComposedOperator,
                      AddedScalarOperator, ComposedScalarOperator, ScaledOperator,
                      has_mul, has_ldiv

Random.seed!(0)
n = 3
N = n*n
K = 12

t = rand()
u0 = rand(N, K)
ps = rand(N)

s = rand()
v = rand(N, K)
M = rand(N, N)

sca_update_func = (a, u, p, t) -> sum(p) * s
vec_update_func = (b, u, p, t) -> Diagonal(p) * v
mat_update_func = (A, u, p, t) -> Diagonal(p) * M
inv_update_func = (A, u, p, t) -> inv(M) * inv(Diagonal(p))
tsr_update_func = (A, u, p, t) -> reshape(p, n, n) |> copy

α = ScalarOperator(zero(Float32), update_func = sca_update_func)
L_dia = DiagonalOperator(zeros(N, K); update_func = vec_update_func)
L_mat = MatrixOperator(zeros(N,N); update_func = mat_update_func)
L_aff = AffineOperator(L_mat, L_mat, zeros(N, K); update_func = vec_update_func)
L_sca = α * L_mat
# TODO - fix InvertibleOperator constructor after merging
# https://github.com/SciML/SciMLOperators.jl/pull/179
L_inv = InvertibleOperator(MatrixOperator(M))
L_fun = FunctionOperator((u,p,t) -> Diagonal(p) * u, u0, u0;
                         op_inverse=(u,p,t) -> Diagonal(p) \ u)

Ti = MatrixOperator(zeros(n, n); update_func = tsr_update_func)
To = deepcopy(Ti)
L_tsr = TensorProductOperator(To, Ti)

for (op_type, A) in
    (
     (IdentityOperator, IdentityOperator(N)),
     (NullOperator, NullOperator(N)),
     (MatrixOperator, L_mat),
     (AffineOperator, L_aff),
     (ScaledOperator, L_sca),
     (InvertedOperator, InvertedOperator(L_mat)),
     (InvertibleOperator, L_inv),
     (BatchedDiagonalOperator, L_dia),
     (AddedOperator, L_mat + L_dia),
     (ComposedOperator, L_mat * L_dia),
     (TensorProductOperator, L_tsr),
     (FunctionOperator, L_fun),

     ## ignore wrappers
     # (AdjointOperator, AdjointOperator(rand(N,N) |> MatrixOperator) |> adjoint),
     # (TransposedOperator, TransposedOperator(rand(N,N) |> MatrixOperator) |> transpose),

     (ScalarOperator, α),
     (AddedScalarOperator, α + α),
     (ComposedScalarOperator, α * α),
    )

    @assert A isa op_type

    loss_mul = function(p)

        v = Diagonal(p) * u0
        w = A(v, p, t)
        l = sum(w)
    end

    loss_div = function(p)

        v = Diagonal(p) * u0

        A = update_coefficients(A, v, p, t)
        w = A \ v

        l = sum(w)
    end

    @testset "$op_type" begin
        l_mul = loss_mul(ps)
        g_mul = Zygote.gradient(loss_mul, ps)[1]

        if A isa NullOperator
            @test isa(g_mul, Nothing)
        else
            @test !isa(g_mul, Nothing)
        end

        if has_ldiv(A)
            l_div = loss_div(ps)
            g_div = Zygote.gradient(loss_div, ps)[1]

            @test !isa(g_div, Nothing)
        end
    end
end
