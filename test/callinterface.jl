# test/callinterface.jl
using SciMLOperators, LinearAlgebra, Random, Test
using SciMLOperators:
    AbstractSciMLOperator, has_mul, has_mul!, cache_operator,
    MatrixOperator, IdentityOperator, NullOperator,
    DiagonalOperator, ScalarOperator,
    AddedOperator, ComposedOperator, ScaledOperator,
    InvertedOperator, InvertibleOperator,
    FunctionOperator, AffineOperator              # ← TensorProduct left out

Random.seed!(42)
N = 6
u = rand(N)
v = rand(N)
α, β = rand(), rand()
p, t = nothing, 0.7

function exercise(L::AbstractSciMLOperator)
    # ---- separate treatment for ScalarOperator ----------------------------
    if L isa ScalarOperator
        op = L(u, p, t)            # updated scalar operator
        @test op * u ≈ L * u       # behaviour unchanged
        return
    end

    # ---- generic operators ------------------------------------------------
    op = L(u, p, t)
    @test op isa AbstractSciMLOperator
    if has_mul(op)
        @test op * u ≈ L * u
        @test op * v ≈ L * v
    end

    if has_mul!(op) && !(L isa NullOperator)
        op = iscached(op) ? op : cache_operator(op, v)
        w      = zeros(eltype(v), length(v))
        w_old  = copy(w)
        mul!(w, op, v, α, β)
        @test w ≈ α * (L * v) + β * w_old
    end
end

# concrete test operators ----------------------------------------------------
A, B = rand(N, N), rand(N, N)
d     = rand(N)
b     = rand(N)
u0    = rand(N)

ops = (
    MatrixOperator(A),
    IdentityOperator(N),
    NullOperator(N),
    DiagonalOperator(d),
    ScalarOperator(2.5),
    AddedOperator(MatrixOperator(A), DiagonalOperator(d)),
    ComposedOperator(MatrixOperator(A), DiagonalOperator(d)),
    ScaledOperator(3.0, MatrixOperator(A)),
    InvertedOperator(MatrixOperator(A)),
    InvertibleOperator(MatrixOperator(A), MatrixOperator(inv(A))),
    FunctionOperator((u,p,t)->A*u, u0),                     # out‑of‑place
    FunctionOperator((w,x,u,p,t)->mul!(w,A,x), u0),         # in‑place
    AffineOperator(MatrixOperator(A), MatrixOperator(B), b),
)

@testset "Call‑interface sanity" begin
    for L in ops
        exercise(L)
    end
end
