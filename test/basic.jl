using SciMLOperators, LinearAlgebra
using Random

using SciMLOperators: IdentityOperator,
                      NullOperator,
                      ScaledOperator,
                      AddedOperator,
                      ComposedOperator,
                      AdjointOperator,
                      TransposedOperator,
                      InvertedOperator,

                      AbstractAdjointVecOrMat,
                      AbstractTransposedVecOrMat,

                      getops,
                      cache_operator

Random.seed!(0)
N = 8
K = 12

@testset "IdentityOperator" begin
    A  = rand(N, N) |> MatrixOperator
    u  = rand(N,K)
    α = rand()
    β = rand()
    Id = IdentityOperator{N}()

    @test IdentityOperator(u) isa IdentityOperator{N}
    @test one(A) isa IdentityOperator{N}
    @test convert(AbstractMatrix, Id) == Matrix(I, N, N)

    @test size(Id) == (N, N)
    @test Id' isa IdentityOperator{N}

    for op in (
               *, \,
              )
        @test op(Id, u) ≈ u
    end

    v=rand(N,K); @test mul!(v, Id, u) ≈ u
    v=rand(N,K); w=copy(v); @test mul!(v, Id, u, α, β) ≈ α*(I*u) + β*w

    v=rand(N,K); @test ldiv!(v, Id, u) ≈ u
    v=copy(u);   @test ldiv!(Id, u) ≈ v

    for op in (
               *, ∘,
              )
        @test op(Id, A) isa MatrixOperator
        @test op(A, Id) isa MatrixOperator
    end
end

@testset "NullOperator" begin
    A = rand(N, N) |> MatrixOperator
    u = rand(N,K)
    α = rand()
    β = rand()
    Z = NullOperator{N}()

    @test NullOperator(u) isa NullOperator{N}
    @test zero(A) isa NullOperator{N}
    @test convert(AbstractMatrix, Z) == zeros(size(Z))

    @test size(Z) == (N, N)
    @test Z' isa NullOperator{N}

    @test Z * u ≈ zero(u)

    v=rand(N,K); @test mul!(v, Z, u) ≈ zero(u)
    v=rand(N,K); w=copy(v); @test mul!(v, Z, u, α, β) ≈ α*(0*u) + β*w

    for op in (
               *, ∘,
              )
        @test op(Z, A) isa NullOperator
        @test op(A, Z) isa NullOperator
    end
    for op in (
               +, -,
              )
        @test op(Z, A) isa MatrixOperator
        @test op(A, Z) isa MatrixOperator
    end
end

@testset "ScaledOperator" begin
    A = rand(N,N)
    D = Diagonal(rand(N))
    u = rand(N,K)
    α = rand()
    β = rand()
    a = rand()
    b = rand()

    op = ScaledOperator(α, MatrixOperator(A))

    @test op * u       ≈     α * A * u
    @test (β * op) * u ≈ β * α * A * u

    v=rand(N,K); @test mul!(v, op, u) ≈ α * A * u
    v=rand(N,K); w=copy(v); @test mul!(v, op, u, a, b) ≈ a*(α*A*u) + b*w

    op = ScaledOperator(α, MatrixOperator(D))
    v=rand(N,K); @test ldiv!(v, op, u) ≈ (α * D) \ u
    v=copy(u); @test ldiv!(op, u) ≈ (α * D) \ v
end

@testset "AddedOperator" begin
    A = rand(N,N) |> MatrixOperator
    B = rand(N,N) |> MatrixOperator
    C = rand(N,N) |> MatrixOperator
    α = rand()
    β = rand()
    u = rand(N,K)

    for op in (
               +, -
              )
        op1 = op(A  , B  )
        op2 = op(α*A, B  )
        op3 = op(A  , β*B)
        op4 = op(α*A, β*B)

        @test op1 isa AddedOperator
        @test op2 isa AddedOperator
        @test op3 isa AddedOperator
        @test op4 isa AddedOperator

        @test op1 * u ≈ op(  A*u,   B*u)
        @test op2 * u ≈ op(α*A*u,   B*u)
        @test op3 * u ≈ op(  A*u, β*B*u)
        @test op4 * u ≈ op(α*A*u, β*B*u)
    end

    op = AddedOperator(A, B)
    v=rand(N,K); @test mul!(v, op, u) ≈ (A+B) * u
    v=rand(N,K); w=copy(v); @test mul!(v, op, u, α, β) ≈ α*(A+B)*u + β*w
end

@testset "ComposedOperator" begin
    A = rand(N,N)
    B = rand(N,N)
    C = rand(N,N)
    u = rand(N,K)
    α = rand()
    β = rand()

    ABCmulu = (A * B * C) * u
    ABCdivu = (A * B * C) \ u

    op = ∘(MatrixOperator.((A, B, C))...)

    @test op isa ComposedOperator
    @test *(op.ops...) isa ComposedOperator

    opF = factorize(op)

    @test opF isa ComposedOperator

    @test ABCmulu ≈ op * u
    @test ABCdivu ≈ op \ u ≈ opF \ u

    op = cache_operator(op, u)
    v=rand(N,K); @test mul!(v, op, u) ≈ ABCmulu
    v=rand(N,K); w=copy(v); @test mul!(v, op, u, α, β) ≈ α*ABCmulu + β*w

    A = rand(N) |> Diagonal
    B = rand(N) |> Diagonal
    C = rand(N) |> Diagonal

    op = ∘(MatrixOperator.((A, B, C))...)
    op = cache_operator(op, u)
    v=rand(N,K); @test ldiv!(v, op, u) ≈ (A * B * C) \ u
    v=copy(u);   @test ldiv!(op, u)    ≈ (A * B * C) \ v

    # Test caching of composed operator when inner ops do not support Base.:*
    # See issue #129
    inner_op = qr(MatrixOperator(rand(N, N)))
    # We use the QR factorization of a non-square matrix, which does
    # not support * as verified below.
    @test !has_mul(inner_op)
    @test has_ldiv(inner_op)
    @test_throws MethodError inner_op * u
    # We can now test that caching does not rely on matmul
    op = inner_op * factorize(MatrixOperator(rand(N, N)))
    @test_nowarn op = cache_operator(op, rand(N)) 
    u = rand(N)
    @test ldiv!(rand(N), op, u) ≈ op \ u
end

@testset "Adjoint, Transpose" begin

    for (op, LType, VType) in (
                               (adjoint,   AdjointOperator,    AbstractAdjointVecOrMat   ),
                               (transpose, TransposedOperator, AbstractTransposedVecOrMat),
                              )
        A = rand(N,N)
        D = Bidiagonal(rand(N,N), :L)
        u = rand(N,K)
        α = rand()
        β = rand()
        a = rand()
        b = rand()

        At = op(A)
        Dt = op(A)

        AA = MatrixOperator(A)
        DD = MatrixOperator(D)

        AAt = LType(AA)
        DDt = LType(DD)

        @test AAt.L === AA
        @test op(u) isa VType

        @test op(u) * AAt ≈ op(A * u)
        @test op(u) / AAt ≈ op(A \ u)

        v=rand(N,K); @test mul!(op(v), op(u), AAt) ≈ op(A * u)
        v=rand(N,K); w=copy(v); @test mul!(op(v), op(u), AAt, α, β) ≈ α*op(A * u) + β*op(w)

        v=rand(N,K); @test ldiv!(op(v), op(u), DDt) ≈ op(D \ u)
        v=copy(u); @test ldiv!(op(u), DDt) ≈ op(D \ v)
    end
end

@testset "InvertedOperator" begin
    s  = rand(N)
    D  = Diagonal(s) |> MatrixOperator
    Di = InvertedOperator(D)
    u  = rand(N)
    α  = rand()
    β  = rand()

    Di = cache_operator(Di, u)

    @test Di * u ≈ u ./ s
    v=rand(N); @test mul!(v, Di, u) ≈ u ./ s
    v=rand(N); w=copy(v); @test mul!(v, Di, u, α, β) ≈ α *(u ./ s) + β*w

    @test Di \ u ≈ u .* s
    v=rand(N); @test ldiv!(v, Di, u) ≈ u .* s
    v=copy(u); @test ldiv!(Di, u) ≈ v .* s
end
#
