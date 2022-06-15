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

@testset "IdentityOperator" begin
    A  = rand(N, N) |> MatrixOperator
    u  = rand(N)
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

    v=rand(N); @test mul!(v, Id, u) ≈ u
    v=rand(N); w=copy(v); @test mul!(v, Id, u, α, β) ≈ α*(I*u) + β*w

    v=rand(N); @test ldiv!(v, Id, u) ≈ u
    v=copy(u); @test ldiv!(Id, u) ≈ v

    for op in (
               *, ∘,
              )
        @test op(Id, A) isa MatrixOperator
        @test op(A, Id) isa MatrixOperator
    end
end

@testset "NullOperator" begin
    A = rand(N, N) |> MatrixOperator
    u = rand(N)
    α = rand()
    β = rand()
    Z = NullOperator{N}()

    @test NullOperator(u) isa NullOperator{N}
    @test zero(A) isa NullOperator{N}
    @test convert(AbstractMatrix, Z) == zeros(size(Z))

    @test size(Z) == (N, N)
    @test Z' isa NullOperator{N}

    @test Z * u ≈ zero(u)

    v=rand(N); @test mul!(v, Z, u) ≈ zero(u)
    v=rand(N); w=copy(v); @test mul!(v, Z, u, α, β) ≈ α*(0*u) + β*w

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

@testset "ScalarOperator" begin
    a = rand()
    b = rand()
    x = rand()
    α = ScalarOperator(x)
    u = rand(N)

    @test α isa ScalarOperator
    @test convert(Number, α) isa Number
    @test convert(ScalarOperator, a) isa ScalarOperator

    @test size(α) == ()

    for op in (
               *, /, \, +, -,
              )
        @test op(α, a) ≈ op(x, a)
        @test op(a, α) ≈ op(a, x)
    end

    v = copy(u); @test lmul!(α, u) == v * x
    v = copy(u); @test rmul!(u, α) == x * v

    v=rand(N); @test mul!(v, α, u) == u * x
    v=rand(N); w=copy(v); @test mul!(v, α, u, a, b) ≈ a*(x*u) + b*w

    v=rand(N); @test ldiv!(v, α, u) == u / x
    w=copy(u); @test ldiv!(α, u) == w / x

    v = rand(N); w = copy(v); @test axpy!(α, u, v) == u * x + w

    @test abs(ScalarOperator(-x)) == x
end

@testset "ScaledOperator" begin
    A = rand(N,N)
    D = Diagonal(rand(N))
    u = rand(N)
    α = rand()
    β = rand()
    a = rand()
    b = rand()

    op = ScaledOperator(α, MatrixOperator(A))

    @test op * u       ≈     α * A * u
    @test (β * op) * u ≈ β * α * A * u

    v=rand(N); @test mul!(v, op, u) ≈ α * A * u
    v=rand(N); w=copy(v); @test mul!(v, op, u, a, b) ≈ a*(α*A*u) + b*w

    op = ScaledOperator(α, MatrixOperator(D))
    v=rand(N); @test ldiv!(v, op, u) ≈ (α * D) \ u
    v=copy(u); @test ldiv!(op, u) ≈ (α * D) \ v
end

@testset "AddedOperator" begin
    A = rand(N,N) |> MatrixOperator
    B = rand(N,N) |> MatrixOperator
    C = rand(N,N) |> MatrixOperator
    α = rand()
    β = rand()
    u = rand(N)

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
    v=rand(N); @test mul!(v, op, u) ≈ (A+B) * u
    v=rand(N); w=copy(v); @test mul!(v, op, u, α, β) ≈ α*(A+B)*u + β*w
end

@testset "ComposedOperator" begin
    A = rand(N,N)
    B = rand(N,N)
    C = rand(N,N)
    u = rand(N)
    α = rand()
    β = rand()

    ABCmulu = (A * B * C) * u
    ABCdivu = (A * B * C) \ u

    op = ∘(MatrixOperator.((A, B, C))...)

    @test op isa ComposedOperator
    @test *(op.ops...) isa ComposedOperator

    @test op * u ≈ ABCmulu
    @test op \ u ≈ ABCdivu

    op = cache_operator(op, u)
    v=rand(N); @test mul!(v, op, u) ≈ ABCmulu
    v=rand(N); w=copy(v); @test mul!(v, op, u, α, β) ≈ α*ABCmulu + β*w

    A = rand(N) |> Diagonal
    B = rand(N) |> Diagonal
    C = rand(N) |> Diagonal

    op = ∘(MatrixOperator.((A, B, C))...)
    op = cache_operator(op, u)
    v=rand(N); @test ldiv!(v, op, u) ≈ (A * B * C) \ u
    v=copy(u); @test ldiv!(op, u)    ≈ (A * B * C) \ v
end

@testset "Adjoint, Transpose" begin

    for (op, LType, VType) in (
                               (adjoint,   AdjointOperator,    AbstractAdjointVecOrMat   ),
                               (transpose, TransposedOperator, AbstractTransposedVecOrMat),
                              )
        A = rand(N,N)
        D = Bidiagonal(rand(N,N), :L)
        u = rand(N)
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

        v=rand(N); @test mul!(op(v), op(u), AAt) ≈ op(A * u)
        v=rand(N); w=copy(v); @test mul!(op(v), op(u), AAt, α, β) ≈ α*op(A * u) + β*op(w)

        v=rand(N); @test ldiv!(op(v), op(u), DDt) ≈ op(D \ u)
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
