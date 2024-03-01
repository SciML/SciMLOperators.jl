using SciMLOperators, LinearAlgebra
using Random

using SciMLOperators: IdentityOperator,
                      NullOperator,
                      ScaledOperator,
                      AddedOperator,
                      ComposedOperator,
                      AdjointOperator,
                      TransposedOperator,
                      InvertedOperator, AbstractAdjointVecOrMat,
                      AbstractTransposedVecOrMat, getops,
                      cache_operator

Random.seed!(0)
N = 8
K = 12

@testset "IdentityOperator" begin
    A = rand(N, N) |> MatrixOperator
    u = rand(N, K)
    α = rand()
    β = rand()
    Id = IdentityOperator(N)

    @test issquare(Id)
    @test islinear(Id)
    @test convert(AbstractMatrix, Id) == Matrix(I, N, N)

    _Id = one(A)
    @test _Id isa IdentityOperator
    @test size(_Id) == (N, N)

    @test iscached(Id)
    @test size(Id) == (N, N)
    @test Id' isa IdentityOperator
    @test isconstant(Id)
    @test_throws MethodError resize!(Id, N)

    for op in (*, \)
        @test op(Id, u) ≈ u
    end

    v = rand(N, K)
    @test mul!(v, Id, u) ≈ u
    v = rand(N, K)
    w = copy(v)
    @test mul!(v, Id, u, α, β) ≈ α * (I * u) + β * w

    v = rand(N, K)
    @test ldiv!(v, Id, u) ≈ u
    v = copy(u)
    @test ldiv!(Id, u) ≈ v

    for op in (*, ∘)
        @test op(Id, A) isa MatrixOperator
        @test op(A, Id) isa MatrixOperator
    end
end

@testset "NullOperator" begin
    A = rand(N, N) |> MatrixOperator
    u = rand(N, K)
    α = rand()
    β = rand()
    Z = NullOperator(N)

    @test issquare(Z)
    @test islinear(Z)
    @test isconstant(Z)
    @test_throws MethodError resize!(Z, N)
    @test convert(AbstractMatrix, Z) == zeros(size(Z))

    _Z = zero(A)
    @test _Z isa NullOperator
    @test size(_Z) == (N, N)

    @test iscached(Z)
    @test size(Z) == (N, N)
    @test Z' isa NullOperator

    @test Z * u ≈ zero(u)

    v = rand(N, K)
    @test mul!(v, Z, u) ≈ zero(u)
    v = rand(N, K)
    w = copy(v)
    @test mul!(v, Z, u, α, β) ≈ α * (0 * u) + β * w

    for op in (*, ∘)
        @test op(Z, A) isa NullOperator
        @test op(A, Z) isa NullOperator
    end
    for op in (+, -)
        @test op(Z, A) isa MatrixOperator
        @test op(A, Z) isa MatrixOperator
    end
end

@testset "ScaledOperator" begin
    A = rand(N, N)
    D = Diagonal(rand(N))
    u = rand(N, K)
    α = rand()
    β = rand()
    a = rand()
    b = rand()

    op = ScaledOperator(α, MatrixOperator(A))

    @test op isa ScaledOperator
    @test isconstant(op)
    @test iscached(op)
    @test issquare(op)
    @test islinear(op)

    @test α * A * u ≈ op * u
    @test (β * op) * u ≈ β * α * A * u

    opF = factorize(op)

    @test opF isa ScaledOperator
    @test isconstant(opF)
    @test iscached(opF)

    @test α * A ≈ convert(AbstractMatrix, op) ≈ convert(AbstractMatrix, opF)

    v = rand(N, K)
    @test mul!(v, op, u) ≈ α * A * u
    v = rand(N, K)
    w = copy(v)
    @test mul!(v, op, u, a, b) ≈ a * (α * A * u) + b * w

    op = ScaledOperator(α, MatrixOperator(D))
    v = rand(N, K)
    @test ldiv!(v, op, u) ≈ (α * D) \ u
    v = copy(u)
    @test ldiv!(op, u) ≈ (α * D) \ v
end

@testset "AddedOperator" begin
    A = rand(N, N) |> MatrixOperator
    B = rand(N, N) |> MatrixOperator
    C = rand(N, N) |> MatrixOperator
    α = rand()
    β = rand()
    u = rand(N, K)

    for op in (+, -)
        op1 = op(A, B)
        op2 = op(α * A, B)
        op3 = op(A, β * B)
        op4 = op(α * A, β * B)

        @test op1 isa AddedOperator
        @test op2 isa AddedOperator
        @test op3 isa AddedOperator
        @test op4 isa AddedOperator

        @test isconstant(op1)
        @test isconstant(op2)
        @test isconstant(op3)
        @test isconstant(op4)

        @test op1 * u ≈ op(A * u, B * u)
        @test op2 * u ≈ op(α * A * u, B * u)
        @test op3 * u ≈ op(A * u, β * B * u)
        @test op4 * u ≈ op(α * A * u, β * B * u)
    end

    op = AddedOperator(A, B)
    @test iscached(op)

    v = rand(N, K)
    @test mul!(v, op, u) ≈ (A + B) * u
    v = rand(N, K)
    w = copy(v)
    @test mul!(v, op, u, α, β) ≈ α * (A + B) * u + β * w

    # ensure AddedOperator doesn't nest
    A = MatrixOperator(rand(N, N))
    L = A + (A + A) + A
    @test L isa AddedOperator
    for op in L.ops
        @test !isa(op, AddedOperator)
    end
end

@testset "ComposedOperator" begin
    A = rand(N, N)
    B = rand(N, N)
    C = rand(N, N)
    u = rand(N, K)
    α = rand()
    β = rand()

    ABCmulu = (A * B * C) * u
    ABCdivu = (A * B * C) \ u

    op = ∘(MatrixOperator.((A, B, C))...)

    @test op isa ComposedOperator
    @test isconstant(op)

    @test *(op.ops...) isa ComposedOperator
    @test issquare(op)
    @test islinear(op)

    opF = factorize(op)

    @test opF isa ComposedOperator
    @test isconstant(opF)
    @test issquare(opF)
    @test islinear(opF)

    @test ABCmulu ≈ op * u
    @test ABCdivu ≈ op \ u ≈ opF \ u

    @test !iscached(op)
    op = cache_operator(op, u)
    @test iscached(op)

    v = rand(N, K)
    @test mul!(v, op, u) ≈ ABCmulu
    v = rand(N, K)
    w = copy(v)
    @test mul!(v, op, u, α, β) ≈ α * ABCmulu + β * w

    A = rand(N) |> Diagonal
    B = rand(N) |> Diagonal
    C = rand(N) |> Diagonal

    op = ∘(MatrixOperator.((A, B, C))...)
    @test !iscached(op)
    op = cache_operator(op, u)
    @test iscached(op)
    v = rand(N, K)
    @test ldiv!(v, op, u) ≈ (A * B * C) \ u
    v = copy(u)
    @test ldiv!(op, u) ≈ (A * B * C) \ v

    # ensure composedoperators doesn't nest
    A = MatrixOperator(rand(N, N))
    L = A * (A * A) * A
    @test L isa ComposedOperator
    for op in L.ops
        @test !isa(op, ComposedOperator)
    end

    # Test caching of composed operator when inner ops do not support Base.:*
    # ComposedOperator caching was modified in PR # 174
    inner_op = qr(MatrixOperator(rand(N, N)))
    op = inner_op * factorize(MatrixOperator(rand(N, N)))
    @test !iscached(op)
    @test_nowarn op = cache_operator(op, rand(N))
    @test iscached(op)
    u = rand(N)
    @test ldiv!(rand(N), op, u) ≈ op \ u
end

@testset "Adjoint, Transpose" begin
    for (op,
    LType,
    VType) in ((adjoint, AdjointOperator, AbstractAdjointVecOrMat),
        (transpose, TransposedOperator, AbstractTransposedVecOrMat))
        A = rand(N, N)
        D = Bidiagonal(rand(N, N), :L)
        u = rand(N, K)
        α = rand()
        β = rand()
        a = rand()
        b = rand()

        At = op(A)
        Dt = op(D)

        @test issquare(At)
        @test issquare(Dt)

        @test islinear(At)
        @test islinear(Dt)

        AA = MatrixOperator(A)
        DD = MatrixOperator(D)

        AAt = LType(AA)
        DDt = LType(DD)

        @test isconstant(AAt)
        @test isconstant(DDt)

        @test AAt.L === AA
        @test op(u) isa VType

        @test op(u) * AAt ≈ op(A * u)
        @test op(u) / AAt ≈ op(A \ u)

        v = rand(N, K)
        @test mul!(op(v), op(u), AAt) ≈ op(A * u)
        v = rand(N, K)
        w = copy(v)
        @test mul!(op(v), op(u), AAt, α, β) ≈ α * op(A * u) + β * op(w)

        v = rand(N, K)
        @test ldiv!(op(v), op(u), DDt) ≈ op(D \ u)
        v = copy(u)
        @test ldiv!(op(u), DDt) ≈ op(D \ v)
    end
end

@testset "InvertedOperator" begin
    s = rand(N)
    D = Diagonal(s) |> MatrixOperator
    Di = InvertedOperator(D)
    u = rand(N)
    α = rand()
    β = rand()

    @test !iscached(Di)
    Di = cache_operator(Di, u)
    @test isconstant(Di)
    @test iscached(Di)

    @test issquare(Di)
    @test islinear(Di)

    @test Di * u ≈ u ./ s
    v = rand(N)
    @test mul!(v, Di, u) ≈ u ./ s
    v = rand(N)
    w = copy(v)
    @test mul!(v, Di, u, α, β) ≈ α * (u ./ s) + β * w

    @test Di \ u ≈ u .* s
    v = rand(N)
    @test ldiv!(v, Di, u) ≈ u .* s
    v = copy(u)
    @test ldiv!(Di, u) ≈ v .* s
end
#
