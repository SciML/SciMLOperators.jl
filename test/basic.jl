using SciMLOperators, LinearAlgebra, SparseArrays
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

function apply_op!(H, du, u, p, t)
    H(du, u, p, t)
    return nothing
end

test_apply_noalloc(H, du, u, p, t) = @test (@allocations apply_op!(H, du, u, p, t)) == 0

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

    # Allocations Tests

    @allocations apply_op!(op, v, u, (), 1.0) # warmup
    test_apply_noalloc(op, v, u, (), 1.0)

    ## Time-Dependent Coefficients

    for T in (Float32, Float64, ComplexF32, ComplexF64)
        N = 100
        A1_sparse = MatrixOperator(sprand(T, N, N, 5 / N))
        A2_sparse = MatrixOperator(sprand(T, N, N, 5 / N))
        A3_sparse = MatrixOperator(sprand(T, N, N, 5 / N))

        A1_dense = MatrixOperator(rand(T, N, N))
        A2_dense = MatrixOperator(rand(T, N, N))
        A3_dense = MatrixOperator(rand(T, N, N))

        coeff1(a, u, p, t) = sin(p.ω * t)
        coeff2(a, u, p, t) = cos(p.ω * t)
        coeff3(a, u, p, t) = sin(p.ω * t) * cos(p.ω * t)

        c1 = ScalarOperator(rand(T), coeff1)
        c2 = ScalarOperator(rand(T), coeff2)
        c3 = ScalarOperator(rand(T), coeff3)

        H_sparse = c1 * A1_sparse + c2 * A2_sparse + c3 * A3_sparse
        H_dense = c1 * A1_dense + c2 * A2_dense + c3 * A3_dense

        u = rand(T, N)
        du = similar(u)
        p = (ω = 0.1,)
        t = 0.1

        @allocations apply_op!(H_sparse, du, u, p, t) # warmup
        @allocations apply_op!(H_dense, du, u, p, t) # warmup
        test_apply_noalloc(H_sparse, du, u, p, t)
        test_apply_noalloc(H_dense, du, u, p, t)
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

@testset "ComposedOperator nonlinear operator composition test" begin
    u = rand(N)
    p = nothing
    t = 0.0

    square(u) = u .^ 2
    square(u, p, t) = u .^ 2
    square(v, u, p, t) = v .= u .* u

    root(u) = u .^ 2
    root(u, p, t) = u .^ 2
    root(v, u, p, t) = v .= u .* u

    F = FunctionOperator(square, u; islinear = false, op_inverse = root)

    A = DiagonalOperator(zeros(N); update_func = (d, u, p, t) -> copy!(d, u)) # u .^2
    B = DiagonalOperator(zeros(N); update_func = (d, u, p, t) -> copy!(d, u))
    C = DiagonalOperator(zeros(N); update_func = (d, u, p, t) -> copy!(d, u))

    L  = A ∘ B ∘ C
    F3 = F ∘ F ∘ F

    sq = u |> square |> square |> square

    @test A(B(C(u, p, t), p, t), p, t) ≈ sq
    @test L(u, p, t) ≈ sq
    @test F3(u, p, t) ≈ sq

    L = cache_operator(L, u)
    v = rand(N); @test L(v, u, p, t) ≈ sq

    Fi = inv(F)
    F3i = inv(F3)

    rt = u |> root |> root |> root
    @test F3i(u, p, t) ≈ rt

    Ai = inv(A)
    Bi = inv(B)
    Ci = inv(C)

    Li = inv(L)
    Fi = inv(F)
    for op in (Ai, Bi, Ci, Li)
        @test op isa SciMLOperators.InvertedOperator
    end

    rt = Ai(Bi(Ci(u, p, t), p, t), p, t)
    @test Ai(u, p, t) ≈ ones(N)
    # TODO - overwrite L(u, p, t) for InvertedOperator
    @test_broken Li(u, p, t) ≈ ones(N)
    v = rand(N); @test_broken Li(v, u, p, t) ≈ ones(N)
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
