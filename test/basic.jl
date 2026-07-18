using SciMLOperators, LinearAlgebra, SparseArrays
using ArrayInterface
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
    v = rand(N, K)
    w = zeros(N, K)
    p = nothing
    t = 0
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
    @test !ArrayInterface.issingular(Id)
    @test_throws MethodError resize!(Id, N)

    for op in (*, \)
        @test op(Id, u) ≈ u
    end

    # Test with new interface - same update and action vector
    @test Id(u, u, p, t) ≈ u

    # Test with different vectors for update and action
    @test Id(v, u, p, t) ≈ v

    # Test in-place operation
    copy!(w, zeros(N, K))
    Id(w, v, u, p, t)
    @test w ≈ v

    # Test in-place with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    Id(w, v, u, p, t, α, β)
    @test w ≈ α * v + β * orig_w

    # Original tests
    v = rand(N, K)
    @test mul!(v, Id, u) ≈ u
    v = rand(N, K)
    w_orig = copy(v)
    @test mul!(v, Id, u, α, β) ≈ α * (I * u) + β * w_orig

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
    B = rand(N + 3, N + 2) |> MatrixOperator
    C = rand(N, N + 3) |> MatrixOperator
    u = rand(N, K)
    v = rand(N, K)
    w = zeros(N, K)
    p = nothing
    t = 0
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
    @test size(Z') == (N, N)

    @test Z * u ≈ zero(u)

    # Test with new interface - same update and action vector
    @test Z(u, u, p, t) ≈ zero(u)

    # Test with different vectors for update and action
    @test Z(v, u, p, t) ≈ zero(v)

    # Test in-place operation
    copy!(w, ones(N, K))
    Z(w, v, u, p, t)
    @test w ≈ zero(v)

    # Test in-place with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    Z(w, v, u, p, t, α, β)
    @test w ≈ β * orig_w

    # Original tests
    v = rand(N, K)
    @test mul!(v, Z, u) ≈ zero(u)
    v = rand(N, K)
    w_orig = copy(v)
    @test mul!(v, Z, u, α, β) ≈ α * (0 * u) + β * w_orig

    for op in (*, ∘)
        @test op(Z, A) isa NullOperator
        @test op(A, Z) isa NullOperator
    end
    for op in (+, -)
        @test op(Z, A) isa MatrixOperator
        @test op(A, Z) isa MatrixOperator
    end

    Zrect = NullOperator(N + 2, N)
    urect = rand(N, K)
    wrect = zeros(N + 2, K)

    @test !issquare(Zrect)
    @test !issymmetric(Zrect)
    @test !ishermitian(Zrect)
    @test size(Zrect) == (N + 2, N)
    @test convert(AbstractMatrix, Zrect) == zeros(Bool, size(Zrect))
    @test size(Zrect') == (N, N + 2)
    @test size(transpose(Zrect)) == (N, N + 2)

    @test Zrect * urect ≈ zero(wrect)
    @test Zrect(urect, urect, p, t) ≈ zero(wrect)

    copy!(wrect, ones(N + 2, K))
    Zrect(wrect, urect, urect, p, t)
    @test wrect ≈ zero(wrect)

    copy!(wrect, rand(N + 2, K))
    orig_wrect = copy(wrect)
    Zrect(wrect, urect, urect, p, t, α, β)
    @test wrect ≈ β * orig_wrect

    @test mul!(wrect, Zrect, urect) ≈ zero(wrect)
    copy!(wrect, rand(N + 2, K))
    orig_wrect = copy(wrect)
    @test mul!(wrect, Zrect, urect, α, β) ≈ β * orig_wrect

    @test size(Zrect * C) == (N + 2, N + 3)
    @test size(B * Zrect) == (N + 3, N)
    @test size(Zrect ∘ C) == (N + 2, N + 3)
    @test size(B ∘ Zrect) == (N + 3, N)
end

@testset "BlockDiagonalOperator" begin
    A = rand(3, 2)
    B = rand(4, 5)
    L = BlockDiagonalOperator(MatrixOperator(A), MatrixOperator(B))
    Lmat = [A zeros(3, 5); zeros(4, 2) B]
    v = rand(7, K)
    u = rand(7, K)
    w = zeros(7, K)
    p = nothing
    t = 0
    α = rand()
    β = rand()

    @test size(L) == (7, 7)
    @test islinear(L)
    @test isconstant(L)
    @test convert(AbstractMatrix, L) ≈ Lmat
    @test BlockDiagonalOperator(A, B) isa BlockDiagonalOperator

    @test L * v ≈ Lmat * v
    @test L(v, u, p, t) ≈ Lmat * v

    mul!(w, L, v)
    @test w ≈ Lmat * v

    copy!(w, rand(7, K))
    orig_w = copy(w)
    mul!(w, L, v, α, β)
    @test w ≈ α * Lmat * v + β * orig_w

    copy!(w, zeros(7, K))
    L(w, v, u, p, t)
    @test w ≈ Lmat * v

    copy!(w, rand(7, K))
    orig_w = copy(w)
    L(w, v, u, p, t, α, β)
    @test w ≈ α * Lmat * v + β * orig_w

    x = rand(7)
    @test L * x ≈ Lmat * x
    @test convert(AbstractMatrix, L') ≈ Lmat'
    @test convert(AbstractMatrix, transpose(L)) ≈ transpose(Lmat)
end

@testset "Unary +/-" begin
    A = MatrixOperator(rand(N, N))
    v = rand(N, K)

    # Test unary +
    @test +A === A

    # Test unary - on constant MatrixOperator (simplified to MatrixOperator)
    minusA = -A
    @test minusA isa ScaledOperator
    @test minusA * v ≈ -A.A * v
    @test eltype(minusA.λ) == eltype(A.A)
end

@testset "ScaledOperator" begin
    A = rand(N, N)
    D = Diagonal(rand(N))
    u = rand(N, K)       # Update vector
    v = rand(N, K)       # Action vector
    w = zeros(N, K)      # Output vector
    p = nothing
    t = 0
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

    @test α * A * v ≈ op * v
    @test (β * op) * v ≈ β * α * A * v

    # Test with new interface - same vector for update and action
    @test op(u, u, p, t) ≈ α * A * u

    # Test with different vectors for update and action
    @test op(v, u, p, t) ≈ α * A * v

    # Test in-place operation
    copy!(w, zeros(N, K))
    op(w, v, u, p, t)
    @test w ≈ α * A * v

    # Test in-place with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    op(w, v, u, p, t, a, b)
    @test w ≈ a * (α * A * v) + b * orig_w

    opF = factorize(op)

    @test opF isa ScaledOperator
    @test isconstant(opF)
    @test iscached(opF)

    @test α * A ≈ convert(AbstractMatrix, op) ≈ convert(AbstractMatrix, opF)

    w = rand(N, K)
    @test mul!(w, op, v) ≈ α * A * v
    w = rand(N, K)
    w_orig = copy(w)
    @test mul!(w, op, v, a, b) ≈ a * (α * A * v) + b * w_orig

    op = ScaledOperator(α, MatrixOperator(D))
    w = rand(N, K)
    @test ldiv!(v, op, w) ≈ (α * D) \ w
    w = copy(v)
    @test ldiv!(op, w) ≈ (α * D) \ v
end

@testset "AddedOperator" begin
    A = rand(N, N) |> MatrixOperator
    B = rand(N, N) |> MatrixOperator
    C = rand(N, N) |> MatrixOperator
    α = rand()
    β = rand()
    u = rand(N, K)       # Update vector
    v = rand(N, K)       # Action vector
    w = zeros(N, K)      # Output vector
    p = nothing
    t = 0

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

        # Test new interface - combined case
        @test op1(u, u, p, t) ≈ op(A * u, B * u)
        @test op2(u, u, p, t) ≈ op(α * A * u, B * u)
        @test op3(u, u, p, t) ≈ op(A * u, β * B * u)
        @test op4(u, u, p, t) ≈ op(α * A * u, β * B * u)

        # Test new interface - separate vectors
        @test op1(v, u, p, t) ≈ op(A * v, B * v)
        @test op2(v, u, p, t) ≈ op(α * A * v, B * v)
        @test op3(v, u, p, t) ≈ op(A * v, β * B * v)
        @test op4(v, u, p, t) ≈ op(α * A * v, β * B * v)

        # Test in-place operation
        copy!(w, zeros(N, K))
        op1(w, v, u, p, t)
        @test w ≈ op(A * v, B * v)

        # Test in-place with scaling
        copy!(w, rand(N, K))
        orig_w = copy(w)
        op1(w, v, u, p, t, α, β)
        @test w ≈ α * op(A * v, B * v) + β * orig_w
    end

    op = AddedOperator(A, B)
    @test iscached(op)

    v = rand(N, K)
    @test mul!(v, op, u) ≈ (A + B) * u
    v = rand(N, K)
    w_orig = copy(v)
    @test mul!(v, op, u, α, β) ≈ α * (A + B) * u + β * w_orig

    # Test flattening of nested AddedOperators via direct constructor
    A = MatrixOperator(rand(N, N))
    B = MatrixOperator(rand(N, N))
    C = MatrixOperator(rand(N, N))

    # Create nested structure: (A + B) is an AddedOperator
    AB = A + B
    @test AB isa AddedOperator

    # When we create AddedOperator((AB, C)), it should flatten
    L = AddedOperator((AB, C))
    @test L isa AddedOperator
    @test length(L.ops) == 3  # Should have A, B, C (not AB and C)
    @test all(op -> !isa(op, AddedOperator), L.ops)

    # Verify correctness
    test_vec = rand(N, K)
    @test L * test_vec ≈ (A + B + C) * test_vec

    ## Time-Dependent Coefficients
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        N = 100
        A = sprand(T, N, N, 2 / N)

        func1(a, u, p, t) = t
        func2(a, u, p, t) = t^2
        func3(a, u, p, t) = t^3
        func4(a, u, p, t) = t^4
        func5(a, u, p, t) = t^5

        O1 = MatrixOperator(A) + ScalarOperator(0.0, func1) * MatrixOperator(A) + ScalarOperator(0.0, func2) * MatrixOperator(A)

        O2 = MatrixOperator(A) + ScalarOperator(0.0, func3) * MatrixOperator(A) + ScalarOperator(0.0, func4) * MatrixOperator(A)

        O3 = MatrixOperator(A) + ScalarOperator(0.0, func5) * MatrixOperator(A)

        Op = -1im * (O1 - O2)

        @test length(Op.ops) == length(O1.ops) + length(O2.ops)
        @inferred Op + O3
    end
end

@testset "AddedOperator cache sharing (Composed, Tensor, Composed, Tensor, Tensor)" begin
    using SciMLOperators: cache_operator_hinted, _get_cache_shapes

    m1, m2 = 2, 4   # m1 * m2 == N

    # C1 and C2: same wrapper (ComposedOperator), different inner type params
    # C1 = A1*B1  → ops::Tuple{MatrixOperator, MatrixOperator}
    # C2 = A2*B2' → ops::Tuple{MatrixOperator, AdjointOperator{…}}
    A1 = MatrixOperator(rand(N, N)); B1 = MatrixOperator(rand(N, N))
    A2 = MatrixOperator(rand(N, N)); B2 = MatrixOperator(rand(N, N))
    C1 = A1 * B1
    C2 = A2 * B2'

    # T1, T2, T3: same wrapper (TensorProductOperator), different inner type params
    # T1 = Ao ⊗ Ai, T2 = Ao' ⊗ Ai, T3 = Ao ⊗ Ai'
    Ao = MatrixOperator(rand(m1, m1)); Ai = MatrixOperator(rand(m2, m2))
    T1 = TensorProductOperator(Ao, Ai)
    T2 = TensorProductOperator(Ao', Ai)
    T3 = TensorProductOperator(Ao, Ai')

    L = C1 + T1 + C2 + T2 + T3 + A1 + A2
    @test L isa AddedOperator
    @test length(L.ops) == 7

    for input in (rand(N, K), rand(N))
        L = cache_operator(L, input)
        expected = C1 * input + T1 * input + C2 * input + T2 * input + T3 * input +
            A1 * input + A2 * input

        # Correctness: out-of-place (*) and in-place (mul!) paths
        @test L * input ≈ expected
        w = similar(input)
        mul!(w, L, input)
        @test w ≈ expected

        # Cache sharing: same-wrapper ops with compatible sizes share physical buffers
        @test L.ops[3].cache === L.ops[1].cache   # C2 (A2*B2') reuses C1's cache
        @test L.ops[4].cache === L.ops[2].cache   # T2 (Ao'⊗Ai) reuses T1's cache
        @test L.ops[5].cache === L.ops[2].cache   # T3 (Ao⊗Ai') reuses T1's cache
    end

    # --- Mixed-eltype: Float64 + ComplexF64 ComposedOperators ---
    # promote_type(Float64, ComplexF64) = ComplexF64 = eltype(Ac) → Ac is donor, Ar reuses its cache
    Ar = MatrixOperator(rand(Float64, N, N)) * MatrixOperator(rand(Float64, N, N))
    Ac = MatrixOperator(rand(ComplexF64, N, N)) * MatrixOperator(rand(ComplexF64, N, N))
    v_real = rand(Float64, N)
    Lcs = cache_operator(Ar + Ac, v_real)
    @test Lcs.ops[1].cache === Lcs.ops[2].cache   # Ar reuses Ac's ComplexF64 cache
    w_c = similar(v_real, ComplexF64)
    mul!(w_c, Lcs, v_real)
    @test w_c ≈ Ar * v_real + Ac * v_real

    # --- No sharing: Float64 + ComplexF32 → promote_type = ComplexF64 (neither op's eltype) ---
    Af32 = MatrixOperator(rand(ComplexF32, N, N)) * MatrixOperator(rand(ComplexF32, N, N))
    Lf32s = cache_operator(Ar + Af32, rand(Float64, N))
    @test Lf32s.ops[1].cache !== Lf32s.ops[2].cache   # independent caches

    # --- Non-square ComposedOperator: _get_cache_shapes must match cache_self's allocation ---
    M1, M2, M3 = 5, 3, 4
    P = MatrixOperator(rand(M1, M2)); Q = MatrixOperator(rand(M2, M3))
    PQc = cache_operator(P * Q, rand(M3))

    @test _get_cache_shapes(PQc, rand(M3)) == ((M2,), (M3,))
    @test map(size, PQc.cache) == ((M2,), (M3,))
    v_ns = rand(M3)
    w_ns = zeros(M1)
    mul!(w_ns, PQc, v_ns)
    @test w_ns ≈ P * (Q * v_ns)
end


@testset "ComposedOperator" begin
    A = rand(N, N)
    B = rand(N, N)
    C = rand(N, N)
    u = rand(N, K)       # Update vector
    v = rand(N, K)       # Action vector
    w = zeros(N, K)      # Output vector
    p = nothing
    t = 0
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

    # Test new interface - combined case
    @test op(u, u, p, t) ≈ ABCmulu

    # Test new interface - separate vectors
    @test op(v, u, p, t) ≈ (A * B * C) * v

    @test !iscached(op)
    op = cache_operator(op, u)
    @test iscached(op)

    # Test in-place operation with new interface
    copy!(w, zeros(N, K))
    op(w, v, u, p, t)
    @test w ≈ (A * B * C) * v

    # Test in-place with scaling with new interface
    copy!(w, rand(N, K))
    orig_w = copy(w)
    op(w, v, u, p, t, α, β)
    @test w ≈ α * ((A * B * C) * v) + β * orig_w

    # Original tests
    v = rand(N, K)
    @test mul!(v, op, u) ≈ ABCmulu
    v = rand(N, K)
    w_orig = copy(v)
    @test mul!(v, op, u, α, β) ≈ α * ABCmulu + β * w_orig

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

@testset "has_concretization composites" begin
    A = MatrixOperator(rand(N, N))
    B = MatrixOperator(rand(N, N))
    F = FunctionOperator(
        (du, u, p, t) -> copyto!(du, u),
        zeros(N),
        zeros(N);
        isinplace = true,
        T = Float64,
        islinear = true
    )

    @test has_concretization(A)
    @test has_concretization(2A)
    @test has_concretization(A + B)
    @test has_concretization(A * B)
    @test has_concretization(inv(A))
    @test !has_concretization(F)
    @test !has_concretization(F * A)
    @test !has_concretization(A + F)
    @test !ishermitian(F * A)
end

@testset "Adjoint, Transpose" begin
    for (
            op,
            LType,
            VType,
        ) in (
            (adjoint, AdjointOperator, AbstractAdjointVecOrMat),
            (transpose, TransposedOperator, AbstractTransposedVecOrMat),
        )
        A = rand(N, N)
        D = Bidiagonal(rand(N, N), :L)
        u = rand(N, K)       # Update vector
        v = rand(N, K)       # Action vector
        w = zeros(N, K)      # Output vector
        p = nothing
        t = 0
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

        # Not implementing separate test for adjoint/transpose operators
        # since they typically rely on the base operator implementations

        v = rand(N, K)
        @test mul!(op(v), op(u), AAt) ≈ op(A * u)
        v = rand(N, K)
        w_orig = copy(v)
        @test mul!(op(v), op(u), AAt, α, β) ≈ α * op(A * u) + β * op(w_orig)

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
    u = rand(N)       # Update vector
    v = rand(N)       # Action vector
    w = zeros(N)      # Output vector
    p = nothing
    t = 0
    α = rand()
    β = rand()

    @test !iscached(Di)
    Di = cache_operator(Di, u)
    @test isconstant(Di)
    @test iscached(Di)

    @test issquare(Di)
    @test islinear(Di)

    @test Di * u ≈ u ./ s

    # Test new interface - same vectors
    @test Di(u, u, p, t) ≈ u ./ s

    # Test new interface - separate vectors
    @test Di(v, u, p, t) ≈ v ./ s

    # Test in-place operation
    copy!(w, zeros(N))
    Di(w, v, u, p, t)
    @test w ≈ v ./ s

    # Test in-place with scaling
    copy!(w, rand(N))
    orig_w = copy(w)
    Di(w, v, u, p, t, α, β)
    @test w ≈ α * (v ./ s) + β * orig_w

    # Original tests
    v = rand(N)
    @test mul!(v, Di, u) ≈ u ./ s
    v = rand(N)
    w_orig = copy(v)
    @test mul!(v, Di, u, α, β) ≈ α * (u ./ s) + β * w_orig

    @test Di \ u ≈ u .* s
    v = rand(N)
    @test ldiv!(v, Di, u) ≈ u .* s
    v = copy(u)
    @test ldiv!(Di, u) ≈ v .* s
end
