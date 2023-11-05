#
using SciMLOperators, LinearAlgebra
using Random

using SciMLOperators: ⊗

Random.seed!(0)
N = 8
K = 12
NK = N * K

@testset "(Unbatched) FunctionOperator ND array" begin
    N1, N2, N3 = 3, 4, 5
    M1, M2, M3 = 4, 5, 6

    p = nothing
    t = 0.0
    α = rand()
    β = rand()

    for (sz_in, sz_out) in (((N1, N2, N3), (N1, N2, N3)), # equal size
        ((N1, N2, N3), (M1, M2, M3)))
        N = prod(sz_in)
        M = prod(sz_out)

        A = rand(M, N)
        u = rand(sz_in...)
        v = rand(sz_out...)

        _mul(A, u) = reshape(A * vec(u), sz_out)
        f(u, p, t) = _mul(A, u)
        f(du, u, p, t) = (mul!(vec(du), A, vec(u)); du)

        kw = (;) # FunctionOp kwargs

        if sz_in == sz_out
            F = lu(A)
            _div(A, v) = reshape(A \ vec(v), sz_in)
            fi(u, p, t) = _div(A, u)
            fi(du, u, p, t) = (ldiv!(vec(du), F, vec(u)); du)

            kw = (; op_inverse = fi)
        end

        L = FunctionOperator(f, u, v; kw...)
        L = cache_operator(L, u)

        # test with ND-arrays
        @test _mul(A, u) ≈ L(u, p, t) ≈ L * u ≈ mul!(zero(v), L, u)
        @test α * _mul(A, u) + β * v ≈ mul!(copy(v), L, u, α, β)

        if sz_in == sz_out
            @test _div(A, v) ≈ L \ v ≈ ldiv!(zero(u), L, v) ≈ ldiv!(L, copy(v))
        end

        # test with vec(Array)
        @test vec(_mul(A, u)) ≈ L(vec(u), p, t) ≈ L * vec(u) ≈ mul!(vec(zero(v)), L, vec(u))
        @test vec(α * _mul(A, u) + β * v) ≈ mul!(vec(copy(v)), L, vec(u), α, β)

        if sz_in == sz_out
            @test vec(_div(A, v)) ≈ L \ vec(v) ≈ ldiv!(vec(zero(u)), L, vec(v)) ≈
                  ldiv!(L, vec(copy(v)))
        end

        @test_throws DimensionMismatch mul!(vec(v), L, u)
        @test_throws DimensionMismatch mul!(v, L, vec(u))
    end # for
end

@testset "(Unbatched) FunctionOperator" begin
    u = rand(N, K)
    p = nothing
    t = 0.0
    α = rand()
    β = rand()

    _mul(A, u) = reshape(A * vec(u), N, K)
    _div(A, u) = reshape(A \ vec(u), N, K)

    A = rand(NK, NK) |> Symmetric
    F = lu(A)
    Ai = inv(A)

    f1(u, p, t) = _mul(A, u)
    f1i(u, p, t) = _div(A, u)

    f2(du, u, p, t) = (mul!(vec(du), A, vec(u)); du)
    f2(du, u, p, t, α, β) = (mul!(vec(du), A, vec(u), α, β); du)
    f2i(du, u, p, t) = (ldiv!(vec(du), F, vec(u)); du)
    f2i(du, u, p, t, α, β) = (mul!(vec(du), Ai, vec(u), α, β); du)

    # out of place
    op1 = FunctionOperator(f1, u; op_inverse = f1i, ifcache = false, islinear = true,
        opnorm = true,
        issymmetric = true,
        ishermitian = true,
        isposdef = true,)

    # in place
    op2 = FunctionOperator(f2, u; op_inverse = f2i, ifcache = false, islinear = true,
        opnorm = true,
        issymmetric = true,
        ishermitian = true,
        isposdef = true,)

    @test issquare(op1)
    @test issquare(op2)

    @test islinear(op1)
    @test islinear(op2)

    @test op1' === op1

    @test size(op1) == (NK, NK)
    @test has_adjoint(op1)
    @test has_mul(op1)
    @test !has_mul!(op1)
    @test has_ldiv(op1)
    @test !has_ldiv!(op1)

    @test size(op2) == (NK, NK)
    @test has_adjoint(op2)
    @test has_mul(op2)
    @test has_mul!(op2)
    @test has_ldiv(op2)
    @test has_ldiv!(op2)

    @test !iscached(op1)
    @test !iscached(op2)

    @test !op1.traits.has_mul5
    @test op2.traits.has_mul5

    # 5-arg mul! (w/o cache)
    v = rand(N, K)
    w = copy(v)
    @test α * _mul(A, u) + β * w ≈ mul!(v, op2, u, α, β)

    op1 = cache_operator(op1, u)
    op2 = cache_operator(op2, u)

    @test iscached(op1)
    @test iscached(op2)

    v = rand(N, K)
    @test _mul(A, u) ≈ op1 * u ≈ mul!(v, op2, u)
    v = rand(N, K)
    @test _mul(A, u) ≈ op1(u, p, t) ≈ op2(v, u, p, t)
    v = rand(N, K)
    w = copy(v)
    @test α * _mul(A, u) + β * w ≈ mul!(v, op2, u, α, β)

    v = rand(N, K)
    @test _div(A, u) ≈ op1 \ u ≈ ldiv!(v, op2, u)
    v = copy(u)
    @test _div(A, v) ≈ ldiv!(op2, u)
end

@testset "Batched FunctionOperator" begin
    u = rand(N, K)
    p = nothing
    t = 0.0
    α = rand()
    β = rand()

    A = rand(N, N) |> Symmetric
    F = lu(A)
    Ai = inv(A)

    f1(u, p, t) = A * u
    f1i(u, p, t) = A \ u

    f2(du, u, p, t) = mul!(du, A, u)
    f2(du, u, p, t, α, β) = mul!(du, A, u, α, β)
    f2i(du, u, p, t) = ldiv!(du, F, u)
    f2i(du, u, p, t, α, β) = mul!(du, Ai, u, α, β)

    # out of place
    op1 = FunctionOperator(f1, u, A * u; op_inverse = f1i, ifcache = false,
        batch = true,
        islinear = true,
        opnorm = true,
        issymmetric = true,
        ishermitian = true,
        isposdef = true,)

    # in place
    op2 = FunctionOperator(f2, u, A * u; op_inverse = f2i, ifcache = false,
        batch = true,
        islinear = true,
        opnorm = true,
        issymmetric = true,
        ishermitian = true,
        isposdef = true,)

    @test issquare(op1)
    @test issquare(op2)

    @test islinear(op1)
    @test islinear(op2)

    @test op1' === op1

    @test size(op1) == (N, N)
    @test has_adjoint(op1)
    @test has_mul(op1)
    @test !has_mul!(op1)
    @test has_ldiv(op1)
    @test !has_ldiv!(op1)

    @test size(op2) == (N, N)
    @test has_adjoint(op2)
    @test has_mul(op2)
    @test has_mul!(op2)
    @test has_ldiv(op2)
    @test has_ldiv!(op2)

    @test !iscached(op1)
    @test !iscached(op2)

    @test !op1.traits.has_mul5
    @test op2.traits.has_mul5

    # 5-arg mul! (w/o cache)
    v = rand(N, K)
    w = copy(v)
    @test α * *(A, u) + β * w ≈ mul!(v, op2, u, α, β)

    op1 = cache_operator(op1, u)
    op2 = cache_operator(op2, u)

    @test iscached(op1)
    @test iscached(op2)

    v = rand(N, K)
    @test *(A, u) ≈ op1 * u ≈ mul!(v, op2, u)
    v = rand(N, K)
    @test *(A, u) ≈ op1(u, p, t) ≈ op2(v, u, p, t)
    v = rand(N, K)
    w = copy(v)
    @test α * *(A, u) + β * w ≈ mul!(v, op2, u, α, β)

    v = rand(N, K)
    @test \(A, u) ≈ op1 \ u ≈ ldiv!(v, op2, u)
    v = copy(u)
    @test \(A, v) ≈ ldiv!(op2, u)
end

@testset "FunctionOperator update test" begin
    u = rand(N, K)
    p = rand(N)
    t = rand()
    scale = rand()

    # Accept a kwarg "scale" in operator action
    f(du, u, p, t; scale = 1.0) = mul!(du, Diagonal(p * t * scale), u)
    f(u, p, t; scale = 1.0) = Diagonal(p * t * scale) * u

    L = FunctionOperator(f, u, u; p = zero(p), t = zero(t), batch = true,
        accepted_kwargs = (:scale,))

    @test size(L) == (N, N)

    ans = @. u * p * t * scale
    @test L(u, p, t; scale) ≈ ans
    v = copy(u)
    @test L(v, u, p, t; scale) ≈ ans

    # test that output isn't accidentally mutated by passing an internal cache.

    A = Diagonal(p * t * scale)
    u1 = rand(N, K)
    u2 = rand(N, K)

    v1 = L * u1
    @test v1 ≈ A * u1
    v2 = L * u2
    @test v2 ≈ A * u2
    @test v1 ≈ A * u1
    @test v1 + v2 ≈ A * (u1 + u2)

    v1 .= 0.0
    v2 .= 0.0

    mul!(v1, L, u1)
    @test v1 ≈ A * u1
    mul!(v2, L, u2)
    @test v2 ≈ A * u2
    @test v1 ≈ A * u1
    @test v1 + v2 ≈ A * (u1 + u2)

    v1 = rand(N, K)
    w1 = copy(v1)
    v2 = rand(N, K)
    w2 = copy(v2)
    a1, a2, b1, b2 = rand(4)

    mul!(v1, L, u1, a1, b1)
    @test v1 ≈ a1 * A * u1 + b1 * w1
    mul!(v2, L, u2, a2, b2)
    @test v2 ≈ a2 * A * u2 + b2 * w2
    @test v1 ≈ a1 * A * u1 + b1 * w1
    @test v1 + v2 ≈ (a1 * A * u1 + b1 * w1) + (a2 * A * u2 + b2 * w2)
end
#
