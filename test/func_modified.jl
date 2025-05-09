using SciMLOperators, LinearAlgebra
using Random
using Test

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
        u = rand(sz_in...)  # Update and action vector
        v = rand(sz_out...) # Output vector for in-place tests

        _mul(A, x) = reshape(A * vec(x), sz_out)
        f(x, p, t) = _mul(A, x)
        f(y, x, p, t) = (mul!(vec(y), A, vec(x)); y)

        kw = (;) # FunctionOp kwargs

        if sz_in == sz_out
            F = lu(A)
            _div(A, v) = reshape(A \ vec(v), sz_in)
            fi(x, p, t) = _div(A, x)
            fi(y, x, p, t) = (ldiv!(vec(y), F, vec(x)); y)

            kw = (; op_inverse = fi)
        end

        L = FunctionOperator(f, u, v; kw...)
        L = cache_operator(L, u)

        # test with ND-arrays and new interface
        @test _mul(A, u) ≈ L(u, u, p, t) ≈ L * u ≈ mul!(zero(v), L, u)
        
        # Test with different update and action vectors
        action_vec = rand(sz_in...)
        @test _mul(A, action_vec) ≈ L(action_vec, u, p, t)
        
        # Test in-place with different update and action vectors
        output_vec = zeros(sz_out...)
        L(output_vec, action_vec, u, p, t)
        @test output_vec ≈ _mul(A, action_vec)
        
        # Test in-place with scaling
        output_vec = rand(sz_out...)
        orig_output = copy(output_vec)
        L(output_vec, action_vec, u, p, t, α, β)
        @test output_vec ≈ α * _mul(A, action_vec) + β * orig_output

        if sz_in == sz_out
            # Test inverse operations with new interface
            @test _div(A, v) ≈ L \ v
            
            # Test in-place inverse
            w = zeros(sz_in...)
            ldiv!(w, L, v)
            @test w ≈ _div(A, v)
        end
    end
end

@testset "(Unbatched) FunctionOperator" begin
    u = rand(N, K)  # Update and action vector
    v = zeros(N, K) # Output vector
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
        isposdef = true)

    # in place
    op2 = FunctionOperator(f2, u; op_inverse = f2i, ifcache = false, islinear = true,
        opnorm = true,
        issymmetric = true,
        ishermitian = true,
        isposdef = true)

    # Test traits
    @test issquare(op1)
    @test issquare(op2)
    @test islinear(op1)
    @test islinear(op2)
    @test op1' === op1
    
    # Test operator properties
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

    # Create test vectors for new interface
    action_vec = rand(N, K)  # Action vector
    update_vec = rand(N, K)  # Update vector 
    result_vec = zeros(N, K) # Result vector

    # Cache operators
    op1 = cache_operator(op1, u)
    op2 = cache_operator(op2, u)

    @test iscached(op1)
    @test iscached(op2)

    # Test with new interface - out of place
    @test _mul(A, action_vec) ≈ op1(action_vec, update_vec, p, t)
    
    # Test with new interface - in place
    op2(result_vec, action_vec, update_vec, p, t)
    @test result_vec ≈ _mul(A, action_vec)
    
    # Test in-place with scaling
    result_vec = rand(N, K)
    orig_result = copy(result_vec)
    op2(result_vec, action_vec, update_vec, p, t, α, β)
    @test result_vec ≈ α * _mul(A, action_vec) + β * orig_result

    # Test inverse operations with new interface
    inv_result = zeros(N, K)
    @test _div(A, action_vec) ≈ op1 \ action_vec
    ldiv!(inv_result, op2, action_vec)
    @test inv_result ≈ _div(A, action_vec)
end

@testset "FunctionOperator update test" begin
    u = rand(N, K)  # Update vector
    v = rand(N, K)  # Action vector
    w = zeros(N, K) # Result vector
    p = rand(N)
    t = rand()
    scale = rand()

    # Accept a kwarg "scale" in operator action
    f(du, x, p, t; scale = 1.0) = mul!(du, Diagonal(p * t * scale), x)
    f(x, p, t; scale = 1.0) = Diagonal(p * t * scale) * x

    # Function operator with keyword arguments
    L = FunctionOperator(f, u, u; 
                         p = zero(p), 
                         t = zero(t), 
                         batch = true,
                         accepted_kwargs = (:scale,), 
                         scale = 1.0)

    @test size(L) == (N, N)

    # Expected result with scaling
    A = Diagonal(p * t * scale)
    expected = A * v
    
    # Test with new interface
    @test L(v, u, p, t; scale) ≈ expected
    
    # Test in-place with new interface
    copy!(w, zeros(N, K))
    L(w, v, u, p, t; scale) 
    @test w ≈ expected
    
    # Test in-place with scaling
    copy!(w, rand(N, K))
    orig_w = copy(w)
    α_val = rand()
    β_val = rand()
    L(w, v, u, p, t, α_val, β_val; scale)
    @test w ≈ α_val * expected + β_val * orig_w
    
    # Test that outputs aren't accidentally mutated
    u1 = rand(N, K)
    u2 = rand(N, K)
    v1 = rand(N, K)
    v2 = rand(N, K)

    # Expected results with different vectors
    result1 = A * v1
    result2 = A * v2
    
    # Test output consistency
    w1 = zeros(N, K)
    w2 = zeros(N, K)
    
    L(w1, v1, u1, p, t; scale)
    L(w2, v2, u2, p, t; scale)
    
    @test w1 ≈ result1
    @test w2 ≈ result2
end