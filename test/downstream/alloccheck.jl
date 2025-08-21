using SciMLOperators, Random, SparseArrays, Test
using SciMLOperators: IdentityOperator,
                      NullOperator,
                      ScaledOperator,
                      AddedOperator

function apply_op!(H, w, v, u, p, t)
    H(w, v, u, p, t)
    return nothing
end

test_apply_noalloc(H, w, v, u, p, t) = @test (@allocations apply_op!(H, w, v, u, p, t)) == 0

@testset "Allocations Check" begin
    Random.seed!(0)
    N = 8
    K = 12
    A = rand(N, N) |> MatrixOperator
    B = rand(N, N) |> MatrixOperator
    u = rand(N, K)       # Update vector
    v = rand(N, K)       # Action vector
    w = zeros(N, K)      # Output vector
    p = ()
    t = 0
    op = AddedOperator(A, B)

    apply_op!(op, w, v, u, p, t) # Warm up
    if VERSION < v"1.10" || VERSION >= v"1.11"
        test_apply_noalloc(op, w, v, u, p, t)
    else
        # Julia 1.10 has a known allocation issue with AddedOperator
        @test (@allocations apply_op!(op, w, v, u, p, t)) == 1
    end

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
        v = rand(T, N)
        w = similar(u)
        p = (ω = 0.1,)
        t = 0.1

        apply_op!(H_sparse, w, v, u, p, t) # Warm up
        apply_op!(H_dense, w, v, u, p, t) # Warm up
        test_apply_noalloc(H_sparse, w, v, u, p, t)
        test_apply_noalloc(H_dense, w, v, u, p, t)
    end
end
