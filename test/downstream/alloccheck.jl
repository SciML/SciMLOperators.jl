using SciMLOperators, AllocCheck, Random, SparseArrays, Test
using SciMLOperators: IdentityOperator,
                      NullOperator,
                      ScaledOperator,
                      AddedOperator
Random.seed!(0)
N = 8
K = 12
A = rand(N, N) |> MatrixOperator
B = rand(N, N) |> MatrixOperator
C = rand(N, N) |> MatrixOperator
α = rand()
β = rand()
u = rand(N, K)       # Update vector
v = rand(N, K)       # Action vector
w = zeros(N, K)      # Output vector
p = ()
t = 0
op = AddedOperator(A, B)

# Define a function to test allocations with the new interface
@check_allocs ignore_throw = true function apply_op!(H, w, v, u, p, t)
    H(w, v, u, p, t)
    return nothing
end

if VERSION >= v"1.12-beta"
    apply_op!(op, w, v, u, p, t)
else
    @test_throws AllocCheckFailure apply_op!(op, w, v, u, p, t)
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

    @test_throws AllocCheckFailure apply_op!(H_sparse, w, v, u, p, t)
    @test_throws AllocCheckFailure apply_op!(H_dense, w, v, u, p, t)
end
