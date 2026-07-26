using SciMLOperators, LinearAlgebra
using Random
using Test

Random.seed!(0)
@testset "WOperator" begin
    J = rand(12, 12)
    u = rand(12)
    M = I(12)
    gamma = 1 / 123
    W = WOperator{true}(M, gamma, J, u)

    @test convert(AbstractMatrix, W) ≈ J - M / gamma

    # Scalar WOperator
    J_scalar = ScalarOperator(2.0)
    gamma_scalar = 0.5
    u_scalar = 1.0
    W_scalar = WOperator{false}(I, gamma_scalar, J_scalar, u_scalar)
    @test convert(Number, W_scalar) ≈ 2.0 - 1.0 / 0.5

    # Update and convert again
    update_coefficients!(W_scalar; gamma = 0.25)
    @test convert(Number, W_scalar) ≈ 2.0 - 1.0 / 0.25

    # Test deprecated dtgamma kwarg on WOperator
    W_scalar2 = WOperator{false}(I, 0.5, ScalarOperator(2.0), 1.0)
    @test_deprecated update_coefficients!(W_scalar2; dtgamma = 0.25)
    @test convert(Number, W_scalar2) ≈ 2.0 - 1.0 / 0.25

    # OOP WOperator with a MatrixOperator mass matrix: convert(AbstractMatrix, W)
    # must materialize the mass matrix so the result fits in the
    # Matrix-typed _concrete_form slot (issue #396).
    n = 3
    mm_data = randn(n, n)
    update_mm!(A, u, p, t) = (A[1, 1] = cos(t); A[2, 2] = u[1]; A)
    update_mm(A, u, p, t) = (B = copy(A); update_mm!(B, u, p, t); B)
    mm_op = MatrixOperator(
        copy(mm_data); update_func = update_mm, update_func! = update_mm!,
    )
    J_mm = randn(n, n)
    u_mm = randn(n)
    gamma_mm = 0.1
    W_mm = WOperator{false}(mm_op, gamma_mm, J_mm, u_mm)
    W_ref = -convert(AbstractMatrix, mm_op) / gamma_mm + J_mm
    @test convert(AbstractMatrix, W_mm) ≈ W_ref
    # Re-materialization after update_coefficients!
    update_coefficients!(W_mm, randn(n), nothing, 1.5; gamma = 0.2)
    W_ref2 = -convert(AbstractMatrix, W_mm.mass_matrix) / 0.2 + J_mm
    @test convert(AbstractMatrix, W_mm) ≈ W_ref2
    # And `\` works end-to-end
    b = randn(n)
    @test W_mm \ b ≈ convert(AbstractMatrix, W_mm) \ b

    # In-place WOperator with an operator-typed Jacobian: no external caller
    # maintains _concrete_form (jacobian2W! only handles plain-matrix J), so
    # convert must rebuild it with the current gamma. Regression test for the
    # silently-stale form built at init with gamma = 0 (Inf diagonal), see
    # SciML/OrdinaryDiffEq.jl#3933.
    J_op = MatrixOperator(randn(n, n))
    W_op = WOperator{true}(I(n), 0.0, J_op, randn(n))
    update_coefficients!(W_op; gamma = 0.25)
    @test convert(AbstractMatrix, W_op) ≈ -Matrix(1.0I, n, n) / 0.25 +
        convert(AbstractMatrix, J_op)
    @test all(isfinite, convert(AbstractMatrix, W_op))
    # Rebuilt again after gamma changes
    update_coefficients!(W_op; gamma = 0.5)
    @test convert(AbstractMatrix, W_op) ≈ -Matrix(1.0I, n, n) / 0.5 +
        convert(AbstractMatrix, J_op)

    J_sum = MatrixOperator(randn(n, n)) + MatrixOperator(randn(n, n))
    W_sum = WOperator{true}(I, 0.0, J_sum, randn(n))
    update_coefficients!(W_sum; gamma = 0.25)
    @test convert(AbstractMatrix, W_sum) ≈ -Matrix(1.0I, n, n) / 0.25 +
        convert(AbstractMatrix, J_sum)
    update_coefficients!(W_sum; gamma = 0.5)
    @test convert(AbstractMatrix, W_sum) ≈ -Matrix(1.0I, n, n) / 0.5 +
        convert(AbstractMatrix, J_sum)
end

@testset "WOperator isconvertible honesty" begin
    n = 5
    u = rand(n)
    gamma = 0.1
    Mmat = Matrix(1.0I, n, n)
    concrete_J = MatrixOperator(rand(n, n))
    # A matrix-free operator: only `mul!`, no `convert(AbstractMatrix, ·)`.
    matfree(A) = FunctionOperator(
        (w, v, p, t) -> mul!(w, A, v), zeros(n), zeros(n); islinear = true
    )
    matfree_J = matfree(rand(n, n))
    matfree_M = matfree(Mmat)

    @test isconvertible(matfree_J) == false
    # `W` is convertible only when both the mass matrix and the Jacobian are; a matrix-free
    # part on either side makes `W` matrix-free (previously `isconvertible(W)` was vacuously
    # `true` because `getops(W) == ()`).
    @test isconvertible(WOperator{true}(Mmat, gamma, concrete_J, u)) == true
    @test isconvertible(WOperator{true}(Mmat, gamma, matfree_J, u)) == false
    @test isconvertible(WOperator{true}(matfree_M, gamma, concrete_J, u)) == false
end
