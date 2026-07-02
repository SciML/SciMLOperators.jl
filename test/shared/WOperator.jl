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
end
