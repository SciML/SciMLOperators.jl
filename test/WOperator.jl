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
end
