# Test Enzyme integration with SciMLOperators
# Verifies that gradients can be computed through operators with parameter-dependent coefficients
# Related to issue #319

using SciMLOperators, Enzyme, LinearAlgebra, SparseArrays, Test

const T = Float64

# Test basic operator autodiff with Enzyme
@testset "Enzyme autodiff with ScalarOperator" begin
    # Create operators with parameter-dependent coefficients
    coef1(a, u, p, t) = -p[1]
    coef2(a, u, p, t) = p[2]

    A1_data = sparse(T[0.0 1.0; 0.0 0.0])
    A2_data = sparse(T[0.0 0.0; 1.0 0.0])

    c1 = ScalarOperator(one(T), coef1)
    c2 = ScalarOperator(one(T), coef2)

    U = c1 * MatrixOperator(A1_data) + c2 * MatrixOperator(A2_data)

    # Simple loss function that uses the operator
    function loss(p)
        x = T[3.0, 4.0]
        t = 0.0

        # Update coefficients and apply operator
        U_updated = update_coefficients(U, x, p, t)
        y = U_updated * x

        return sum(abs2, y)
    end

    # Test that Enzyme can compute gradients
    p = T[1.0, 2.0]
    dp = Enzyme.make_zero(p)

    result = Enzyme.autodiff(Enzyme.Reverse, loss, Active, Duplicated(p, dp))

    # Gradient should not be NaN (the original bug)
    @test !any(isnan, dp)
    @test !any(isinf, dp)

    # Gradient should be non-zero (operators depend on parameters)
    @test any(!iszero, dp)
end

@testset "Enzyme autodiff with MatrixOperator" begin
    # Test with matrix operator that has update function
    update_func(A, u, p, t) = p[1] * A

    A_data = T[1.0 2.0; 3.0 4.0]
    L = MatrixOperator(A_data; update_func = update_func)

    function loss2(p)
        x = T[1.0, 1.0]
        t = 0.0

        L_updated = update_coefficients(L, x, p, t)
        y = L_updated * x

        return sum(abs2, y)
    end

    p = T[2.0]
    dp = Enzyme.make_zero(p)

    result = Enzyme.autodiff(Enzyme.Reverse, loss2, Active, Duplicated(p, dp))

    # Gradient should be valid
    @test !any(isnan, dp)
    @test !any(isinf, dp)
    @test any(!iszero, dp)
end

@testset "Enzyme autodiff with composed operators" begin
    # Test more complex operator composition
    coef(a, u, p, t) = p[1]

    A = MatrixOperator(T[1.0 0.0; 0.0 1.0])
    B = MatrixOperator(T[2.0 1.0; 1.0 2.0])
    α = ScalarOperator(one(T), coef)

    # Composed operator: α * A + B
    C = α * A + B

    function loss3(p)
        x = T[1.0, 2.0]
        t = 0.0

        C_updated = update_coefficients(C, x, p, t)
        y = C_updated * x

        return sum(y)
    end

    p = T[3.0]
    dp = Enzyme.make_zero(p)

    result = Enzyme.autodiff(Enzyme.Reverse, loss3, Active, Duplicated(p, dp))

    # Gradient should be valid
    @test !any(isnan, dp)
    @test !any(isinf, dp)
end
