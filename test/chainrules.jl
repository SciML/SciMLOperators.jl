# Tests for ChainRules extension fixing gradient double-counting issue
# These tests specifically target issue #305

using SciMLOperators
using LinearSolve, Zygote, Test
using SciMLOperators: ScaledOperator

@testset "ChainRules fix for ScalarOperator gradient double-counting" begin
    # Test 1: Simple ScaledOperator creation
    @testset "Simple ScaledOperator gradient" begin
        simple_func = p -> 2.0 * p
        
        # Create ScalarOperator and matrix operator
        S = ScalarOperator(0.0, (A, u, p, t) -> simple_func(p))
        M = MatrixOperator(ones(2, 2))
        
        # Test that ScaledOperator creation doesn't double-count gradients
        function test_scaled(p)
            S_val = ScalarOperator(simple_func(p))
            scaled = S_val * M
            return scaled.λ.val
        end
        
        p_val = 0.5
        result = test_scaled(p_val)
        grad = Zygote.gradient(test_scaled, p_val)[1]
        
        @test result ≈ simple_func(p_val)
        @test grad ≈ 2.0  # Should not be doubled (4.0)
    end
    
    # Test 2: Full update_coefficients pipeline
    @testset "update_coefficients pipeline" begin
        exp_func = p -> exp(1 - p)
        
        A1 = MatrixOperator(rand(3, 3))
        A2 = MatrixOperator(rand(3, 3))
        Func = ScalarOperator(0.0, (A, u, p, t) -> exp_func(p))
        A = A1 + Func * A2
        
        # Test that update_coefficients doesn't cause gradient doubling
        function test_update(p)
            A_updated = update_coefficients(A, 0, p, 0)
            # Access the scalar value from the updated composition
            scaled_op = A_updated.ops[2]  # This should be the ScaledOperator
            return scaled_op.λ.val
        end
        
        p_val = 0.3
        result = test_update(p_val)
        grad = Zygote.gradient(test_update, p_val)[1]
        
        @test result ≈ exp_func(p_val)
        # Check that gradient matches the derivative of exp_func
        expected_grad = -exp(1 - p_val)  # derivative of exp(1-p) is -exp(1-p)
        @test grad ≈ expected_grad
    end
    
    # Test 3: Original MWE from issue #305
    @testset "Original MWE from issue #305" begin
        a1 = rand(3, 3)
        a2 = rand(3, 3)
        func = p -> exp(1 - p)
        a = p -> a1 + func(p) * a2
        
        A1 = MatrixOperator(a1)
        A2 = MatrixOperator(a2)
        Func = ScalarOperator(0.0, (A, u, p, t) -> func(p))
        A = A1 + Func * A2
        
        b = rand(3)
        
        function sol1(p)
            Ap = update_coefficients(A, 0, p, 0) |> concretize
            prob = LinearProblem(Ap, b)
            sol = solve(prob, KrylovJL_GMRES())
            return sum(sol.u)
        end
        
        function sol2(p)
            Ap = a(p)
            prob = LinearProblem(Ap, b)
            sol = solve(prob, KrylovJL_GMRES())
            return sum(sol.u)
        end
        
        p_val = rand()
        s1, s2 = sol1(p_val), sol2(p_val)
        
        # Primal solutions should match
        @test s1 ≈ s2
        
        grad1 = Zygote.gradient(sol1, p_val)[1]
        grad2 = Zygote.gradient(sol2, p_val)[1]
        
        # Gradients should match (no more doubling)
        @test grad1 ≈ grad2 rtol=1e-10
        @test !(grad1 ≈ 2 * grad2)  # Should NOT be doubled anymore
    end
    
    # Test 4: Direct ScaledOperator constructor (the specific case our rrule fixes)
    @testset "Direct ScaledOperator constructor" begin
        func = p -> 3.0 * p
        
        function test_direct_constructor(p)
            S = ScalarOperator(func(p))
            M = MatrixOperator([2.0 1.0; 1.0 2.0])
            scaled = ScaledOperator(S, M)  # This should use our rrule
            return scaled.λ.val
        end
        
        p_val = 0.5
        result = test_direct_constructor(p_val)
        grad = Zygote.gradient(test_direct_constructor, p_val)[1]
        
        @test result ≈ func(p_val)
        @test grad ≈ 3.0  # Should not be doubled (6.0)
    end
end