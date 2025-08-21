using Pkg
Pkg.activate(".")
using SciMLOperators
using LinearAlgebra
using Test

println("Testing basic operators with copy methods...")

# Test that operators still work correctly after copying
@testset "Basic operator functionality after copy" begin
    # MatrixOperator
    A = rand(5, 5)
    L = MatrixOperator(A)
    L_copy = copy(L)
    v = rand(5)
    @test L * v ≈ L_copy * v
    
    # ScalarOperator
    α = ScalarOperator(2.0)
    α_copy = copy(α)
    @test α * v ≈ α_copy * v
    
    # ComposedOperator
    B = MatrixOperator(rand(5, 5))
    comp = L ∘ B
    comp_copy = copy(comp)
    @test comp * v ≈ comp_copy * v
    
    # AffineOperator
    b = rand(5)
    aff = AffineOperator(L, B, b)
    aff_copy = copy(aff)
    @test aff * v ≈ aff_copy * v
end

println("All basic functionality tests passed!")