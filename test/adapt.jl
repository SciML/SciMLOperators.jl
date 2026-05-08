using SciMLOperators, Adapt, Test, LinearAlgebra

#
# Adapt.jl Integration Tests
#
# Tests verify that Adapt.adapt_structure correctly handles array-backed operators
# and recursively adapts nested/composite operators without GPU dependencies.
#

# ============================================================================
# Custom Test Adaptor (no GPU required)
# ============================================================================

"""
    CustomArray{T, N, AT <: AbstractArray{T, N}}

Lightweight array wrapper used to prove Adapt changed storage types.
"""
struct CustomArray{T, N, AT <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data::AT
end

CustomArray(x::AT) where {T, N, AT <: AbstractArray{T, N}} = CustomArray{T, N, AT}(x)

Base.size(x::CustomArray) = size(x.data)
Base.getindex(x::CustomArray, I...) = getindex(x.data, I...)

function Adapt.adapt_storage(::Type{CustomArray}, x::AbstractArray)
    return CustomArray(x)
end

# ============================================================================
# Helpers
# ============================================================================

"""
    has_customarray(obj)

Recursively detect whether any field in `obj` contains a `CustomArray`.
"""
function has_customarray(obj)
    if obj isa CustomArray
        return true
    elseif obj isa Tuple || obj isa AbstractVector
        return any(has_customarray, obj)
    elseif obj isa SciMLOperators.AbstractSciMLOperator
        return any(fname -> has_customarray(getfield(obj, fname)), fieldnames(typeof(obj)))
    else
        return false
    end
end

# ============================================================================
# Test Suite
# ============================================================================

@testset "Adapt.jl Support for SciMLOperators" begin

    # ========================================================================
    # Direct Array-Backed Operators
    # ========================================================================

    @testset "MatrixOperator" begin
        L = MatrixOperator(rand(5, 5))
        L_adapted = Adapt.adapt_structure(CustomArray, L)

        @test L_adapted isa MatrixOperator
        @test L_adapted.A isa CustomArray
        @test L_adapted.update_func === L.update_func
        @test L_adapted.update_func! === L.update_func!
    end

    @testset "BatchedDiagonalOperator" begin
        L = DiagonalOperator(rand(5, 3))
        L_adapted = Adapt.adapt_structure(CustomArray, L)

        # DiagonalOperator constructor returns BatchedDiagonalOperator.
        @test L_adapted isa SciMLOperators.BatchedDiagonalOperator
        @test L_adapted.diag isa CustomArray
        @test L_adapted.update_func === L.update_func
        @test L_adapted.update_func! === L.update_func!
    end

    @testset "AffineOperator" begin
        A = MatrixOperator(rand(5, 5))
        B = MatrixOperator(rand(5, 5))
        L = AffineOperator(A, B, rand(5))
        L_adapted = Adapt.adapt_structure(CustomArray, L)

        @test L_adapted isa AffineOperator
        @test L_adapted.A isa MatrixOperator
        @test L_adapted.B isa MatrixOperator
        @test L_adapted.A.A isa CustomArray
        @test L_adapted.B.A isa CustomArray
        @test L_adapted.b isa CustomArray
    end

    @testset "FunctionOperator" begin
        input = rand(5)
        output = rand(5)
        op = (v, u, p, t) -> 0.5 .* v
        L = FunctionOperator(op, input, output; u = rand(3), p = rand(4), t = 0.0)
        L_adapted = Adapt.adapt_structure(CustomArray, L)

        @test L_adapted isa FunctionOperator
        @test L_adapted.u isa CustomArray
        @test L_adapted.p isa CustomArray
    end

    # ========================================================================
    # Recursive Wrapper Operators
    # ========================================================================

    @testset "ScaledOperator" begin
        L = 2.5 * MatrixOperator(rand(5, 5))
        L_adapted = Adapt.adapt_structure(CustomArray, L)

        @test L_adapted isa SciMLOperators.ScaledOperator
        @test L_adapted.L isa MatrixOperator
        @test L_adapted.L.A isa CustomArray
    end

    @testset "AddedOperator" begin
        L = MatrixOperator(rand(5, 5)) + DiagonalOperator(rand(5, 3))
        L_adapted = Adapt.adapt_structure(CustomArray, L)

        @test L_adapted isa SciMLOperators.AddedOperator
        @test L_adapted.ops[1] isa MatrixOperator
        @test L_adapted.ops[2] isa SciMLOperators.BatchedDiagonalOperator
        @test L_adapted.ops[1].A isa CustomArray
        @test L_adapted.ops[2].diag isa CustomArray
    end

    @testset "ComposedOperator" begin
        L = MatrixOperator(rand(5, 5)) * MatrixOperator(rand(5, 5))
        L_adapted = Adapt.adapt_structure(CustomArray, L)

        @test L_adapted isa SciMLOperators.ComposedOperator
        @test L_adapted.ops[1].A isa CustomArray
        @test L_adapted.ops[2].A isa CustomArray
    end

    @testset "InvertedOperator" begin
        L = inv(MatrixOperator(rand(5, 5)))
        L_adapted = Adapt.adapt_structure(CustomArray, L)

        @test L_adapted isa SciMLOperators.InvertedOperator
        @test L_adapted.L.A isa CustomArray
    end

    @testset "InvertibleOperator" begin
        L = lu(MatrixOperator(rand(5, 5)))
        L_adapted = Adapt.adapt_structure(CustomArray, L)

        @test L_adapted isa SciMLOperators.InvertibleOperator
        @test L_adapted.L.A isa CustomArray
    end

    @testset "BlockDiagonalOperator" begin
        L = BlockDiagonalOperator(MatrixOperator(rand(3, 3)), MatrixOperator(rand(2, 2)))
        L_adapted = Adapt.adapt_structure(CustomArray, L)

        @test L_adapted isa SciMLOperators.BlockDiagonalOperator
        @test L_adapted.ops[1].A isa CustomArray
        @test L_adapted.ops[2].A isa CustomArray
    end

    @testset "AdjointOperator/TransposedOperator" begin
        L_adj = SciMLOperators.AdjointOperator(MatrixOperator(rand(5, 5)))
        L_t = SciMLOperators.TransposedOperator(MatrixOperator(rand(5, 5)))

        L_adj_adapted = Adapt.adapt_structure(CustomArray, L_adj)
        L_t_adapted = Adapt.adapt_structure(CustomArray, L_t)

        @test L_adj_adapted isa SciMLOperators.AdjointOperator
        @test L_adj_adapted.L isa MatrixOperator
        @test L_adj_adapted.L.A isa CustomArray

        @test L_t_adapted isa SciMLOperators.TransposedOperator
        @test L_t_adapted.L isa MatrixOperator
        @test L_t_adapted.L.A isa CustomArray
    end

    @testset "TensorProductOperator" begin
        import SciMLOperators.⊗
        L = MatrixOperator(rand(3, 3)) ⊗ MatrixOperator(rand(2, 2))
        L_adapted = Adapt.adapt_structure(CustomArray, L)

        @test L_adapted isa SciMLOperators.TensorProductOperator
        @test L_adapted.ops[1].A isa CustomArray
        @test L_adapted.ops[2].A isa CustomArray
    end

    @testset "TensorSumOperator" begin
        L = kronsum(MatrixOperator(rand(3, 3)), MatrixOperator(rand(2, 2)))
        L_adapted = Adapt.adapt_structure(CustomArray, L)

        @test L_adapted isa SciMLOperators.TensorSumOperator
        @test L_adapted.ops[1].A isa CustomArray
        @test L_adapted.ops[2].A isa CustomArray
    end

    # ========================================================================
    # Complex Nested Structures
    # ========================================================================

    @testset "Deeply Nested Composite Operator" begin
        L_nested = (MatrixOperator(rand(5, 5)) + DiagonalOperator(rand(5, 3))) * MatrixOperator(rand(5, 5))
        L_adapted = Adapt.adapt_structure(CustomArray, L_nested)

        @test L_adapted isa SciMLOperators.ComposedOperator
        @test L_adapted.ops[1] isa SciMLOperators.AddedOperator
        @test has_customarray(L_adapted)
    end

    @testset "Affine with Composed Operators" begin
        L = 2.0 * AffineOperator(MatrixOperator(rand(5, 5)), MatrixOperator(rand(5, 5)), rand(5))
        L_adapted = Adapt.adapt_structure(CustomArray, L)

        @test L_adapted isa SciMLOperators.ScaledOperator
        @test L_adapted.L isa SciMLOperators.AffineOperator
        @test L_adapted.L.A.A isa CustomArray
        @test L_adapted.L.B.A isa CustomArray
        @test L_adapted.L.b isa CustomArray
    end

    # ========================================================================
    # Array-Free Operators
    # ========================================================================

    @testset "Array-Free Operators" begin
        L_id = Adapt.adapt_structure(CustomArray, IdentityOperator(5))
        L_null = Adapt.adapt_structure(CustomArray, NullOperator(5, 5))

        @test L_id isa SciMLOperators.IdentityOperator
        @test L_null isa SciMLOperators.NullOperator
        @test !has_customarray(L_id)
        @test !has_customarray(L_null)
    end

end

println("All Adapt tests passed!")
