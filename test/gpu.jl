# GPU tests for SciMLOperators
# These tests require CUDA.jl and a CUDA-capable GPU
# Run with: julia --project=. -e 'include("test/gpu.jl")'

using Test
using SciMLOperators
using FillArrays
using SparseArrays

# Only run tests if CUDA is available
cuda_available = try
    using CUDA
    CUDA.functional()
catch
    false
end

if cuda_available
    using CUDA

    @testset "GPU Scalar Indexing Tests (Issue #338)" begin
        # Disable scalar indexing to catch any fallback iterations
        CUDA.allowscalar(false)

        N = 4

        @testset "AddedOperator with CuArray (dense)" begin
            A = CUDA.rand(Float32, N, N)
            II = Eye{Float32}(N)
            v = CUDA.rand(Float32, N)

            # Test MatrixOperator alone
            M1 = MatrixOperator(A)
            @test M1 * v isa CuArray

            # Test AddedOperator - this was failing before the fix
            M2 = MatrixOperator(A) - II
            @test_nowarn result = M2 * v
            @test result isa CuArray
        end

        @testset "AddedOperator with CuSparseMatrix" begin
            # Create sparse matrix on GPU
            A_cpu = sprand(Float32, N, N, 0.5)
            A = CUDA.CUSPARSE.CuSparseMatrixCSC(A_cpu)
            II = Eye{Float32}(N)
            v = CUDA.rand(Float32, N)

            # Test MatrixOperator alone
            M1 = MatrixOperator(A)
            @test M1 * v isa CuArray

            # Test AddedOperator - this was the main issue in #338
            M2 = MatrixOperator(A) - II
            @test_nowarn result = M2 * v
            @test result isa CuArray
        end

        @testset "AddedOperator with multiple operators" begin
            A = CUDA.rand(Float32, N, N)
            B = CUDA.rand(Float32, N, N)
            v = CUDA.rand(Float32, N)

            # Multiple MatrixOperators added together
            M = MatrixOperator(A) + MatrixOperator(B)
            @test_nowarn result = M * v
            @test result isa CuArray
        end

        @testset "mul! should also work" begin
            A = CUDA.rand(Float32, N, N)
            II = Eye{Float32}(N)
            v = CUDA.rand(Float32, N)
            w = similar(v)

            M2 = MatrixOperator(A) - II
            @test_nowarn mul!(w, M2, v)
            @test w isa CuArray
        end

        # Re-enable scalar indexing for any subsequent tests
        CUDA.allowscalar(true)
    end
else
    @info "CUDA not available, skipping GPU tests"
    @test_skip "GPU tests require CUDA"
end
