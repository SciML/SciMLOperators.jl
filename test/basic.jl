using SciMLOperators, LinearAlgebra
using Random

using SciMLOperators: SciMLIdentity,
                      SciMLNullOperator,
                      SciMLScaledOperator,
                      SciMLAddedOperator,
                      SciMLComposedOperator,

                      getops

Random.seed!(0)
N = 8

@testset "SciMLIdentity" begin
    A  = rand(N, N) |> MatrixOperator
    u  = rand(N)
    v  = rand(N)
    Id = SciMLIdentity{N}()

    @test SciMLIdentity(u) isa SciMLIdentity{N}
    @test one(A) isa SciMLIdentity{N}
    @test convert(AbstractMatrix, Id) == Matrix(I, N, N)

    @test size(Id) == (N, N)
    @test Id' isa SciMLIdentity{N}

    for op in (
               *, \,
              )
        @test op(Id, u) ≈ u
    end

    v .= 0; @test mul!(v, Id, u) ≈ u
    v .= 0; @test ldiv!(v, Id, u) ≈ u

    # TODO fix after working on composition operator
    #for op in (
    #           *, ∘,
    #          )
    #    @test op(id, a) ≈ a
    #    @test op(a, id) ≈ a
    #end
end

@testset "SciMLNullOperator" begin
    A = rand(N, N) |> MatrixOperator
    u = rand(N)
    v = rand(N)
    Z = SciMLNullOperator{N}()

    @test SciMLNullOperator(u) isa SciMLNullOperator{N}
    @test zero(A) isa SciMLNullOperator{N}
    @test convert(AbstractMatrix, Z) == zeros(size(Z))

    @test size(Z) == (N, N)
    @test Z' isa SciMLNullOperator{N}

    @test *(Z, u) ≈ zero(u)

    v = rand(N); @test mul!(v, Z, u) ≈ zero(u)

    # TODO fix after working on composition operator
    #for op in (
    #           *, ∘,
    #          )
    #    @test op(id, a) ≈ a
    #    @test op(a, id) ≈ a
    #end
end

@testset "SciMLScalar" begin
    a = rand()
    x = rand()
    α = SciMLScalar(x)
    u = rand(N)

    @test α isa SciMLScalar
    @test convert(Number, α) isa Number
    @test convert(SciMLScalar, a) isa SciMLScalar

    @test size(α) == ()

    for op in (
               *, /, \, +, -,
              )
        @test op(α, a) ≈ op(x, a)
        @test op(a, α) ≈ op(a, x)
    end

    v = copy(u); @test lmul!(α, u) == v * x
    v = copy(u); @test rmul!(u, α) == x * v

    v .= 0; @test mul!(v, α, u) == u * x

    v = rand(N)
    w = copy(v)
    @test axpy!(α, u, v) == u * x + w

    @test abs(SciMLScalar(-x)) == x
end

@testset "SciMLScaledOperator" begin
    # TODO change A to a differnt type of SciMLScalar
    A = rand(N,N) |> MatrixOperator

    for T in (
              SciMLScalar,
              Number,
              UniformScaling
             )
        u = rand(N)
        α = rand()
        β = rand()

         αAu =     α * A * u
        βαAu = β * α * A * u

        α = α |> T
        β = β |> T

        op1 = α * A # not SciMLScaledOperator
        op2 = A * α # as * shortcircuits for SciMLScalar

        op1 = SciMLScaledOperator(α, A)
        op2 = SciMLScaledOperator(α, A)

        @test op1 isa SciMLScaledOperator
        @test op2 isa SciMLScaledOperator

        @test op1 * u ≈ αAu
        @test op2 * u ≈ αAu

        @test (β * op1) * u ≈ βαAu
        @test (β * op2) * u ≈ βαAu
    end
end

@testset "SciMLAddedOperator" begin
    A = rand(N,N) |> MatrixOperator
    B = rand(N,N) |> MatrixOperator
    C = rand(N,N) |> MatrixOperator
    α = rand() |> SciMLScalar
    β = rand() |> SciMLScalar
    u = rand(N)

    for op in (
               +, -
              )
        op1 = op(A  , B  )
        op2 = op(α*A, B  )
        op3 = op(A  , β*B)
        op4 = op(α*A, β*B)

        @test op1 isa SciMLAddedOperator
        @test op2 isa SciMLAddedOperator
        @test op3 isa SciMLAddedOperator
        @test op4 isa SciMLAddedOperator

        @test op1 * u ≈ op(  A*u,   B*u)
        @test op2 * u ≈ op(α*A*u,   B*u)
        @test op3 * u ≈ op(  A*u, β*B*u)
        @test op4 * u ≈ op(α*A*u, β*B*u)
    end
end

@testset "SciMLComposedOperator" begin
    A = rand(N,N) |> MatrixOperator
    B = rand(N,N) |> MatrixOperator
    C = rand(N,N) |> MatrixOperator

    u = rand(N)
    ABCmulu = (A * B * C) * u
    ABCdivu = (A * B * C) \ u

    op = ∘(A, B, C)

    @test op isa SciMLComposedOperator
    @test *(op.ops...) isa MatrixOperator

    @test op * u ≈ ABCmulu
    @test op \ u ≈ ABCdivu
end
#
