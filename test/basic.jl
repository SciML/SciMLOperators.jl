using SciMLOperators, LinearAlgebra
using Random

using SciMLOperators: IdentityOperator,
                      NullOperator,
                      ScaledOperator,
                      AddedOperator,
                      ComposedOperator,

                      getops

Random.seed!(0)
N = 8

@testset "IdentityOperator" begin
    A  = rand(N, N) |> MatrixOperator
    u  = rand(N)
    v  = rand(N)
    Id = IdentityOperator{N}()

    @test IdentityOperator(u) isa IdentityOperator{N}
    @test one(A) isa IdentityOperator{N}
    @test convert(AbstractMatrix, Id) == Matrix(I, N, N)

    @test size(Id) == (N, N)
    @test Id' isa IdentityOperator{N}

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

@testset "NullOperator" begin
    A = rand(N, N) |> MatrixOperator
    u = rand(N)
    v = rand(N)
    Z = NullOperator{N}()

    @test NullOperator(u) isa NullOperator{N}
    @test zero(A) isa NullOperator{N}
    @test convert(AbstractMatrix, Z) == zeros(size(Z))

    @test size(Z) == (N, N)
    @test Z' isa NullOperator{N}

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

@testset "ScalarOperator" begin
    a = rand()
    x = rand()
    α = ScalarOperator(x)
    u = rand(N)

    @test α isa ScalarOperator
    @test convert(Number, α) isa Number
    @test convert(ScalarOperator, a) isa ScalarOperator

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

    @test abs(ScalarOperator(-x)) == x
end

@testset "ScaledOperator" begin
    # TODO change A to a differnt type of ScalarOperator
    A = rand(N,N) |> MatrixOperator

    for T in (
              ScalarOperator,
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

        op1 = α * A # not ScaledOperator
        op2 = A * α # as * shortcircuits for ScalarOperator

        op1 = ScaledOperator(α, A)
        op2 = ScaledOperator(α, A)

        @test op1 isa ScaledOperator
        @test op2 isa ScaledOperator

        @test op1 * u ≈ αAu
        @test op2 * u ≈ αAu

        @test (β * op1) * u ≈ βαAu
        @test (β * op2) * u ≈ βαAu
    end
end

@testset "AddedOperator" begin
    A = rand(N,N) |> MatrixOperator
    B = rand(N,N) |> MatrixOperator
    C = rand(N,N) |> MatrixOperator
    α = rand() |> ScalarOperator
    β = rand() |> ScalarOperator
    u = rand(N)

    for op in (
               +, -
              )
        op1 = op(A  , B  )
        op2 = op(α*A, B  )
        op3 = op(A  , β*B)
        op4 = op(α*A, β*B)

        @test op1 isa AddedOperator
        @test op2 isa AddedOperator
        @test op3 isa AddedOperator
        @test op4 isa AddedOperator

        @test op1 * u ≈ op(  A*u,   B*u)
        @test op2 * u ≈ op(α*A*u,   B*u)
        @test op3 * u ≈ op(  A*u, β*B*u)
        @test op4 * u ≈ op(α*A*u, β*B*u)
    end
end

@testset "ComposedOperator" begin
    A = rand(N,N) |> MatrixOperator
    B = rand(N,N) |> MatrixOperator
    C = rand(N,N) |> MatrixOperator

    u = rand(N)
    ABCmulu = (A * B * C) * u
    ABCdivu = (A * B * C) \ u

    op = ∘(A, B, C)

    @test op isa ComposedOperator
    @test *(op.ops...) isa MatrixOperator

    @test op * u ≈ ABCmulu
    @test op \ u ≈ ABCdivu
end
#
