using SciMLOperators, LinearAlgebra
using Random

using SciMLOperators: DiffEqIdentity,
                      DiffEqNullOperator,
                      ScaledDiffEqOperator,
                      AddedDiffEqOperator

Random.seed!(0)
N = 8

@testset "DiffEqIdentity" begin
    A  = rand(N, N) |> DiffEqArrayOperator
    u  = rand(N)
    v  = rand(N)
    Id = DiffEqIdentity{N}()

    @test DiffEqIdentity(u) isa DiffEqIdentity{N}
    @test one(A) isa DiffEqIdentity{N}
    @test convert(AbstractMatrix, Id) == Matrix(I, N, N)

    @test size(Id) == (N, N)
    @test Id' isa DiffEqIdentity{N}

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

@testset "DiffEqNullOperator" begin
    A = rand(N, N) |> DiffEqArrayOperator
    u = rand(N)
    v = rand(N)
    Z = DiffEqNullOperator{N}()

    @test DiffEqNullOperator(u) isa DiffEqNullOperator{N}
    @test zero(A) isa DiffEqNullOperator{N}
    @test convert(AbstractMatrix, Z) == zeros(size(Z))

    @test size(Z) == (N, N)
    @test Z' isa DiffEqNullOperator{N}

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

@testset "DiffEqScalar" begin
    a = rand()
    x = rand()
    α = DiffEqScalar(x)
    u = rand(N)

    @test α isa DiffEqScalar
    @test convert(Number, α) isa Number
    @test convert(DiffEqScalar, a) isa DiffEqScalar

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

    @test abs(DiffEqScalar(-x)) == x
end

@testset "ScaledDiffEqOperator" begin
    # TODO change A to a differnt type of diffeqoperator
    A = rand(N,N) |> DiffEqArrayOperator

    for T in (
              DiffEqScalar,
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

        op1 = α * A # not ScaledDiffEqOperator
        op2 = A * α # as * shortcircuits for DiffEqArrayOp

        op1 = ScaledDiffEqOperator(α, A)
        op2 = ScaledDiffEqOperator(α, A)

        @test op1 isa ScaledDiffEqOperator
        @test op2 isa ScaledDiffEqOperator

        @test op1 * u ≈ αAu
        @test op2 * u ≈ αAu

        @test (β * op1) * u ≈ βαAu
        @test (β * op2) * u ≈ βαAu
    end
end

@testset "AddedDiffEqOperator" begin
    A = rand(N,N) |> DiffEqArrayOperator
    B = rand(N,N) |> DiffEqArrayOperator
    C = rand(N,N) |> DiffEqArrayOperator
    α = rand() |> DiffEqScalar
    β = rand() |> DiffEqScalar
    u = rand(N)

    for op in (
               +, -
              )
        op1 = op(A  , B  )
        op2 = op(α*A, B  )
        op3 = op(A  , β*B)
        op4 = op(α*A, β*B)

        @test op1 isa AddedDiffEqOperator
        @test op2 isa AddedDiffEqOperator
        @test op3 isa AddedDiffEqOperator
        @test op4 isa AddedDiffEqOperator

        @test op1 * u ≈ op(  A*u,   B*u)
        @test op2 * u ≈ op(α*A*u,   B*u)
        @test op3 * u ≈ op(  A*u, β*B*u)
        @test op4 * u ≈ op(α*A*u, β*B*u)
    end
end

@testset "ComposedDiffEqOperator" begin
end

@testset "Operator Algebra" begin
end
#
