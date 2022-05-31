using SciMLOperators, LinearAlgebra
using Random

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
               *, /, \,
              )
        @test op(Id, u) ≈ u
        @test op(u, Id) ≈ u
    end

    v .= 0; @test mul!(v, Id, u) ≈ u
    v .= 0; @test ldiv!(v, Id, u) ≈ u

    # fix after working on composition operator
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

    for op in (
               *, /, \,
              )
        @test op(Z, u) ≈ zero(u)
        @test op(u, Z) ≈ zero(u)
    end

    v .= 0; @test mul!(v, Z, u) ≈ zero(u)

    # fix after working on composition operator
    #for op in (
    #           *, ∘,
    #          )
    #    @test op(id, a) ≈ a
    #    @test op(a, id) ≈ a
    #end
end

@testset "DiffEqScalar" begin
end

@testset "DiffEqArrayOperator" begin
    u = rand(N)
    p = nothing
    t = 0

    A  = rand(N,N)
    At = A'

    AA  = DiffEqArrayOperator(A)
    AAt = AA'

    @test AA  isa DiffEqArrayOperator
    @test AAt isa DiffEqArrayOperator

    FF  = factorize(AA)
    FFt = FF'

    @test FF  isa FactorizedDiffEqArrayOperator
    @test FFt isa FactorizedDiffEqArrayOperator

    @test eachindex(A)  === eachindex(AA)
    @test eachindex(A') === eachindex(AAt) === eachindex(DiffEqArrayOperator(At))

    @test A  ≈ convert(AbstractMatrix, AA ) ≈ convert(AbstractMatrix, FF )
    @test At ≈ convert(AbstractMatrix, AAt) ≈ convert(AbstractMatrix, FFt)

    @test A  ≈ Matrix(AA ) ≈ Matrix(FF )
    @test At ≈ Matrix(AAt) ≈ Matrix(FFt)

    @test A  * u ≈ AA(u,p,t)  ≈ FF(u,p,t)
    @test At * u ≈ AAt(u,p,t) ≈ FFt(u,p,t)

    @test A  \ u ≈ AA  \ u ≈ FF  \ u
    @test At \ u ≈ AAt \ u ≈ FFt \ u
end
#