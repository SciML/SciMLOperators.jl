using LinearAlgebra, SciMLOperators, Test

@testset "Documented public operator API" begin
    owner_public_names = (
        :AbstractSciMLOperator,
        :AbstractSciMLScalarOperator,
        :ScaledOperator,
        :AddedOperator,
        :ComposedOperator,
        :InvertedOperator,
        :AdjointOperator,
        :TransposedOperator,
        :AddedScalarOperator,
        :ComposedScalarOperator,
        :InvertedScalarOperator,
        :has_tensor_outer_mul_fast,
        :tensor_outer_mul_fast!,
        :getcache,
        :update_cache,
        :adopt_cache,
    )

    for name in owner_public_names
        @test isdefined(SciMLOperators, name)
        @test !Base.isexported(SciMLOperators, name)
        @static if isdefined(Base, :ispublic)
            @test Base.ispublic(SciMLOperators, name)
        end

        @static if isdefined(Base.Docs, :hasdoc)
            @test Base.Docs.hasdoc(SciMLOperators, name)
        else
            doc = sprint(
                show,
                MIME"text/plain"(),
                Base.Docs.doc(Base.Docs.Binding(SciMLOperators, name)),
            )
            @test !occursin("No documentation found", doc)
        end
    end
end

mutable struct GenericDiagonalOperator{T} <: SciMLOperators.AbstractSciMLOperator{T}
    diagonal::Vector{T}
    scale::T
end

Base.size(L::GenericDiagonalOperator) = (length(L.diagonal), length(L.diagonal))
Base.:*(L::GenericDiagonalOperator, v::AbstractVector) = L.scale .* L.diagonal .* v
Base.convert(::Type{AbstractMatrix}, L::GenericDiagonalOperator) = Diagonal(L.scale .* L.diagonal)

function LinearAlgebra.mul!(w::AbstractVector, L::GenericDiagonalOperator, v::AbstractVector)
    w .= L.scale .* L.diagonal .* v
    return w
end

function LinearAlgebra.mul!(
        w::AbstractVector, L::GenericDiagonalOperator, v::AbstractVector, α, β
    )
    w .= α .* L.scale .* L.diagonal .* v .+ β .* w
    return w
end

SciMLOperators.islinear(::GenericDiagonalOperator) = true
SciMLOperators.isconvertible(::GenericDiagonalOperator) = true
SciMLOperators.has_concretization(::GenericDiagonalOperator) = true
SciMLOperators.isconstant(::GenericDiagonalOperator) = false

function SciMLOperators.update_coefficients(
        L::GenericDiagonalOperator, u, p, t; scale = p
    )
    return GenericDiagonalOperator(copy(L.diagonal), scale)
end

function SciMLOperators.update_coefficients!(
        L::GenericDiagonalOperator, u, p, t; scale = p
    )
    L.scale = scale
    return nothing
end

@testset "AbstractSciMLOperator generic interface" begin
    L = GenericDiagonalOperator([2.0, 3.0], 1.0)
    v = [4.0, 5.0]
    w = [7.0, 11.0]

    @test size(L) == (2, 2)
    @test issquare(L)
    @test islinear(L)
    @test !isconstant(L)
    @test isconvertible(L)
    @test has_concretization(L)
    @test has_mul(L)
    @test has_mul!(L)
    @test !has_ldiv(L)
    @test !has_ldiv!(L)
    @test iscached(L)
    @test cache_operator(L, v) === L
    @test concretize(L) == Diagonal([2.0, 3.0])

    updated = update_coefficients(L, nothing, 4.0, 0.0)
    @test updated !== L
    @test updated * v == [32.0, 60.0]
    @test L * v == [8.0, 15.0]
    @test L(v, nothing, 4.0, 0.0) == [32.0, 60.0]

    @test update_coefficients!(L, nothing, 5.0, 0.0) === nothing
    @test L * v == [40.0, 75.0]

    L(w, v, nothing, 2.0, 0.0)
    @test w == [16.0, 30.0]

    L(w, v, nothing, 3.0, 0.0, 2.0, 0.5)
    @test w == [56.0, 105.0]
end

mutable struct GenericScalarOperator <: SciMLOperators.AbstractSciMLScalarOperator{Float64}
    value::Float64
end

Base.convert(::Type{Number}, L::GenericScalarOperator) = L.value
SciMLOperators.isconstant(::GenericScalarOperator) = false
SciMLOperators.has_ldiv(L::GenericScalarOperator) = !iszero(L.value)
SciMLOperators.has_ldiv!(L::GenericScalarOperator) = !iszero(L.value)

function SciMLOperators.update_coefficients(
        L::GenericScalarOperator, u, p, t; scale = p
    )
    return GenericScalarOperator(scale)
end

function SciMLOperators.update_coefficients!(
        L::GenericScalarOperator, u, p, t; scale = p
    )
    L.value = scale
    return nothing
end

@testset "AbstractSciMLScalarOperator generic interface" begin
    L = GenericScalarOperator(2.0)
    v = [4.0, 5.0]
    w = [7.0, 11.0]

    @test size(L) == ()
    @test islinear(L)
    @test !isconstant(L)
    @test has_mul(L)
    @test has_mul!(L)
    @test has_ldiv(L)
    @test has_ldiv!(L)
    @test has_concretization(L)
    @test concretize(L) == 2.0
    @test L * v == [8.0, 10.0]
    @test mul!(copy(w), L, v) == [8.0, 10.0]

    updated = update_coefficients(L, nothing, 3.0, 0.0)
    @test updated !== L
    @test updated * v == [12.0, 15.0]
    @test L(v, nothing, 4.0, 0.0) == [16.0, 20.0]

    @test update_coefficients!(L, nothing, 5.0, 0.0) === nothing
    @test L * v == [20.0, 25.0]
    L(w, v, nothing, 3.0, 0.0)
    @test w == [12.0, 15.0]
end
