#
struct ScaledDiffEqOperator{T,
                            λType<:Number,
                            LType<:AbstractDiffEqOperator,
                           } <: AbstractDiffEqOperator{T}
    λ::λType
    L::LType

    function ScaledDiffEqOperator(λ, L::AbstractDiffEqOperator{Tl}) where{Tl}
        T = promote_type(eltype(λ), Tl)
        new{T,typeof(λ),typeof(op)}(λ, L)
    end
end

function update_coefficients!(L::AffineDiffEqOperator, u, p, t)
    update_coefficients!(L.L, u, p, t)
    update_coefficients!(L.λ, u, p, t)
    nothing
end

# constructor
Base.:*(λ::Number, L::AbstractDiffEqOperator) = ScaledDiffEqOperator(λ, L)
Base.:*(L::AbstractDiffEqOperator, λ::Number) = ScaledDiffEqOperator(λ, L)

Base.convert(::Type{AbstractMatrix}, L::DiffEqScaledOperator) = λ * convert(AbstractMatrix, L.L)

# traits
Base.size(L::ScaledDiffEqOperator) = size(L.L)
Base.adjoint(L::ScaledDiffEqOperator) = ScaledDiffEqOperator(L.λ', L.op')

issquare(L::ScaledDiffEqOperator) = issquare(L.L)
has_adjoint(L::ScaledDiffEqOperator) = has_adjoint(L.L)
has_ldiv(L::ScaledDiffEqOperator) = has_ldiv(L.L) & !iszero(L.λ)
has_ldiv!(L::ScaledDiffEqOperator) = has_ldiv!(L.L) & !iszero(L.λ)

# operator application
for op in (
           :*, :/, :\,
          )
    @eval Base.$op(L::DiffEqScaledOperator, x::AbstractVector) = $op(L.λ, $op(L.L, x))
#   @eval Base.$op(x::AbstractVector, L::DiffEqScaledOperator) = $op($op(x, L.L), L.λ)
end

function LinearAlgebra.mul!(v::AbstractVector, L::DiffEqScaledOperator, u::AbstractVector)
    mul!(v, L.L, u)
    lmul!(L.λ, v)
end

"""
Lazy affine operator combinations αA + βB

v = (αAu + βB)u
"""
struct AffineDiffEqOperator{T,
                            Ta<:AbstractDiffEqOperator,
                            Tb<:AbstractDiffEqOperator,
                            Tα,
                            Tβ,
                            Tc,
                           } <: AbstractDiffEqOperator{T}
    A::Ta
    B::Tb
    α::Tα
    β::Tβ

    cache::Tc
    isunset::Bool

    function AffineDiffEqOperator(A::AbstractDiffEqOperator{Ta},
                                  B::AbstractDiffEqOperator{Tb}, α, β;
                                  cache = nothing,
                                  isunset = cache === nothing,
                                 ) where{Ta,Tb}
        @assert size(A) == size(B)
        T = promote_type(Ta,Tb, eltype(α), eltype(β))

        new{T,
            typeof(A),
            typeof(B),
            typeof(α),
            typeof(β),
            typeof(cache),
            typeof(update_func)
           }(
             A, B, α, β, cache, isunset, update_func,
            )
    end
end

function update_coefficients!(L::AffineDiffEqOperator, u, p, t)
    update_coefficients!(L.A, u, p, t)
    update_coefficients!(L.B, u, p, t)
    update_coefficients!(L.α, u, p, t)
    update_coefficients!(L.β, u, p, t)
    nothing
end

# traits
Base.size(A::AffineDiffEqOperator) = size(A.A)
function Base.adjoint(A::AffineDiffEqOperator)
    if issquare(A) & !(A.isunset)
        AffineDiffEqOperator(A.A',A.B',A.α', A.β', A.cache, A.isunset)
    else
        AffineDiffEqOperator(A.A',A.B',A.α', A.β')
    end
end

issquare(A::AffineDiffEqOperator) = issquare(A.A)

function init_cache(A::AffineDiffEqOperator{<:Number}, u::AbstractField{<:Number})
    cache = A.B * u
end

function Base.:*(A::AffineDiffEqOperator{<:Number}, u::AbstractField{<:Number})
    @unpack A, B, α, β = A
    if iszero(α) | (A isa DiffEqNullOperator)
        β * (B * u)
    elseif iszero(β) | (B isa DiffEqNullOperator)
        α * (A * u)
    else
        α * (A * u) + β * (B * u)
    end
end

function LinearAlgebra.mul!(v::AbstractField{<:Number}, Op::AffineDiffEqOperator{<:Number}, u::AbstractField{<:Number})
    @unpack A, B, α, β, cache, isunset = Op

    if iszero(α) | (A isa DiffEqNullOperator)
        mul!(v, B, u)
        lmul!(β, v)
        return v
    elseif iszero(β) | (B isa DiffEqNullOperator)
        mul!(v, A, u)
        lmul!(α, v)
        return v
    end

    mul!(v, A, u)
    lmul!(α, v)

    if isunset
        cache = init_cache(Op, u)
        Op = set_cache(Op, cache)
    end

    mul!(cache, B, u)
    lmul!(β, cache)
    axpy!(true, cache, v)
end

function Base.:+(A::AbstractOperator{<:Number}, B::AbstractOperator{<:Number})
    AffineDiffEqOperator(A, B, true, true)
end

function Base.:-(A::AbstractOperator{<:Number}, B::AbstractOperator{<:Number})
    AffineDiffEqOperator(A, B, true, -true)
end

function Base.:+(A::AbstractOperator{<:Number}, λ::Number)
    N = size(A, 1)
    Id = DiffEqIdentity{N}()
    AffineDiffEqOperator(A, Id, true, λ)
end

function Base.:+(λ::Number, A::AbstractOperator{<:Number})
    N = size(A, 1)
    Id = DiffEqIdentity{N}()
    AffineDiffEqOperator(A, Id, true, λ) # TODO: what if A isn't square
end

function Base.:-(A::AbstractOperator{<:Number}, λ::Number)
    N = size(A, 1)
    Id = DiffEqIdentity{N}()
    AffineDiffEqOperator(A, Id, -true, λ)
end

function Base.:-(λ::Number, A::AbstractOperator{<:Number})
    N = size(A, 1)
    Id = DiffEqIdentity{N}()
    AffineDiffEqOperator(Id, A, λ, -true)
end

function Base.:-(A::AbstractOperator{<:Number})
    N = size(A, 1)
    Z = DiffEqNullOperator{N}()
    AffineDiffEqOperator(A, -true, false, Z)
end

function Base.:*(A::AbstractOperator{<:Number}, λ::Number)
    N = size(A, 1)
    Z = DiffEqNullOperator{N}()
    AffineDiffEqOperator(A, Z, λ, false)
end

function Base.:*(λ::Number, A::AbstractOperator{<:Number})
    N = size(A, 1)
    Z = DiffEqNullOperator{N}()
    AffineDiffEqOperator(A, Z, λ, false)
end

function Base.:/(A::AbstractOperator{<:Number}, λ::Number)
    N = size(A, 1)
    Z = DiffEqNullOperator{N}()
    AffineDiffEqOperator(A, Z, -true, λ)
end

