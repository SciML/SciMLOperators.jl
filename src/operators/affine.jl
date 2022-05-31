#

struct ScaledDiffEqOperator
    λ
    op
end
"""
Lazy affine operator combinations αA + βB

v = (αAu + βB)u
"""
struct AffineDiffEqOperator{T,
                            Ta <: AbstractDiffEqOperator,
                            Tb <: AbstractDiffEqOperator,
                            Tα,Tβ,Tc
                           } <: AbstractDiffEqOperator{T}
    A::Ta
    B::Tb
    α::Tα
    β::Tβ

    cache::Tc
    isunset::Bool

    function AffineDiffEqOperator(A::AbstractOperator{Ta}, B::AbstractOperator{Tb}, α, β,
                      cache = nothing, isunset = cache === nothing) where{Ta,Tb}
        @assert size(A) == size(B)
        #TODO - update coefficients of A, B separately
        T = promote_type(Ta,Tb)
        new{T,typeof(A),typeof(B),typeof(α),typeof(β),typeof(cache)}(A, B, α, β, cache, isunset)
    end
end

# traits
Base.size(A::AffineDiffEqOperator) = size(A.A)
function Base.adjoint(A::AffineDiffEqOperator)
    if issquare(A)
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

