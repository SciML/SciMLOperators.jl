#
Base.size(α::AbstractScalarOperator) = ()
Base.adjoint(α::AbstractScalarOperator) = conj(α)
Base.transpose(α::AbstractScalarOperator) = α

has_mul!(::AbstractScalarOperator) = true
issquare(L::AbstractScalarOperator) = true
has_adjoint(::AbstractScalarOperator) = true

## AbstractScalarOperators can also be `a, b` or `α, β` in axpby!, or mul!
## write methods for that

function LinearAlgebra.mul!(v::AbstractVecOrMat, α::AbstractScalarOperator, u::AbstractVecOrMat)
    copy!(v, u)
    lmul!(α, v)
end

#function LinearAlgebra.mul!(v::AbstractVecOrMat, α::AbstractScalarOperator, u::AbstractVecOrMat, a::AbstractScalarOperator, b::AbstractScalarOperator)
#    axpby!(a*α, u, β, v)
#end

#function LinearAlgebra.axpy!(α::AbstractScalarOperator, x::AbstractVecOrMat, y::AbstractVecOrMat) = axpy!(α.val, x, y)
#end
#function LinearAlgebra.axpby!(α::AbstractScalarOperator, x::AbstractVecOrMat, β::AbstractScalarOperator, y::AbstractVecOrMat) = axpby!(α.val, x, β.val, y)
#end

"""
    ScalarOperator(val[; update_func])

    (α::ScalarOperator)(a::Number) = α * a

Represents a time-dependent scalar/scaling operator. The update function
is called by `update_coefficients!` and is assumed to have the following
signature:

    update_func(oldval,u,p,t) -> newval
"""
mutable struct ScalarOperator{T<:Number,F} <: AbstractSciMLScalarOperator{T}
    val::T
    update_func::F

    ScalarOperator(val::T; update_func=DEFAULT_UPDATE_FUNC) where{T} =
        new{T,typeof(update_func)}(val, update_func)
end

# constructors
Base.convert(::Type{Number}, α::ScalarOperator) = α.val
Base.convert(::Type{ScalarOperator}, α::Number) = ScalarOperator(α)

ScalarOperator(α::AbstractScalarOperator) = α
ScalarOperator(λ::UniformScaling) = ScalarOperator(λ.λ)

# traits
function Base.:-(α::ScalarOperator) # TODO - test
    val = -α.val
    update_func = (oldval,u,p,t) -> -α.update_func(-oldval,u,p,t)
    ScalarOperator(val; update_func=update_func)
end

function Base.conj(α::ScalarOperator) # TODO - test
    val = α.val'
    update_func = (oldval,u,p,t) -> α.update_func(oldval',u,p,t)'
    ScalarOperator(val; update_func=update_func)
end

Base.one(::AbstractScalarOperator{T}) where{T} = ScalarOperator(one(T))
Base.zero(::AbstractScalarOperator{T}) where{T} = ScalarOperator(zero(T))

Base.one(::Type{<:AbstractSciMLOperator}) = ScalarOperator(true)
Base.zero(::Type{<:AbstractSciMLOperator}) = ScalarOperator(false)
Base.abs(α::ScalarOperator) = abs(α.val)

Base.iszero(α::ScalarOperator) = iszero(α.val)

getops(α::ScalarOperator) = (α.val,)
isconstant(α::ScalarOperator) = α.update_func == DEFAULT_UPDATE_FUNC
has_ldiv(α::ScalarOperator) = iszero(α.val)
has_ldiv!(α::ScalarOperator) = has_ldiv(α)

update_coefficients!(L::ScalarOperator,u,p,t) = (L.update_func(L.val,u,p,t); nothing)

# operator application
Base.:*(α::ScalarOperator, u::Union{Number,AbstractVecOrMat}) = α.val * u
Base.:\(α::ScalarOperator, u::Union{Number,AbstractVecOrMat}) = α.val \ u

LinearAlgebra.lmul!(α::ScalarOperator, u::AbstractVecOrMat) = lmul!(α.val, u)
LinearAlgebra.rmul!(u::AbstractVecOrMat, α::ScalarOperator) = rmul!(u, α.val)
LinearAlgebra.mul!(v::AbstractVecOrMat, α::ScalarOperator, u::AbstractVecOrMat) = mul!(v, α.val, u)
LinearAlgebra.mul!(v::AbstractVecOrMat, α::ScalarOperator, u::AbstractVecOrMat, a, b) = mul!(v, α.val, u, a, b)
LinearAlgebra.axpy!(α::ScalarOperator, x::AbstractVecOrMat, y::AbstractVecOrMat) = axpy!(α.val, x, y)
LinearAlgebra.axpby!(α::ScalarOperator, x::AbstractVecOrMat, β::ScalarOperator, y::AbstractVecOrMat) = axpby!(α.val, x, β.val, y)

LinearAlgebra.ldiv!(v::AbstractVecOrMat, α::ScalarOperator, u::AbstractVecOrMat) = ldiv!(v, α.val, u)
LinearAlgebra.ldiv!(α::ScalarOperator, u::AbstractVecOrMat) = ldiv!(α.val, u)

"""
Lazy addition of Scalar Operators
"""
struct AddedScalarOperator{T} <: AbstractSciMLScalarOperator{T}
    ops
end

for op in (
           :-, :+,
          )
    @eval Base.$op(α::ScalarOperator, x::Number) = AddedScalarOperator(α.val, ScalarOperator(x))
    @eval Base.$op(x::Number, α::ScalarOperator) = AddedScalarOperator(ScalarOperator(x), α.val)
    @eval Base.$op(x::ScalarOperator, y::ScalarOperator) = AddedScalarOperator(x, y)
end

# overload +, -

getops(α::AddedScalarOperator) = α.ops

# operator application

function LinearAlgebra.lmul!(α::ComposedScalarOperator, u::AbstractVecOrMat)
    for i=1:length(α.ops)
        lmul!(α.ops[i], u)
    end
end

function LinearAlgebra.rmul!(u::AbstractVecOrMat, α::ComposedScalarOperator)
    for i=1:length(α.ops)
        rmul!(u, α.ops[i])
    end
end

"""
Lazy composition of Scalar Operators
"""
struct ComposedScalarOperator{T} <: AbstractSciMLScalarOperator{T}
    ops

    function ComposedScalarOperator(ops::NTuple{<:Integer,AbstractSciMLScalarOperator})
        T = promote_type(eltype.(ops)...)
        new{
            T,
            typeof(ops)
           }(
             ops, cache, isset,
            )
    end
end

function ComposedScalarOperator(ops::AbstractSciMLScalarOperator...)
    ComposedScalarOperator(ops)
end

getops(α::ComposedScalarOperator) = α.ops
Base.convert(::Type{Number}, α::ComposedScalarOperator)

Base.*(x::ScalarOperator, y::ScalarOperator) = ComposedScalarOperator(x.val, y.val)
Base.∘(x::ScalarOperator, y::ScalarOperator) = ComposedScalarOperator(x.val, y.val)

# operator application
Base.*(x::ScalarOperator, y::ScalarOperator) = ComposedScalarOperator(x.val, y.val)

function LinearAlgebra.lmul!(α::ComposedScalarOperator, u::AbstractVecOrMat)
    for i=1:length(α.ops)
        lmul!(α.ops[i], u)
    end
end

function LinearAlgebra.rmul!(u::AbstractVecOrMat, α::ComposedScalarOperator)
    for i=1:length(α.ops)
        rmul!(u, α.ops[i])
    end
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, α::ComposedScalarOperator, u::AbstractVecOrMat)
    ldiv!(v, α.ops[1], u)
    for i=1:length(α.ops)
        ldiv!(α.ops[i], v)
    end
end

function LinearAlgebra.ldiv!(α::ComposedScalarOperator, u::AbstractVecOrMat)
    for i=1:length(α.ops)
        ldiv!(α.ops[i], u)
    end
end
#
