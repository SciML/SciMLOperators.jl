#
###
# AbstractSciMLScalarOperator interface
###

Base.size(α::AbstractSciMLScalarOperator) = ()
Base.adjoint(α::AbstractSciMLScalarOperator) = conj(α)
Base.transpose(α::AbstractSciMLScalarOperator) = α

has_mul!(::AbstractSciMLScalarOperator) = true
issquare(L::AbstractSciMLScalarOperator) = true
has_adjoint(::AbstractSciMLScalarOperator) = true

Base.:*(α::AbstractSciMLScalarOperator, u::AbstractVecOrMat) = convert(Number, α) * u
Base.:\(α::AbstractSciMLScalarOperator, u::AbstractVecOrMat) = convert(Number, α) \ u

LinearAlgebra.lmul!(α::AbstractSciMLScalarOperator, u::AbstractVecOrMat) = lmul!(convert(Number, α), u)
LinearAlgebra.rmul!(u::AbstractSciMLScalarOperator, α::AbstractVecOrMat) = rmul!(u, convert(Number, α))
LinearAlgebra.ldiv!(α::AbstractSciMLScalarOperator, u::AbstractVecOrMat) = ldiv!(convert(Number, α), u)
function LinearAlgebra.ldiv!(v::AbstractVecOrMat, α::AbstractSciMLScalarOperator, u::AbstractVecOrMat)
    ldiv!(convert(Number, α), u)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, α::AbstractSciMLScalarOperator, u::AbstractVecOrMat)
    x = convert(Number, α)
    mul!(v, x, u)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat,
                            α::AbstractSciMLScalarOperator,
                            u::AbstractVecOrMat,
                            a::AbstractSciMLScalarOperator,
                            b::AbstractSciMLScalarOperator)
    α = convert(Number, α)
    a = convert(Number, a)
    b = convert(Number, b)
    mul!(v, α, u, a, b)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, α::AbstractSciMLScalarOperator, u::AbstractVecOrMat, a, b)
    α = convert(Number, α)
    mul!(v, α, u, a, b)
end

function LinearAlgebra.axpy!(α::AbstractSciMLScalarOperator, x::AbstractVecOrMat, y::AbstractVecOrMat)
    α = convert(Number, α)
    axpy!(α, x, y)
end

function LinearAlgebra.axpby!(α::AbstractSciMLScalarOperator,
                              x::AbstractVecOrMat,
                              β::AbstractSciMLScalarOperator,
                              y::AbstractVecOrMat)
    α = convert(Number, α)
    β = convert(Number, β)
    axpby!(α, x, β, y)
end

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

ScalarOperator(α::AbstractSciMLScalarOperator) = α
ScalarOperator(λ::UniformScaling) = ScalarOperator(λ.λ)

# traits
Base.:+(α::ScalarOperator) = α
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

Base.one(::AbstractSciMLScalarOperator{T}) where{T} = ScalarOperator(one(T))
Base.zero(::AbstractSciMLScalarOperator{T}) where{T} = ScalarOperator(zero(T))

Base.one(::Type{<:AbstractSciMLScalarOperator}) = ScalarOperator(true)
Base.zero(::Type{<:AbstractSciMLScalarOperator}) = ScalarOperator(false)
Base.abs(α::ScalarOperator) = abs(α.val)

Base.iszero(α::ScalarOperator) = iszero(α.val)

getops(α::ScalarOperator) = (α.val,)
isconstant(α::ScalarOperator) = α.update_func == DEFAULT_UPDATE_FUNC
has_ldiv(α::ScalarOperator) = iszero(α.val)
has_ldiv!(α::ScalarOperator) = has_ldiv(α)

update_coefficients!(L::ScalarOperator,u,p,t) = (L.update_func(L.val,u,p,t); nothing)

"""
Lazy addition of Scalar Operators
"""
struct AddedScalarOperator{T,O} <: AbstractSciMLScalarOperator{T}
    ops::O

    function AddedScalarOperator(ops::NTuple{<:Integer,AbstractSciMLScalarOperator})
        T = promote_type(eltype.(ops)...)
        new{T,typeof(ops)}(ops)
    end
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

function Base.convert(::Type{Number}, α::AddedScalarOperator{T}) where{T}
    sum( op -> convert(Number, op), α.ops; init=zero(T))
end

function cache_internals(α::AddedScalarOperator, u::AbstractVecOrMat)
    for i=1:length(α.ops)
        @set! α.ops[i] = cache_operator(α.ops[i], u)
    end
    α
end

"""
Lazy composition of Scalar Operators
"""
struct ComposedScalarOperator{T,O} <: AbstractSciMLScalarOperator{T}
    ops::O

    function ComposedScalarOperator(ops::NTuple{<:Integer,AbstractSciMLScalarOperator})
        T = promote_type(eltype.(ops)...)
        new{T,typeof(ops)}(ops)
    end
end

# constructor
function ComposedScalarOperator(ops::AbstractSciMLScalarOperator...)
    ComposedScalarOperator(ops)
end

Base.:∘(ops::AbstractSciMLScalarOperator...) = ComposedScalarOperator(ops...)
Base.:∘(A::ComposedScalarOperator, B::ComposedScalarOperator) = ComposedScalarOperator(A.ops..., B.ops...)
Base.:∘(A::AbstractSciMLScalarOperator, B::ComposedScalarOperator) = ComposedScalarOperator(A, B.ops...)
Base.:∘(A::ComposedScalarOperator, B::AbstractSciMLScalarOperator) = ComposedScalarOperator(A.ops..., B)

Base.:*(ops::AbstractSciMLScalarOperator...) = ComposedScalarOperator(ops...)
Base.:*(A::AbstractSciMLScalarOperator, B::AbstractSciMLScalarOperator) = ∘(A, B)
Base.:*(A::ComposedScalarOperator, B::AbstractSciMLScalarOperator) = ∘(A.ops[1:end-1]..., A.ops[end] * B)
Base.:*(A::AbstractSciMLScalarOperator, B::ComposedScalarOperator) = ∘(A * B.ops[1], B.ops[2:end]...)
Base.:*(A::ComposedScalarOperator, B::ComposedScalarOperator) = ComposedScalarOperator(A.ops..., B.ops...)

function Base.convert(::Type{Number}, α::ComposedScalarOperator{T}) where{T}
    iszero(α) && return zero(T)
    prod( op -> convert(Number, op), α.ops; init=one(T))
end

getops(α::ComposedScalarOperator) = α.ops

function cache_internals(α::ComposedScalarOperator, u::AbstractVecOrMat)
    for i=1:length(α.ops)
        @set! α.ops[i] = cache_operator(α.ops[i], u)
    end
    α
end
#
