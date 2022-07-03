#
###
# AbstractSciMLScalarOperator interface
###

ScalingNumberTypes = (
                      :AbstractSciMLScalarOperator,
                      :Number,
                      :UniformScaling,
                     )

Base.size(α::AbstractSciMLScalarOperator) = ()
Base.adjoint(α::AbstractSciMLScalarOperator) = conj(α)
Base.transpose(α::AbstractSciMLScalarOperator) = α

has_mul!(::AbstractSciMLScalarOperator) = true
issquare(L::AbstractSciMLScalarOperator) = true
has_adjoint(::AbstractSciMLScalarOperator) = true

Base.:*(α::AbstractSciMLScalarOperator, u::AbstractVecOrMat) = convert(Number, α) * u
Base.:\(α::AbstractSciMLScalarOperator, u::AbstractVecOrMat) = convert(Number, α) \ u

LinearAlgebra.rmul!(u::AbstractVecOrMat, α::AbstractSciMLScalarOperator) = rmul!(u, convert(Number, α))
LinearAlgebra.lmul!(α::AbstractSciMLScalarOperator, u::AbstractVecOrMat) = lmul!(convert(Number, α), u)
LinearAlgebra.ldiv!(α::AbstractSciMLScalarOperator, u::AbstractVecOrMat) = ldiv!(convert(Number, α), u)
function LinearAlgebra.ldiv!(v::AbstractVecOrMat, α::AbstractSciMLScalarOperator, u::AbstractVecOrMat)
    ldiv!(v, convert(Number, α), u)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, α::AbstractSciMLScalarOperator, u::AbstractVecOrMat)
    x = convert(Number, α)
    mul!(v, x, u)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat,
                            α::AbstractSciMLScalarOperator,
                            u::AbstractVecOrMat,
                            a::Union{Number,AbstractSciMLScalarOperator},
                            b::Union{Number,AbstractSciMLScalarOperator})
    α = convert(Number, α)
    a = convert(Number, a)
    b = convert(Number, b)
    mul!(v, α, u, a, b)
end

function LinearAlgebra.axpy!(α::AbstractSciMLScalarOperator, x::AbstractVecOrMat, y::AbstractVecOrMat)
    α = convert(Number, α)
    axpy!(α, x, y)
end

function LinearAlgebra.axpby!(α::Union{Number,AbstractSciMLScalarOperator},
                              x::AbstractVecOrMat,
                              β::Union{Number,AbstractSciMLScalarOperator},
                              y::AbstractVecOrMat)
    α = convert(Number, α)
    β = convert(Number, β)
    axpby!(α, x, β, y)
end

Base.:+(α::AbstractSciMLScalarOperator) = α

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
function Base.:-(α::ScalarOperator) # TODO - test
    # can also just form ScalarOperator(-1) * α
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

    function AddedScalarOperator(ops::NTuple{N,AbstractSciMLScalarOperator}) where{N}
        T = promote_type(eltype.(ops)...)
        new{T,typeof(ops)}(ops)
    end
end

# constructors
function AddedScalarOperator(ops::AbstractSciMLScalarOperator...)
    AddedScalarOperator(ops)
end

Base.:+(ops::AbstractSciMLScalarOperator...) = AddedScalarOperator(ops...)
Base.:+(A::AddedScalarOperator, B::AddedScalarOperator) = AddedScalarOperator(A.ops..., B.ops...)
Base.:+(A::AbstractSciMLScalarOperator, B::AddedScalarOperator) = AddedScalarOperator(A, B.ops...)
Base.:+(A::AddedScalarOperator, B::AbstractSciMLScalarOperator) = AddedScalarOperator(A.ops..., B)

for op in (
           :-, :+,
          )
    for T in ScalingNumberTypes
        @eval Base.$op(α::AbstractSciMLScalarOperator, x::$T) = AddedScalarOperator(α, ScalarOperator($op(x)))
        @eval Base.$op(x::$T, α::AbstractSciMLScalarOperator) = AddedScalarOperator(ScalarOperator(x), $op(α))
    end
end

function Base.convert(::Type{Number}, α::AddedScalarOperator{T}) where{T}
    sum(op -> convert(Number, op), α.ops; init=zero(T))
end

Base.conj(L::AddedScalarOperator) = AddedScalarOperator(conj.(L.ops))

getops(α::AddedScalarOperator) = α.ops

"""
Lazy composition of Scalar Operators
"""
struct ComposedScalarOperator{T,O} <: AbstractSciMLScalarOperator{T}
    ops::O

    function ComposedScalarOperator(ops::NTuple{N,AbstractSciMLScalarOperator}) where{N}
        T = promote_type(eltype.(ops)...)
        new{T,typeof(ops)}(ops)
    end
end

# constructor
function ComposedScalarOperator(ops::AbstractSciMLScalarOperator...)
    ComposedScalarOperator(ops)
end

for op in (
           :*, :∘,
          )
    @eval Base.$op(ops::AbstractSciMLScalarOperator...) = ComposedScalarOperator(ops...)
    @eval Base.$op(A::ComposedScalarOperator, B::ComposedScalarOperator) = ComposedScalarOperator(A.ops..., B.ops...)
    @eval Base.$op(A::AbstractSciMLScalarOperator, B::ComposedScalarOperator) = ComposedScalarOperator(A, B.ops...)
    @eval Base.$op(A::ComposedScalarOperator, B::AbstractSciMLScalarOperator) = ComposedScalarOperator(A.ops..., B)

    for T in ScalingNumberTypes
        @eval Base.$op(α::ComposedScalarOperator, x::$T) = ComposedScalarOperator(α, ScalarOperator($op(x)))
        @eval Base.$op(x::$T, α::ComposedScalarOperator) = ComposedScalarOperator(ScalarOperator(x), $op(α))
    end
end

function Base.convert(::Type{Number}, α::ComposedScalarOperator{T}) where{T}
    iszero(α) && return zero(T)
    prod( op -> convert(Number, op), α.ops; init=one(T))
end

Base.conj(L::ComposedScalarOperator) = ComposedScalarOperator(conj.(L.ops))
Base.:-(α::AbstractSciMLScalarOperator{T}) where{T} = (-one(T)) * α

getops(α::ComposedScalarOperator) = α.ops
#
