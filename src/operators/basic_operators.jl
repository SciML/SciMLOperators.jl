"""
$(TYPEDEF)
"""
struct DiffEqIdentity{N} <: AbstractDiffEqLinearOperator{Bool} end

# constructors
DiffEqIdentity(u::AbstractVector) = DiffEqIdentity{length(u)}()

function Base.one(A::AbstractDiffEqOperator)
    @assert issquare(A)
    N = size(A, 1)
    DiffEqIdentity{N}()
end

Base.convert(::Type{AbstractMatrix}, ::DiffEqIdentity{N}) where{N} = Diagonal(ones(Bool, N))

# traits
Base.size(::DiffEqIdentity{N}) where{N} = (N, N)
Base.adjoint(A::DiffEqIdentity) = A
LinearAlgebra.opnorm(::DiffEqIdentity{N}, p::Real=2) where{N} = true
for pred in (
             :isreal, :issymmetric, :ishermitian, :isposdef,
            )
    @eval LinearAlgebra.$pred(::DiffEqIdentity) = true
end

issquare(::DiffEqIdentity) = true
isconstant(::DiffEqIdentity) = true
has_adjoint(::DiffEqIdentity) = true
has_mul!(::DiffEqIdentity) = true
has_ldiv(::DiffEqIdentity) = true
has_ldiv!(::DiffEqIdentity) = true

# opeator application
for op in (
           :*, :\,
          )
    @eval Base.$op(::DiffEqIdentity{N}, x::AbstractVector) where{N} = (@assert length(x) == N; copy(x))
    # left multiplication with vector doens't make sense
#   @eval Base.$op(x::AbstractVector, ::DiffEqIdentity{N}) where{N} = (@assert length(x) == N; copy(x))
end

function LinearAlgebra.mul!(v::AbstractVector, ::DiffEqIdentity{N}, u::AbstractVector) where{N}
    @assert length(u) == N
    copy!(v, u)
end

function LinearAlgebra.ldiv!(v::AbstractVector, ::DiffEqIdentity{N}, u::AbstractArray) where{N}
    @assert length(u) == N
    copy!(v, u)
end

# operator fusion, composition
for op in (:*, :∘, :/, :\)
    @eval Base.$op(::DiffEqIdentity{N}, A::AbstractSciMLOperator) where{N} = (@assert size(A, 1) == N; DiffEqIdentity{N}())
    @eval Base.$op(A::AbstractSciMLOperator, ::DiffEqIdentity{N}) where{N} = (@assert size(A, 2) == N; DiffEqIdentity{N}())
end

"""
$(TYPEDEF)
"""
struct DiffEqNullOperator{N} <: AbstractDiffEqLinearOperator{Bool} end

# constructors
DiffEqNullOperator(u::AbstractVector) = DiffEqNullOperator{length(u)}()

function Base.zero(A::AbstractDiffEqOperator)
    @assert issquare(A)
    N = size(A, 1)
    DiffEqNullOperator{N}()
end

Base.convert(::Type{AbstractMatrix}, ::DiffEqNullOperator{N}) where{N} = Diagonal(zeros(Bool, N))

# traits
Base.size(::DiffEqNullOperator{N}) where{N} = (N, N)
Base.adjoint(A::DiffEqNullOperator) = A
LinearAlgebra.opnorm(::DiffEqNullOperator{N}, p::Real=2) where{N} = false
for pred in (
             :isreal, :issymmetric, :ishermitian,
            )
    @eval LinearAlgebra.$pred(::DiffEqNullOperator) = true
end
LinearAlgebra.isposdef(::DiffEqNullOperator) = false

issquare(::DiffEqNullOperator) = true
isconstant(::DiffEqNullOperator) = true
has_adjoint(::DiffEqNullOperator) = true
has_mul!(::DiffEqNullOperator) = true

# opeator application
Base.:*(::DiffEqNullOperator{N}, x::AbstractVector) where{N} = (@assert length(x) == N; zero(x))
Base.:*(x::AbstractVector, ::DiffEqNullOperator{N}) where{N} = (@assert length(x) == N; zero(x))

function LinearAlgebra.mul!(v::AbstractVector, ::DiffEqNullOperator{N}, u::AbstractArray) where{N}
    @assert length(u) == length(v) == N
    lmul!(false, v)
end

# operator fusion, composition
for op in (:*, :∘)
    @eval Base.$op(::DiffEqNullOperator{N}, A::AbstractSciMLOperator) where{N} = (@assert size(A, 1) == N; DiffEqNullOperator{N}())
    @eval Base.$op(A::AbstractSciMLOperator, ::DiffEqNullOperator{N}) where{N} = (@assert size(A, 2) == N; DiffEqNullOperator{N}())
end

"""
    DiffEqScalar(val[; update_func])

    (α::DiffEqScalar)(a::Number) = α * a

Represents a time-dependent scalar/scaling operator. The update function
is called by `update_coefficients!` and is assumed to have the following
signature:

    update_func(oldval,u,p,t) -> newval
"""
mutable struct DiffEqScalar{T<:Number,F} #<: AbstractDiffEqLinearOperator{T}
  val::T
  update_func::F
  DiffEqScalar(val::T; update_func=DEFAULT_UPDATE_FUNC) where{T} =
    new{T,typeof(update_func)}(val, update_func)
end

update_coefficients!(α::DiffEqScalar,u,p,t) = (α.val = α.update_func(α.val,u,p,t); α)

# constructors
Base.convert(::Type{Number}, α::DiffEqScalar) = α.val
Base.convert(::Type{DiffEqScalar}, α::Number) = DiffEqScalar(α)

# traits
Base.size(::DiffEqScalar) = ()
function Base.adjoint(α::DiffEqScalar) # TODO - test
    val = α.val'
    update_func =  (oldval,u,p,t) -> α.update_func(oldval',u,p,t)'
    DiffEqScalar(val; update_func=update_func)
end
isconstant(α::DiffEqScalar) = α.update_func == DEFAULT_UPDATE_FUNC
has_adjoint(::DiffEqScalar) = true
has_mul(::DiffEqScalar) = true
has_ldiv(α::DiffEqScalar) = iszero(α.val)

for op in (:*, :/, :\)
    for T in (
              :Number,
              :AbstractArray, # TODO - to act on AbstractArrays, it needs a notion of size?
             )
        @eval Base.$op(α::DiffEqScalar, x::$T) = $op(α.val, x)
        @eval Base.$op(x::$T, α::DiffEqScalar) = $op(x, α.val)
    end
    # TODO should result be Number or DiffEqScalar
    @eval Base.$op(x::DiffEqScalar, y::DiffEqScalar) = $op(x.val, y.val)
    #@eval function Base.$op(x::DiffEqScalar, y::DiffEqScalar) # TODO - test
    #    val = $op(x.val, y.val)
    #    update_func = (oldval,u,p,t) -> x.update_func(oldval,u,p,t) * y.update_func(oldval,u,p,t)
    #    DiffEqScalar(val; update_func=update_func)
    #end
end

for op in (:-, :+)
    @eval Base.$op(α::DiffEqScalar, x::Number) = $op(α.val, x)
    @eval Base.$op(x::Number, α::DiffEqScalar) = $op(x, α.val)
    # TODO - should result be Number or DiffEqScalar?
    @eval Base.$op(x::DiffEqScalar, y::DiffEqScalar) = $op(x.val, y.val)
end

LinearAlgebra.lmul!(α::DiffEqScalar, B::AbstractArray) = lmul!(α.val, B)
LinearAlgebra.rmul!(B::AbstractArray, α::DiffEqScalar) = rmul!(B, α.val)
LinearAlgebra.mul!(Y::AbstractArray, α::DiffEqScalar, B::AbstractArray) = mul!(Y, α.val, B)
LinearAlgebra.axpy!(α::DiffEqScalar, X::AbstractArray, Y::AbstractArray) = axpy!(α.val, X, Y)
Base.abs(α::DiffEqScalar) = abs(α.val)

"""
    DiffEqArrayOperator(A[; update_func])

Represents a time-dependent linear operator given by an AbstractMatrix. The
update function is called by `update_coefficients!` and is assumed to have
the following signature:

    update_func(A::AbstractMatrix,u,p,t) -> [modifies A]
"""
struct DiffEqArrayOperator{T,AType<:AbstractMatrix{T},F} <: AbstractDiffEqLinearOperator{T}
  A::AType
  update_func::F
  DiffEqArrayOperator(A::AType; update_func=DEFAULT_UPDATE_FUNC) where{AType} =
    new{eltype(A),AType,typeof(update_func)}(A, update_func)
end

# constructors
Base.similar(L::DiffEqArrayOperator, ::Type{T}, dims::Dims) where{T} = similar(L.A, T, dims)

# traits
@forward DiffEqArrayOperator.A (
                                issquare, SciMLBase.has_ldiv, SciMLBase.has_ldiv!
                               )
Base.size(A::DiffEqArrayOperator) = size(A.A)
Base.adjoint(L::DiffEqArrayOperator) = DiffEqArrayOperator(L.A'; update_func=(A,u,p,t)->L.update_func(L.A,u,p,t)')

has_adjoint(A::DiffEqArrayOperator) = has_adjoint(A.A)
update_coefficients!(L::DiffEqArrayOperator,u,p,t) = (L.update_func(L.A,u,p,t); L)
isconstant(L::DiffEqArrayOperator) = L.update_func == DEFAULT_UPDATE_FUNC

# propagate_inbounds here for the getindex fallback
Base.@propagate_inbounds Base.convert(::Type{AbstractMatrix}, L::DiffEqArrayOperator) = L.A
Base.@propagate_inbounds Base.setindex!(L::DiffEqArrayOperator, v, i::Int) = (L.A[i] = v)
Base.@propagate_inbounds Base.setindex!(L::DiffEqArrayOperator, v, I::Vararg{Int, N}) where{N} = (L.A[I...] = v)

Base.eachcol(L::DiffEqArrayOperator) = eachcol(L.A)
Base.eachrow(L::DiffEqArrayOperator) = eachrow(L.A)
Base.length(L::DiffEqArrayOperator) = length(L.A)
Base.iterate(L::DiffEqArrayOperator,args...) = iterate(L.A,args...)
Base.axes(L::DiffEqArrayOperator) = axes(L.A)
Base.eachindex(L::DiffEqArrayOperator) = eachindex(L.A)
Base.IndexStyle(::Type{<:DiffEqArrayOperator{T,AType}}) where{T,AType} = Base.IndexStyle(AType)
Base.copyto!(L::DiffEqArrayOperator, rhs) = (copyto!(L.A, rhs); L)
Base.copyto!(L::DiffEqArrayOperator, rhs::Base.Broadcast.Broadcasted{<:StaticArrays.StaticArrayStyle}) = (copyto!(L.A, rhs); L)
Base.Broadcast.broadcastable(L::DiffEqArrayOperator) = L
Base.ndims(::Type{<:DiffEqArrayOperator{T,AType}}) where{T,AType} = ndims(AType)
ArrayInterfaceCore.issingular(L::DiffEqArrayOperator) = ArrayInterfaceCore.issingular(L.A)
Base.copy(L::DiffEqArrayOperator) = DiffEqArrayOperator(copy(L.A);update_func=L.update_func)

# operator application
Base.:*(L::DiffEqArrayOperator, u::AbstractVector) = L.A * u
LinearAlgebra.mul!(v::AbstractVector, L::DiffEqArrayOperator, u::AbstractVector) = mul!(v, L.A, u)

# operator fusion, composition
function Base.:*(A::DiffEqArrayOperator, B::DiffEqArrayOperator)
    M = A.A * B.A
    update_func = (M,u,p,t) -> A.update_func(M,u,p,t) * B.update_func(M,u,p,t) #TODO
    DiffEqArrayOperator(M; update_func=update_func)
end

for op in (
           :*, :/, :\,
          )
    @eval function Base.$op(L::DiffEqArrayOperator, x::Number)
        M = $op(L.A, x)
        update_func = L.update_func #TODO fix
        DiffEqArrayOperator(M; update_func=update_func)
    end
    @eval function Base.$op(x::Number, L::DiffEqArrayOperator)
        M = $op(L.A, x)
        update_func = L.update_func #TODO fix
        DiffEqArrayOperator(M; update_func=update_func)
    end
end

"""
    FactorizedDiffEqArrayOperator(F)

Like DiffEqArrayOperator, but stores a Factorization instead.

Supports left division and `ldiv!` when applied to an array.
"""
struct FactorizedDiffEqArrayOperator{T<:Number,FType<:Union{
                                                            Factorization{T},
                                                            Diagonal{T},
                                                            Bidiagonal{T},
                                                            Adjoint{T,<:Factorization{T}},
                                                           }
                                    } <: AbstractDiffEqLinearOperator{T}
    F::FType
end

# constructor
function LinearAlgebra.factorize(L::AbstractDiffEqLinearOperator)
    fact = factorize(convert(AbstractMatrix, L))
    FactorizedDiffEqArrayOperator(fact)
end

for fact in (
             :lu, :lu!,
             :qr, :qr!,
             :cholesky, :cholesky!,
             :ldlt, :ldlt!,
             :bunchkaufman, :bunchkaufman!,
             :lq, :lq!,
             :svd, :svd!,
            )
    @eval LinearAlgebra.$fact(L::AbstractDiffEqLinearOperator, args...) =
        FactorizedDiffEqArrayOperator($fact(convert(AbstractMatrix, L), args...))
    @eval LinearAlgebra.$fact(L::AbstractDiffEqLinearOperator; kwargs...) =
        FactorizedDiffEqArrayOperator($fact(convert(AbstractMatrix, L); kwargs...))
end

function Base.convert(::Type{AbstractMatrix}, L::FactorizedDiffEqArrayOperator)
    if L.F isa Adjoint
        convert(AbstractMatrix,L.F')'
    else
        convert(AbstractMatrix, L.F)
    end
end

function Base.Matrix(L::FactorizedDiffEqArrayOperator)
    if L.F isa Adjoint
        Matrix(L.F')'
    else
        Matrix(L.F)
    end
end

# traits
Base.size(L::FactorizedDiffEqArrayOperator, args...) = size(L.F, args...)
Base.adjoint(L::FactorizedDiffEqArrayOperator) = FactorizedDiffEqArrayOperator(L.F')
LinearAlgebra.issuccess(L::FactorizedDiffEqArrayOperator) = issuccess(L.F)

isconstant(::FactorizedDiffEqArrayOperator) = true
has_ldiv(::FactorizedDiffEqArrayOperator) = true
has_ldiv!(::FactorizedDiffEqArrayOperator) = true

# operator application (inversion)
Base.:\(L::FactorizedDiffEqArrayOperator, x::AbstractVecOrMat) = L.F \ x
LinearAlgebra.ldiv!(Y::AbstractVector, L::FactorizedDiffEqArrayOperator, B::AbstractVector) = ldiv!(Y, L.F, B)
LinearAlgebra.ldiv!(L::FactorizedDiffEqArrayOperator, B::AbstractVector) = ldiv!(L.F, B)
