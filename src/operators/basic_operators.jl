#"""
#$(TYPEDEF)
#"""
struct DiffEqIdentity{N} <: AbstractDiffEqLinearOperator{Bool} end

# constructors
DiffEqIdentity(u::AbstractVector) = DiffEqIdentity{length(u)}()

function Base.one(A::AbstractDiffEqOperator)
    @assert issquare(A)
    N = size(A, 1)
    DiffEqIdentity{N}()
end

Base.convert(::Type{AbstractMatrix}, ::DiffEqIdentity{N}) where {N} = Diagonal(ones(Bool, N))

# traits
Base.size(::DiffEqIdentity{N}) where {N} = (N, N)
Base.adjoint(A::DiffEqIdentity) = A
LinearAlgebra.opnorm(::DiffEqIdentity{N}, p::Real=2) where {N} = true
for pred in (
             :isreal, :issymmetric, :ishermitian, :isposdef,
            )
    @eval LinearAlgebra.$pred(::DiffEqIdentity) = true
end
issquare(::DiffEqIdentity) = true
has_adjoint(::DiffEqIdentity) = true
has_mul!(::DiffEqIdentity) = true
has_ldiv(::DiffEqIdentity) = true
has_ldiv!(::DiffEqIdentity) = true

# opeator application
for op in (
           :*, :/, :\,
          )
    @eval Base.$op(::DiffEqIdentity{N}, x::AbstractVector) where {N} = (@assert length(x) == N; copy(x))
    @eval Base.$op(x::AbstractVector, ::DiffEqIdentity{N}) where {N} = (@assert length(x) == N; copy(x))
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
for op in (:*, :∘)
    @eval Base.$op(::DiffEqIdentity{N}, A::AbstractSciMLOperator) where {N} = (@assert size(A, 1) == N; DiffEqIdentity{N}())
    @eval Base.$op(A::AbstractSciMLOperator, ::DiffEqIdentity{N}) where {N} = (@assert size(A, 2) == N; DiffEqIdentity{N}())
end

#"""
#$(TYPEDEF)
#"""
struct DiffEqNullOperator{N} <: AbstractDiffEqLinearOperator{Bool} end

# constructors
DiffEqNullOperator(u::AbstractVector) = DiffEqNullOperator{length(u)}()

function Base.zero(A::AbstractDiffEqOperator)
    @assert issquare(A)
    N = size(A, 1)
    DiffEqNullOperator{N}()
end

Base.convert(::Type{AbstractMatrix}, ::DiffEqNullOperator{N}) where {N} = Diagonal(zeros(Bool, N))

# traits
Base.size(::DiffEqNullOperator{N}) where {N} = (N, N)
Base.adjoint(A::DiffEqNullOperator) = A
LinearAlgebra.opnorm(::DiffEqNullOperator{N}, p::Real=2) where {N} = false
for pred in (
             :isreal, :issymmetric, :ishermitian,
            )
    @eval LinearAlgebra.$pred(::DiffEqNullOperator) = true
end
LinearAlgebra.isposdef(::DiffEqNullOperator) = false
issquare(::DiffEqNullOperator) = true
has_adjoint(::DiffEqNullOperator) = true
has_mul!(::DiffEqNullOperator) = true

# opeator application
for op in (
           :*, :/, :\,
          )
    @eval Base.$op(::DiffEqNullOperator{N}, x::AbstractVector) where {N} = (@assert length(x) == N; zero(x))
    @eval Base.$op(x::AbstractVector, ::DiffEqNullOperator{N}) where {N} = (@assert length(x) == N; zero(x))
end

function LinearAlgebra.mul!(v::AbstractVector, ::DiffEqNullOperator{N}, u::AbstractArray) where{N}
    @assert length(u) == length(v) == N
    lmul!(false, v)
end

# operator fusion, composition
for op in (:*, :∘)
    @eval Base.$op(::DiffEqNullOperator{N}, A::AbstractSciMLOperator) where {N} = (@assert size(A, 1) == N; DiffEqNullOperator{N}())
    @eval Base.$op(A::AbstractSciMLOperator, ::DiffEqNullOperator{N}) where {N} = (@assert size(A, 2) == N; DiffEqNullOperator{N}())
end

"""
    DiffEqScalar(val[; update_func])

Represents a time-dependent scalar/scaling operator. The update function
is called by `update_coefficients!` and is assumed to have the following
signature:

    update_func(oldval,u,p,t) -> newval
"""
mutable struct DiffEqScalar{T<:Number,F} <: AbstractDiffEqLinearOperator{T}
  val::T
  update_func::F
  DiffEqScalar(val::T; update_func=DEFAULT_UPDATE_FUNC) where {T} =
    new{T,typeof(update_func)}(val, update_func)
end

Base.convert(::Type{Number}, α::DiffEqScalar) = α.val
Base.convert(::Type{DiffEqScalar}, α::Number) = DiffEqScalar(α)
Base.size(::DiffEqScalar) = ()
Base.
update_coefficients!(α::DiffEqScalar,u,p,t) = (α.val = α.update_func(α.val,u,p,t); α)
isconstant(α::DiffEqScalar) = α.update_func == DEFAULT_UPDATE_FUNC

for op in (:*, :/, :\)
  for T in (:AbstractArray, :Number)
    @eval Base.$op(α::DiffEqScalar, x::$T) = $op(α.val, x)
    @eval Base.$op(x::$T, α::DiffEqScalar) = $op(x, α.val)
  end
  @eval Base.$op(x::DiffEqScalar, y::DiffEqScalar) = $op(x.val, y.val)
end

for op in (:-, :+)
  @eval Base.$op(α::DiffEqScalar, x::Number) = $op(α.val, x)
  @eval Base.$op(x::Number, α::DiffEqScalar) = $op(x, α.val)
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
  DiffEqArrayOperator(A::AType; update_func=DEFAULT_UPDATE_FUNC) where {AType} =
    new{eltype(A),AType,typeof(update_func)}(A, update_func)
end

@forward DiffEqArrayOperator.A (
                                issquare, SciMLBase.has_ldiv, SciMLBase.has_ldiv!
                               )

Base.size(A::DiffEqArrayOperator) = size(A.A)
has_adjoint(::DiffEqArrayOperator) = true
update_coefficients!(L::DiffEqArrayOperator,u,p,t) = (L.update_func(L.A,u,p,t); L)
isconstant(L::DiffEqArrayOperator) = L.update_func == DEFAULT_UPDATE_FUNC
Base.similar(L::DiffEqArrayOperator, ::Type{T}, dims::Dims) where T = similar(L.A, T, dims)
Base.adjoint(L::DiffEqArrayOperator) = DiffEqArrayOperator(L.A'; update_func = (A,u,p,t) -> L.update_func(L.A,u,p,t)')

# propagate_inbounds here for the getindex fallback
Base.@propagate_inbounds Base.convert(::Type{AbstractMatrix}, L::DiffEqArrayOperator) = L.A
Base.@propagate_inbounds Base.setindex!(L::DiffEqArrayOperator, v, i::Int) = (L.A[i] = v)
Base.@propagate_inbounds Base.setindex!(L::DiffEqArrayOperator, v, I::Vararg{Int, N}) where {N} = (L.A[I...] = v)

Base.eachcol(L::DiffEqArrayOperator) = eachcol(L.A)
Base.eachrow(L::DiffEqArrayOperator) = eachrow(L.A)
Base.length(L::DiffEqArrayOperator) = length(L.A)
Base.iterate(L::DiffEqArrayOperator,args...) = iterate(L.A,args...)
Base.axes(L::DiffEqArrayOperator) = axes(L.A)
Base.eachindex(L::DiffEqArrayOperator) = eachindex(L.A)
Base.IndexStyle(::Type{<:DiffEqArrayOperator{T,AType}}) where {T,AType} = Base.IndexStyle(AType)
Base.copyto!(L::DiffEqArrayOperator, rhs) = (copyto!(L.A, rhs); L)
Base.copyto!(L::DiffEqArrayOperator, rhs::Base.Broadcast.Broadcasted{<:StaticArrays.StaticArrayStyle}) = (copyto!(L.A, rhs); L)
Base.Broadcast.broadcastable(L::DiffEqArrayOperator) = L
Base.ndims(::Type{<:DiffEqArrayOperator{T,AType}}) where {T,AType} = ndims(AType)
ArrayInterfaceCore.issingular(L::DiffEqArrayOperator) = ArrayInterfaceCore.issingular(L.A)
Base.copy(L::DiffEqArrayOperator) = DiffEqArrayOperator(copy(L.A);update_func=L.update_func)

"""
    FactorizedDiffEqArrayOperator(F)

Like DiffEqArrayOperator, but stores a Factorization instead.

Supports left division and `ldiv!` when applied to an array.
"""
struct FactorizedDiffEqArrayOperator{T<:Number, FType <: Union{
                                                               Factorization{T}, Diagonal{T}, Bidiagonal{T},
                                                               Adjoint{T,<:Factorization{T}},
                                                              }
                                    } <: AbstractDiffEqLinearOperator{T}
  F::FType
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
Base.adjoint(L::FactorizedDiffEqArrayOperator) = FactorizedDiffEqArrayOperator(L.F')
Base.size(L::FactorizedDiffEqArrayOperator, args...) = size(L.F, args...)
LinearAlgebra.ldiv!(Y::AbstractVecOrMat, L::FactorizedDiffEqArrayOperator, B::AbstractVecOrMat) = ldiv!(Y, L.F, B)
LinearAlgebra.ldiv!(L::FactorizedDiffEqArrayOperator, B::AbstractVecOrMat) = ldiv!(L.F, B)
Base.:\(L::FactorizedDiffEqArrayOperator, x::AbstractVecOrMat) = L.F \ x
LinearAlgebra.issuccess(L::FactorizedDiffEqArrayOperator) = issuccess(L.F)

LinearAlgebra.ldiv!(y, L::FactorizedDiffEqArrayOperator, x) = ldiv!(y, L.F, x)
#isconstant(::FactorizedDiffEqArrayOperator) = true
has_ldiv(::FactorizedDiffEqArrayOperator) = true
has_ldiv!(::FactorizedDiffEqArrayOperator) = true

# The (u,p,t) and (du,u,p,t) interface
for T in [DiffEqScalar, DiffEqArrayOperator, FactorizedDiffEqArrayOperator, DiffEqIdentity]
  (L::T)(u,p,t) = (update_coefficients!(L,u,p,t); L * u)
  (L::T)(du,u,p,t) = (update_coefficients!(L,u,p,t); mul!(du,L,u))
end
