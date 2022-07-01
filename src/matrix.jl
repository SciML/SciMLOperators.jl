"""
    MatrixOperator(A[; update_func])

Represents a time-dependent linear operator given by an AbstractMatrix. The
update function is called by `update_coefficients!` and is assumed to have
the following signature:

    update_func(A::AbstractMatrix,u,p,t) -> [modifies A]
"""
struct MatrixOperator{T,AType<:AbstractMatrix{T},F} <: AbstractSciMLLinearOperator{T}
    A::AType
    update_func::F
    MatrixOperator(A::AType; update_func=DEFAULT_UPDATE_FUNC) where{AType} =
        new{eltype(A),AType,typeof(update_func)}(A, update_func)
end

# constructors
Base.similar(L::MatrixOperator, ::Type{T}, dims::Dims) where{T} = MatrixOperator(similar(L.A, T, dims))

# traits
@forward MatrixOperator.A (
                           LinearAlgebra.issymmetric,
                           LinearAlgebra.ishermitian,
                           LinearAlgebra.isposdef,

                           issquare,
                           has_ldiv,
                           has_ldiv!,
                          )
Base.size(L::MatrixOperator) = size(L.A)
for op in (
           :adjoint,
           :transpose,
          )
    @eval function Base.$op(L::MatrixOperator)
        MatrixOperator(
                       $op(L.A);
                       update_func = (A,u,p,t) -> $op(L.update_func(L.A,u,p,t))
                      )
    end
end

has_adjoint(A::MatrixOperator) = has_adjoint(A.A)
update_coefficients!(L::MatrixOperator,u,p,t) = (L.update_func(L.A,u,p,t); nothing)

isconstant(L::MatrixOperator) = L.update_func == DEFAULT_UPDATE_FUNC
Base.iszero(L::MatrixOperator) = iszero(L.A)

SparseArrays.sparse(L::MatrixOperator) = sparse(L.A)

# TODO - add tests for MatrixOperator indexing
# propagate_inbounds here for the getindex fallback
Base.@propagate_inbounds Base.convert(::Type{AbstractMatrix}, L::MatrixOperator) = L.A
Base.@propagate_inbounds Base.setindex!(L::MatrixOperator, v, i::Int) = (L.A[i] = v)
Base.@propagate_inbounds Base.setindex!(L::MatrixOperator, v, I::Vararg{Int, N}) where{N} = (L.A[I...] = v)

Base.eachcol(L::MatrixOperator) = eachcol(L.A)
Base.eachrow(L::MatrixOperator) = eachrow(L.A)
Base.length(L::MatrixOperator) = length(L.A)
Base.iterate(L::MatrixOperator,args...) = iterate(L.A,args...)
Base.axes(L::MatrixOperator) = axes(L.A)
Base.eachindex(L::MatrixOperator) = eachindex(L.A)
Base.IndexStyle(::Type{<:MatrixOperator{T,AType}}) where{T,AType} = Base.IndexStyle(AType)
Base.copyto!(L::MatrixOperator, rhs) = (copyto!(L.A, rhs); L)
Base.copyto!(L::MatrixOperator, rhs::Base.Broadcast.Broadcasted{<:StaticArrays.StaticArrayStyle}) = (copyto!(L.A, rhs); L)
Base.Broadcast.broadcastable(L::MatrixOperator) = L
Base.ndims(::Type{<:MatrixOperator{T,AType}}) where{T,AType} = ndims(AType)
ArrayInterfaceCore.issingular(L::MatrixOperator) = ArrayInterfaceCore.issingular(L.A)
Base.copy(L::MatrixOperator) = MatrixOperator(copy(L.A);update_func=L.update_func)

getops(L::MatrixOperator) = (L.A)

# operator application
Base.:*(L::MatrixOperator, u::AbstractVecOrMat) = L.A * u
Base.:\(L::MatrixOperator, u::AbstractVecOrMat) = L.A \ u
LinearAlgebra.mul!(v::AbstractVecOrMat, L::MatrixOperator, u::AbstractVecOrMat) = mul!(v, L.A, u)
LinearAlgebra.mul!(v::AbstractVecOrMat, L::MatrixOperator, u::AbstractVecOrMat, α, β) = mul!(v, L.A, u, α, β)
LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::MatrixOperator, u::AbstractVecOrMat) = ldiv!(v, L.A, u)
LinearAlgebra.ldiv!(L::MatrixOperator, u::AbstractVecOrMat) = ldiv!(L.A, u)

""" Diagonal Operator """
function DiagonalOperator(u::AbstractVector; update_func=DEFAULT_UPDATE_FUNC)
    function diag_update_func(A, u, p, t)
        update_func(A.diag, u, p, t)
        A
    end
    MatrixOperator(Diagonal(u); update_func=diag_update_func)
end
LinearAlgebra.Diagonal(L::MatrixOperator) = MatrixOperator(Diagonal(L.A))

"""
    InvertibleOperator(F)

Like MatrixOperator, but stores a Factorization instead.

Supports left division and `ldiv!` when applied to an array.
"""
struct InvertibleOperator{T,FType} <: AbstractSciMLLinearOperator{T}
    F::FType

    function InvertibleOperator(F)
        @assert has_ldiv(F) | has_ldiv!(F) "$F is not invertible"
        new{eltype(F),typeof(F)}(F)
    end
end

# constructor
function LinearAlgebra.factorize(L::AbstractSciMLLinearOperator)
    fact = factorize(convert(AbstractMatrix, L))
    InvertibleOperator(fact)
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

    @eval LinearAlgebra.$fact(L::AbstractSciMLLinearOperator, args...) =
        InvertibleOperator($fact(convert(AbstractMatrix, L), args...))
    @eval LinearAlgebra.$fact(L::AbstractSciMLLinearOperator; kwargs...) =
        InvertibleOperator($fact(convert(AbstractMatrix, L); kwargs...))
end

function Base.convert(::Type{AbstractMatrix}, L::InvertibleOperator)
    if L.F isa Adjoint
        convert(AbstractMatrix,L.F')'
    else
        convert(AbstractMatrix, L.F)
    end
end

# traits
Base.size(L::InvertibleOperator) = size(L.F)
Base.adjoint(L::InvertibleOperator) = InvertibleOperator(L.F')
LinearAlgebra.opnorm(L::InvertibleOperator{T}, p=2) where{T} = one(T) / opnorm(L.F)
LinearAlgebra.issuccess(L::InvertibleOperator) = issuccess(L.F)

getops(L::InvertibleOperator) = (L.F,)

@forward InvertibleOperator.F (
                               # LinearAlgebra
                               LinearAlgebra.issymmetric,
                               LinearAlgebra.ishermitian,
                               LinearAlgebra.isposdef,

                               # SciML
                               isconstant,
                               has_adjoint,
                               has_mul,
                               has_mul!,
                               has_ldiv,
                               has_ldiv!,
                              )

# operator application
Base.:*(L::InvertibleOperator, x::AbstractVecOrMat) = L.F * x
Base.:\(L::InvertibleOperator, x::AbstractVecOrMat) = L.F \ x
LinearAlgebra.mul!(v::AbstractVecOrMat, L::InvertibleOperator, u::AbstractVecOrMat) = mul!(v, L.F, u)
LinearAlgebra.mul!(v::AbstractVecOrMat, L::InvertibleOperator, u::AbstractVecOrMat,α, β) = mul!(v, L.F, u, α, β)
LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::InvertibleOperator, u::AbstractVecOrMat) = ldiv!(v, L.F, u)
LinearAlgebra.ldiv!(L::InvertibleOperator, u::AbstractVecOrMat) = ldiv!(L.F, u)

"""
    L = AffineOperator(A, B, b[; update_func])
    L(u) = A*u + B*b

Represents a time-dependent affine operator. The update function is called
by `update_coefficients!` and is assumed to have the following signature:

    update_func(b::AbstractArray,u,p,t) -> [modifies b]
"""
struct AffineOperator{T,AType,BType,bType,cType,F} <: AbstractSciMLOperator{T}
    A::AType
    B::BType
    b::bType

    cache::cType
    update_func::F

    function AffineOperator(A, B, b, cache, update_func)
        T = promote_type(eltype.((A,B,b))...)

        new{T,
            typeof(A),
            typeof(B),
            typeof(b),
            typeof(cache),
            typeof(update_func),
           }(
             A, B, b, cache,
            )
    end
end

function AffineOperator(A::Union{AbstractMatrix,AbstractSciMLOperator},
                        B::Union{AbstractMatrix,AbstractSciMLOperator},
                        b::AbstractVecOrMat;
                        update_func=DEFAULT_UPDATE_FUNC,
                       )
    @assert size(A, 1) == size(B, 1) "Dimension mismatch: A, B don't output vectors
    of same size"

    A = A isa AbstractMatrix ? MatrixOperator(A) : A
    B = B isa AbstractMatrix ? MatrixOperator(B) : B
    cache = B * b

    AffineOperator(A, B, b, cache, update_func)
end

function AddVector(b::AbstractVecOrMat; update_func=DEFAULT_UPDATE_FUNC)
    N  = size(b, 1)
    Z  = NullOperator{N}()
    Id = IdentityOperator{N}()

    AffineOperator(Id, B, b; update_func=update_func)
end

function AddVector(B, b::AbstractVecOrMat; update_func=DEFAULT_UPDATE_FUNC)
    N = size(B, 1)
    Z = NullOperator{N}()

    AffineOperator(Z, B, b; update_func=update_func)
end

getops(L::AffineOperator) = (L.A, L.B, L.b)
Base.size(L::AffineOperator) = size(L.A)

update_coefficients!(L::AffineOperator,u,p,t) = (L.update_func(L.b,u,p,t); nothing)

islinear(::AffineOperator) = false
Base.iszero(L::AffineOperator) = all(iszero, getops(L))
has_adjoint(L::AffineOperator) = all(has_adjoint, L.ops)
has_mul!(L::AffineOperator) = has_mul!(L.A)
has_ldiv(L::AffineOperator) = has_ldiv(L.A)
has_ldiv!(L::AffineOperator) = has_ldiv!(L.A)

function cache_internals(L::AffineOperator, u::AbstractVecOrMat)
    @set! L.A = cache_operator(L.A, u)
    @set! L.B = cache_operator(L.B, u)
    @set! L.b = cache_operator(L.b, u)
    L
end

function Base.:*(L::AffineOperator, u::AbstractVecOrMat)
    @assert size(L.b, 2) == size(u, 2)
    (L.A * u) + (L.B * L.b)
end

function Base.:\(L::AffineOperator, u::AbstractVecOrMat)
    @assert size(L.b, 2) == size(u, 2)
    L.A \ (u - (L.B * L.b))
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::AffineOperator, u::AbstractVecOrMat)
    mul!(v, L.A, u)
    mul!(L.cache, L.B, L.b)
    axpy!(true, L.cache, v)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::AffineOperator, u::AbstractVecOrMat, α, β)
    mul!(L.cache, L.B, L.b)
    mul!(v, L.A, u, α, β)
    axpy!(α, L.cache, v)
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::AffineOperator, u::AbstractVecOrMat)
    copy!(v, u)
    ldiv!(L, v)
end

function LinearAlgebra.ldiv!(L::AffineOperator, u::AbstractVecOrMat)
    mul!(L.cache, L.B, L.b)
    axpy!(-true, L.cache, u)
    ldiv!(L.A, u)
end
#
