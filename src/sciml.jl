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
Base.similar(L::MatrixOperator, ::Type{T}, dims::Dims) where{T} = similar(L.A, T, dims)

# traits
@forward MatrixOperator.A (
                           issquare, has_ldiv, has_ldiv!
                          )
Base.size(L::MatrixOperator) = size(L.A)
Base.adjoint(L::MatrixOperator) = MatrixOperator(L.A'; update_func=(A,u,p,t)->L.update_func(L.A,u,p,t)')

has_adjoint(A::MatrixOperator) = has_adjoint(A.A)
update_coefficients!(L::MatrixOperator,u,p,t) = (L.update_func(L.A,u,p,t); L)

isconstant(L::MatrixOperator) = L.update_func == DEFAULT_UPDATE_FUNC
Base.iszero(L::MatrixOperator) = iszero(L.A)

SparseArrays.sparse(L::MatrixOperator) = sparse(L.A)

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
Base.:*(L::MatrixOperator, u::AbstractVector) = L.A * u
Base.:\(L::MatrixOperator, u::AbstractVector) = L.A \ u
LinearAlgebra.mul!(v::AbstractVector, L::MatrixOperator, u::AbstractVector) = mul!(v, L.A, u)

# operator fusion, composition
function Base.:*(A::MatrixOperator, B::MatrixOperator)
    M = A.A * B.A
    update_func = (M,u,p,t) -> A.update_func(M,u,p,t) * B.update_func(M,u,p,t) #TODO
    MatrixOperator(M; update_func=update_func)
end



for op in (
           :*, :/, :\,
          )

    @eval function Base.$op(L::MatrixOperator, x::Number)
        A = $op(L.A, x)
        update_func = L.update_func #TODO
        MatrixOperator(A; update_func=update_func)
    end
    @eval function Base.$op(x::Number, L::MatrixOperator)
        A = $op(x, L.A)
        update_func = L.update_func #TODO
        MatrixOperator(A; update_func=update_func)
    end

    @eval function Base.$op(L::MatrixOperator, x::ScalarOperator)
        A = $op(L.A, x.val)
        update_func = L.update_func #TODO
        MatrixOperator(A; update_func=update_func)
    end
    @eval function Base.$op(x::ScalarOperator, L::MatrixOperator)
        A = $op(x.val, L.A)
        update_func = L.update_func #TODO
        MatrixOperator(A; update_func=update_func)
    end
end

MatMulCompatibleTypes = (
                         :AbstractMatrix,
                         :UniformScaling,
                        )

for op in (
           :+, :-, :*,
          )
    for T in MatMulCompatibleTypes
        @eval function Base.$op(L::MatrixOperator, M::$T)
            A = $op(L.A, M)
            MatrixOperator(A)
        end

        @eval function Base.$op(M::$T, L::MatrixOperator)
            A = $op(M, L.A)
            MatrixOperator(A)
        end
    end
end

"""
    FactorizedOperator(F)

Like MatrixOperator, but stores a Factorization instead.

Supports left division and `ldiv!` when applied to an array.
"""
struct FactorizedOperator{T<:Number,FType<:Union{
                                                 Factorization{T},
                                                 Diagonal{T},
                                                 Bidiagonal{T},
                                                 Adjoint{T,<:Factorization{T}},
                                                }
                         } <: AbstractSciMLLinearOperator{T}
    F::FType
end

# constructor
function LinearAlgebra.factorize(L::AbstractSciMLLinearOperator)
    fact = factorize(convert(AbstractMatrix, L))
    FactorizedOperator(fact)
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
        FactorizedOperator($fact(convert(AbstractMatrix, L), args...))
    @eval LinearAlgebra.$fact(L::AbstractSciMLLinearOperator; kwargs...) =
        FactorizedOperator($fact(convert(AbstractMatrix, L); kwargs...))
end

function Base.convert(::Type{AbstractMatrix}, L::FactorizedOperator)
    if L.F isa Adjoint
        convert(AbstractMatrix,L.F')'
    else
        convert(AbstractMatrix, L.F)
    end
end

# traits
Base.size(L::FactorizedOperator, args...) = size(L.F, args...)
Base.adjoint(L::FactorizedOperator) = FactorizedOperator(L.F')
LinearAlgebra.issuccess(L::FactorizedOperator) = issuccess(L.F)

getops(::FactorizedOperator) = ()
isconstant(::FactorizedOperator) = true
has_ldiv(::FactorizedOperator) = true
has_ldiv!(::FactorizedOperator) = true

# operator application (inversion)
Base.:\(L::FactorizedOperator, x::AbstractVector) = L.F \ x
LinearAlgebra.ldiv!(Y::AbstractVector, L::FactorizedOperator, B::AbstractVector) = ldiv!(Y, L.F, B)
LinearAlgebra.ldiv!(L::FactorizedOperator, B::AbstractVector) = ldiv!(L.F, B)

"""
    L = AffineOperator(A, b)
    L(u) = A*u + b
"""
struct AffineOperator{T,AType,bType} <: AbstractSciMLOperator{T}
    A::AType
    b::bType

    function AffineOperator(A::AbstractSciMLOperator, b::AbstractVector)
        T = promote_type(eltype.((A,b))...)
        new{T,typeof(A),typeof(b)}(A, b)
    end
end

getops(L::AffineOperator) = (L.A, L.b)
Base.size(L::AffineOperator) = size(L.A)

islinear(::AffineOperator) = false
Base.iszero(L::AffineOperator) = all(iszero, getops(L))
has_adjoint(L::AffineOperator) = all(has_adjoint, L.ops)
has_mul!(L::AffineOperator) = has_mul!(L.A)
has_ldiv(L::AffineOperator) = has_ldiv(L.A)
has_ldiv!(L::AffineOperator) = has_ldiv!(L.A)


Base.:*(L::AffineOperator, u::AbstractVector) = L.A * u + L.b
Base.:\(L::AffineOperator, u::AbstractVector) = L.A \ (u - L.b)

function LinearAlgebra.mul!(v::AbstractVector, L::AffineOperator, u::AbstractVector)
    mul!(v, L.A, u)
    axpy!(true, L.b, v)
end

function LinearAlgebra.mul!(v::AbstractVector, L::AffineOperator, u::AbstractVector, α::Number, β::Number)
    mul!(v, L.A, u, α, β)
    axpy!(α, L.b, v)
end

function LinearAlgebra.ldiv!(v::AbstractVector, L::AffineOperator, u::AbstractVector)
    copy!(v, u)
    ldiv!(L, v)
end

function LinearAlgebra.ldiv!(L::AffineOperator, u::AbstractVector)
    axpy!(-true, L.b, u)
    ldiv!(L.A, u)
end

"""
    Matrix free operators (given by a function)
"""
struct FunctionOperator{isinplace,T,F,Fa,Fi,P,S} <: AbstractSciMLOperator{T} # TODO
    """ Function with signature op(u, p, t) and (optionally) op(du, u, p, t) """
    op::F
    """ Adjoint function operator signature op(u, p, t) and (optionally) op(du, u, p, t) """
    op_adjoint::Fa
    """ Adjoint function operator signature op(u, p, t) and (optionally) op(du, u, p, t) """
    op_inverse::Fi
    """ Size """
    size::S
    """ Parameters """
    p::P

    function FunctionOperator(op;
                              isinplace=false,
                              op_adjoint=nothing,
                              op_inverse=nothing,
                              p=nothing,

                              # LinearAlgebra
                              opnorm=nothing,
                              isreal=true,
                              issymmetric=false,
                              ishermitian=false,
                             )
        T = eltype(op)

        if LinearAlgebra.ishermitian(op) & (adjoint === nothing)
            adjoint = op
        end

        new{isinplace,
            T,
            typeof(op),
            typeof(op_adjoint),
            typeof(op_inverse),
            typeof(size),
            typeof(p),
           }(
             op, op_adjoint, op_inverse, size, p,
            )
    end
end

Base.size(L::FunctionOperator) = L.size
Base.adjoint(L::FunctionOperator) = FunctionOperator(L.op_adjoint; op_inverse=L.op)

has_adjoint(L::FunctionOperator) = L.op_adjoint isa Nothing
has_mul!(L::FunctionOperator{iip}) where{iip} = iip
has_ldiv(L::FunctionOperator{iip}) where{iip} = L.op_inverse isa Nothing
has_ldiv!(L::FunctionOperator{iip}) where{iip} = iip & has_ldiv(L)

getops(L::FunctionOperator) = (L.p,)

# operator application
Base.:*(L::FunctionOperator, u::AbstractVector) = L.op(u, p, t)
Base.:\(L::FunctionOperator, u::AbstractVector) = L.op_inverse(u, p, t)
function LinearAlgebra.mul!(v::AbstractVector, L::FunctionOperator, u::AbstractVector)
    L.op(v, u, p, t)
end
function LinearAlgebra.ldiv!(v::AbstractVector, L::FunctionOperator, u::AbstractVector)
    L.op_inverse(v, u, p, t)
end
#
