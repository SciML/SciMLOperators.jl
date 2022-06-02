"""
    SciMLMatrixOperator(A[; update_func])

Represents a time-dependent linear operator given by an AbstractMatrix. The
update function is called by `update_coefficients!` and is assumed to have
the following signature:

    update_func(A::AbstractMatrix,u,p,t) -> [modifies A]
"""
struct SciMLMatrixOperator{T,AType<:AbstractMatrix{T},F} <: AbstractSciMLLinearOperator{T}
    A::AType
    update_func::F
    SciMLMatrixOperator(A::AType; update_func=DEFAULT_UPDATE_FUNC) where{AType} =
        new{eltype(A),AType,typeof(update_func)}(A, update_func)
end

# constructors
Base.similar(L::SciMLMatrixOperator, ::Type{T}, dims::Dims) where{T} = similar(L.A, T, dims)

# traits
@forward SciMLMatrixOperator.A (
                                issquare, has_ldiv, has_ldiv!
                               )
Base.size(L::SciMLMatrixOperator) = size(L.A)
Base.adjoint(L::SciMLMatrixOperator) = SciMLMatrixOperator(L.A'; update_func=(A,u,p,t)->L.update_func(L.A,u,p,t)')

has_adjoint(A::SciMLMatrixOperator) = has_adjoint(A.A)
update_coefficients!(L::SciMLMatrixOperator,u,p,t) = (L.update_func(L.A,u,p,t); L)

isconstant(L::SciMLMatrixOperator) = L.update_func == DEFAULT_UPDATE_FUNC
Base.iszero(L::SciMLMatrixOperator) = iszero(L.A)

# propagate_inbounds here for the getindex fallback
Base.@propagate_inbounds Base.convert(::Type{AbstractMatrix}, L::SciMLMatrixOperator) = L.A
Base.@propagate_inbounds Base.setindex!(L::SciMLMatrixOperator, v, i::Int) = (L.A[i] = v)
Base.@propagate_inbounds Base.setindex!(L::SciMLMatrixOperator, v, I::Vararg{Int, N}) where{N} = (L.A[I...] = v)

Base.eachcol(L::SciMLMatrixOperator) = eachcol(L.A)
Base.eachrow(L::SciMLMatrixOperator) = eachrow(L.A)
Base.length(L::SciMLMatrixOperator) = length(L.A)
Base.iterate(L::SciMLMatrixOperator,args...) = iterate(L.A,args...)
Base.axes(L::SciMLMatrixOperator) = axes(L.A)
Base.eachindex(L::SciMLMatrixOperator) = eachindex(L.A)
Base.IndexStyle(::Type{<:SciMLMatrixOperator{T,AType}}) where{T,AType} = Base.IndexStyle(AType)
Base.copyto!(L::SciMLMatrixOperator, rhs) = (copyto!(L.A, rhs); L)
Base.copyto!(L::SciMLMatrixOperator, rhs::Base.Broadcast.Broadcasted{<:StaticArrays.StaticArrayStyle}) = (copyto!(L.A, rhs); L)
Base.Broadcast.broadcastable(L::SciMLMatrixOperator) = L
Base.ndims(::Type{<:SciMLMatrixOperator{T,AType}}) where{T,AType} = ndims(AType)
ArrayInterfaceCore.issingular(L::SciMLMatrixOperator) = ArrayInterfaceCore.issingular(L.A)
Base.copy(L::SciMLMatrixOperator) = SciMLMatrixOperator(copy(L.A);update_func=L.update_func)

getops(L::SciMLMatrixOperator) = (L.A)

# operator application
Base.:*(L::SciMLMatrixOperator, u::AbstractVector) = L.A * u
Base.:\(L::SciMLMatrixOperator, u::AbstractVector) = L.A \ u
LinearAlgebra.mul!(v::AbstractVector, L::SciMLMatrixOperator, u::AbstractVector) = mul!(v, L.A, u)

# operator fusion, composition
function Base.:*(A::SciMLMatrixOperator, B::SciMLMatrixOperator)
    M = A.A * B.A
    update_func = (M,u,p,t) -> A.update_func(M,u,p,t) * B.update_func(M,u,p,t) #TODO
    SciMLMatrixOperator(M; update_func=update_func)
end



for op in (
           :*, :/, :\,
          )

    @eval function Base.$op(L::SciMLMatrixOperator, x::Number)
        A = $op(L.A, x)
        update_func = L.update_func #TODO
        SciMLMatrixOperator(A; update_func=update_func)
    end
    @eval function Base.$op(x::Number, L::SciMLMatrixOperator)
        A = $op(x, L.A)
        update_func = L.update_func #TODO
        SciMLMatrixOperator(A; update_func=update_func)
    end

    @eval function Base.$op(L::SciMLMatrixOperator, x::SciMLScalar)
        A = $op(L.A, x.val)
        update_func = L.update_func #TODO
        SciMLMatrixOperator(A; update_func=update_func)
    end
    @eval function Base.$op(x::SciMLScalar, L::SciMLMatrixOperator)
        A = $op(x.val, L.A)
        update_func = L.update_func #TODO
        SciMLMatrixOperator(A; update_func=update_func)
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
        @eval function Base.$op(L::SciMLMatrixOperator, M::$T)
            A = $op(L.A, M)
            SciMLMatrixOperator(A)
        end

        @eval function Base.$op(M::$T, L::SciMLMatrixOperator)
            A = $op(M, L.A)
            SciMLMatrixOperator(A)
        end
    end
end

"""
    SciMLFactorizedOperator(F)

Like SciMLMatrixOperator, but stores a Factorization instead.

Supports left division and `ldiv!` when applied to an array.
"""
struct SciMLFactorizedOperator{T<:Number,FType<:Union{
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
    SciMLFactorizedOperator(fact)
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
        SciMLFactorizedOperator($fact(convert(AbstractMatrix, L), args...))
    @eval LinearAlgebra.$fact(L::AbstractSciMLLinearOperator; kwargs...) =
        SciMLFactorizedOperator($fact(convert(AbstractMatrix, L); kwargs...))
end

function Base.convert(::Type{AbstractMatrix}, L::SciMLFactorizedOperator)
    if L.F isa Adjoint
        convert(AbstractMatrix,L.F')'
    else
        convert(AbstractMatrix, L.F)
    end
end

# traits
Base.size(L::SciMLFactorizedOperator, args...) = size(L.F, args...)
Base.adjoint(L::SciMLFactorizedOperator) = SciMLFactorizedOperator(L.F')
LinearAlgebra.issuccess(L::SciMLFactorizedOperator) = issuccess(L.F)

getops(::SciMLFactorizedOperator) = ()
isconstant(::SciMLFactorizedOperator) = true
has_ldiv(::SciMLFactorizedOperator) = true
has_ldiv!(::SciMLFactorizedOperator) = true

# operator application (inversion)
Base.:\(L::SciMLFactorizedOperator, x::AbstractVector) = L.F \ x
LinearAlgebra.ldiv!(Y::AbstractVector, L::SciMLFactorizedOperator, B::AbstractVector) = ldiv!(Y, L.F, B)
LinearAlgebra.ldiv!(L::SciMLFactorizedOperator, B::AbstractVector) = ldiv!(L.F, B)

"""
AffineSciMLOperator{T} <: AbstractSciMLOperator{T}

`Ex: (A₁(t) + ... + Aₙ(t))*u + B₁(t) + ... + Bₘ(t)`

AffineSciMLOperator{T}(As,Bs,du_cache=nothing)

Takes in two tuples for split Affine DiffEqs

1. update_coefficients! works by updating the coefficients of the component
   operators.
2. Function calls L(u, p, t) and L(du, u, p, t) are fallbacks interpretted in this form.
   This will allow them to work directly in the nonlinear ODE solvers without
   modification.
3. f(du, u, p, t) is only allowed if a du_cache is given
4. B(t) can be Union{Number,AbstractArray}, in which case they are constants.
   Otherwise they are interpreted they are functions v=B(t) and B(v,t)

Solvers will see this operator from integrator.f and can interpret it by
checking the internals of As and Bs. For example, it can check isconstant(As[1])
etc.
"""
struct AffineSciMLOperator{T,T1,T2,U} <: AbstractSciMLOperator{T}
    As::T1
    Bs::T2
    du_cache::U
    function AffineSciMLOperator{T}(As,Bs,du_cache=nothing) where T
        all([size(a) == size(As[1])
             for a in As]) || error("Operator sizes do not agree")
        new{T,typeof(As),typeof(Bs),typeof(du_cache)}(As,Bs,du_cache)
    end
end

Base.size(L::AffineSciMLOperator) = size(L.As[1])

getops(L::AffineSciMLOperator) = (L.As..., L.Bs...)

function (L::AffineSciMLOperator)(u,p,t::Number)
    update_coefficients!(L,u,p,t)
    du = sum(A*u for A in L.As)
    for B in L.Bs
        if typeof(B) <: Union{Number,AbstractArray}
            du .+= B
        else
            du .+= B(t)
        end
    end
    du
end

function (L::AffineSciMLOperator)(du,u,p,t::Number)
    update_coefficients!(L,u,p,t)
    L.du_cache === nothing && error("Can only use inplace AffineSciMLOperator if du_cache is given.")
    du_cache = L.du_cache
    fill!(du,zero(first(du)))
    # TODO: Make type-stable via recursion
    for A in L.As
        mul!(du_cache,A,u)
        du .+= du_cache
    end
    for B in L.Bs
        if typeof(B) <: Union{Number,AbstractArray}
            du .+= B
        else
            B(du_cache,t)
            du .+= du_cache
        end
    end
end

function update_coefficients!(L::AffineSciMLOperator,u,p,t)
    # TODO: Make type-stable via recursion
    for A in L.As; update_coefficients!(A,u,p,t); end
    for B in L.Bs; update_coefficients!(B,u,p,t); end
end

"""
    Matrix free operators (given by a function)
"""
struct SciMLFunctionOperator{isinplace,T,F,Fa,Fi,P,S} <: AbstractSciMLOperator{T} # TODO
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

    function SciMLFunctionOperator(op;
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

Base.size(L::SciMLFunctionOperator) = L.size
Base.adjoint(L::SciMLFunctionOperator) = SciMLFunctionOperator(L.op_adjoint; op_inverse=L.op)

has_adjoint(L::SciMLFunctionOperator) = L.op_adjoint isa Nothing
has_mul!(L::SciMLFunctionOperator{iip}) where{iip} = iip
has_ldiv(L::SciMLFunctionOperator{iip}) where{iip} = L.op_inverse isa Nothing
has_ldiv!(L::SciMLFunctionOperator{iip}) where{iip} = iip & has_ldiv(L)

getops(L::SciMLFunctionOperator) = (L.p,)

# operator application
Base.:*(L::SciMLFunctionOperator, u::AbstractVector) = L.op(u, p, t)
Base.:\(L::SciMLFunctionOperator, u::AbstractVector) = L.op_inverse(u, p, t)
function LinearAlgebra.mul!(v::AbstractVector, L::SciMLFunctionOperator, u::AbstractVector)
    L.op(v, u, p, t)
end
function LinearAlgebra.ldiv!(v::AbstractVector, L::SciMLFunctionOperator, u::AbstractVector)
    L.op_inverse(v, u, p, t)
end
#
