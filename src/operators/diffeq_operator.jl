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
iszero(L::DiffEqArrayOperator) = iszero(L.A)

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

getops(L::DiffEqArrayOperator) = (L.A)

# operator application
Base.:*(L::DiffEqArrayOperator, u::AbstractVector) = L.A * u
LinearAlgebra.mul!(v::AbstractVector, L::DiffEqArrayOperator, u::AbstractVector) = mul!(v, L.A, u)

# operator fusion, composition
function Base.:*(A::DiffEqArrayOperator, B::DiffEqArrayOperator)
    M = A.A * B.A
    update_func = (M,u,p,t) -> A.update_func(M,u,p,t) * B.update_func(M,u,p,t) #TODO
    DiffEqArrayOperator(M; update_func=update_func)
end



NumberCompatibleTypes =  (
                          :DiffEqScalar,
                          :Number,
                         )
for op in (
           :*, :/, :\,
          )

    for T in NumberCompatibleTypes
        @eval function Base.$op(L::DiffEqArrayOperator, x::$T)
            A = $op(L.A, x)
            update_func = L.update_func #TODO
            DiffEqArrayOperator(A; update_func=update_func)
        end
        @eval function Base.$op(x::$T, L::DiffEqArrayOperator)
            A = $op(x, L.A)
            update_func = L.update_func #TODO
            DiffEqArrayOperator(A; update_func=update_func)
        end
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
        @eval function Base.$op(L::DiffEqArrayOperator, M::$T)
            A = $op(L.A, M)
            DiffEqArrayOperator(A)
        end

        @eval function Base.$op(M::$T, L::DiffEqArrayOperator)
            A = $op(M, L.A)
            DiffEqArrayOperator(A)
        end
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

getops(::FactorizedDiffEqArrayOperator) = ()
isconstant(::FactorizedDiffEqArrayOperator) = true
has_ldiv(::FactorizedDiffEqArrayOperator) = true
has_ldiv!(::FactorizedDiffEqArrayOperator) = true

# operator application (inversion)
Base.:\(L::FactorizedDiffEqArrayOperator, x::AbstractVecOrMat) = L.F \ x
LinearAlgebra.ldiv!(Y::AbstractVector, L::FactorizedDiffEqArrayOperator, B::AbstractVector) = ldiv!(Y, L.F, B)
LinearAlgebra.ldiv!(L::FactorizedDiffEqArrayOperator, B::AbstractVector) = ldiv!(L.F, B)

"""
AffineDiffEqOperator{T} <: AbstractDiffEqOperator{T}

`Ex: (A₁(t) + ... + Aₙ(t))*u + B₁(t) + ... + Bₘ(t)`

AffineDiffEqOperator{T}(As,Bs,du_cache=nothing)

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
struct AffineDiffEqOperator{T,T1,T2,U} <: AbstractDiffEqOperator{T}
    As::T1
    Bs::T2
    du_cache::U
    function AffineDiffEqOperator{T}(As,Bs,du_cache=nothing) where T
        all([size(a) == size(As[1])
             for a in As]) || error("Operator sizes do not agree")
        new{T,typeof(As),typeof(Bs),typeof(du_cache)}(As,Bs,du_cache)
    end
end

Base.size(L::AffineDiffEqOperator) = size(L.As[1])

getops(L::AffineDiffEqOperator) = (L.As..., L.Bs...)

function (L::AffineDiffEqOperator)(u,p,t::Number)
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

function (L::AffineDiffEqOperator)(du,u,p,t::Number)
    update_coefficients!(L,u,p,t)
    L.du_cache === nothing && error("Can only use inplace AffineDiffEqOperator if du_cache is given.")
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

function update_coefficients!(L::AffineDiffEqOperator,u,p,t)
    # TODO: Make type-stable via recursion
    for A in L.As; update_coefficients!(A,u,p,t); end
    for B in L.Bs; update_coefficients!(B,u,p,t); end
end

@deprecate is_constant(L::AbstractDiffEqOperator) isconstant(L)
#
