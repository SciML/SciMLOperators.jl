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
                           LinearAlgebra.isreal,
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
LinearAlgebra.mul!(v::AbstractVector, L::MatrixOperator, u::AbstractVector, α, β) = mul!(v, L.A, u, α, β)
LinearAlgebra.ldiv!(v::AbstractVector, L::MatrixOperator, u::AbstractVector) = ldiv!(v, L.A, u)
LinearAlgebra.ldiv!(L::MatrixOperator, u::AbstractVector) = ldiv!(L.A, u)

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

for op in (
           :*, :∘
          )
    @eval Base.$op(A::AbstractMatrix, B::AbstractSciMLOperator) = $op(MatrixOperator(A), B)
    @eval Base.$op(A::AbstractSciMLOperator, B::AbstractMatrix) = $op(A, MatrixOperator(B))
end

""" Diagonal Operator """
DiagonalOperator(u::AbstractVector) = MatrixOperator(Diagonal(u))
LinearAlgebra.Diagonal(L::MatrixOperator) = MatrixOperator(Diagonal(L.A))

"""
    InvertibleOperator(F)

Like MatrixOperator, but stores a Factorization instead.

Supports left division and `ldiv!` when applied to an array.
"""
# diagonal, bidiagonal, adjoint(factorization)
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
                               LinearAlgebra.isreal,
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
Base.:*(L::InvertibleOperator, x::AbstractVector) = L.F * x
Base.:\(L::InvertibleOperator, x::AbstractVector) = L.F \ x
LinearAlgebra.mul!(v::AbstractVector, L::InvertibleOperator, u::AbstractVector) = mul!(v, L.F, u)
LinearAlgebra.mul!(v::AbstractVector, L::InvertibleOperator, u::AbstractVector,α, β) = mul!(v, L.F, u, α, β)
LinearAlgebra.ldiv!(v::AbstractVector, L::InvertibleOperator, u::AbstractVector) = ldiv!(v, L.F, u)
LinearAlgebra.ldiv!(L::InvertibleOperator, u::AbstractVector) = ldiv!(L.F, u)

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

function LinearAlgebra.mul!(v::AbstractVector, L::AffineOperator, u::AbstractVector, α, β)
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
struct FunctionOperator{isinplace,T,F,Fa,Fi,Fai,Tr,P,Tt,C} <: AbstractSciMLOperator{T}
    """ Function with signature op(u, p, t) and (if isinplace) op(du, u, p, t) """
    op::F
    """ Adjoint operator"""
    op_adjoint::Fa
    """ Inverse operator"""
    op_inverse::Fi
    """ Adjoint inverse operator"""
    op_adjoint_inverse::Fai
    """ Traits """
    traits::Tr
    """ Parameters """
    p::P
    """ Time """
    t::Tt
    """ Is cache set? """
    isset::Bool
    """ Cache """
    cache::C

    function FunctionOperator(op,
                              op_adjoint,
                              op_inverse,
                              op_adjoint_inverse,
                              traits,
                              p,
                              t,
                              isset,
                              cache
                             )

        iip = traits.isinplace
        T   = traits.T

        isset = cache !== nothing

        new{iip,
            T,
            typeof(op),
            typeof(op_adjoint),
            typeof(op_inverse),
            typeof(op_adjoint_inverse),
            typeof(traits),
            typeof(p),
            typeof(t),
            typeof(cache),
           }(
             op,
             op_adjoint,
             op_inverse,
             op_adjoint_inverse,
             traits,
             p,
             t,
             isset,
             cache,
            )
    end
end

function FunctionOperator(op;

                          # necessary
                          isinplace=nothing,
                          T=nothing,
                          size=nothing,

                          # optional
                          op_adjoint=nothing,
                          op_inverse=nothing,
                          op_adjoint_inverse=nothing,

                          p=nothing,
                          t=nothing,

                          cache=nothing,

                          # traits
                          opnorm=nothing,
                          isreal=true,
                          issymmetric=false,
                          ishermitian=false,
                          isposdef=false,
                         )

    isinplace isa Nothing  && @error "Please provide a funciton signature
    by specifying `isinplace` as either `true`, or `false`.
    If `isinplace = false`, the signature is `op(u, p, t)`,
    and if `isinplace = true`, the signature is `op(du, u, p, t)`.
    Further, it is assumed that the function call would be nonallocating
    when called in-place"
    T isa Nothing  && @error "Please provide a Number type for the Operator"
    size isa Nothing  && @error "Please provide a size (m, n)"

    adjointable = ishermitian | (isreal & issymmetric)
    invertible  = !(op_inverse isa Nothing)

    if adjointable & (op_adjoint isa Nothing) 
        op_adjoint = op
    end

    if invertible & (op_adjoint_inverse isa Nothing)
        op_adjoint_inverse = op_inverse
    end

    t = t isa Nothing ? zero(T) : t

    traits = (;
              opnorm = opnorm,
              isreal = isreal,
              issymmetric = issymmetric,
              ishermitian = ishermitian,
              isposdef = isposdef,

              isinplace = isinplace,
              T = T,
              size = size,
             )

    isset = cache !== nothing

    FunctionOperator(
                     op,
                     op_adjoint,
                     op_inverse,
                     op_adjoint_inverse,
                     traits,
                     p,
                     t,
                     isset,
                     cache,
                    )
end

function update_coefficients!(L::FunctionOperator, u, p, t)
    @set! L.p = p
    @set! L.t = t
    L
end

Base.size(L::FunctionOperator) = L.traits.size
function Base.adjoint(L::FunctionOperator)

    if ishermitian(L) | (isreal(L) & issymmetric(L))
        return L
    end

    if !(has_adjoint(L))
        return AdjointedOperator(L)
    end

    op = L.op_adjoint
    op_adjoint = L.op

    op_inverse = L.op_adjoint_inverse
    op_adjoint_inverse = L.op_inverse

    traits = (L.traits[1:end-1]..., size=reverse(size(L)))

    p = L.p
    t = L.t

    cache = issquare(L) ? cache : nothing
    isset = cache !== nothing


    FuncitonOperator(op,
                     op_adjoint,
                     op_inverse,
                     op_adjoint_inverse,
                     traits,
                     p,
                     t,
                     isset,
                     cache
                    )
end

function LinearAlgebra.opnorm(L::FunctionOperator, p)
  L.traits.opnorm === nothing && error("""
    M.opnorm is nothing, please define opnorm as a function that takes one
    argument. E.g., `(p::Real) -> p == Inf ? 100 : error("only Inf norm is
    defined")`
  """)
  opn = L.opnorm
  return opn isa Number ? opn : M.opnorm(p)
end
LinearAlgebra.isreal(L::FunctionOperator) = L.traits.isreal
LinearAlgebra.issymmetric(L::FunctionOperator) = L.traits.issymmetric
LinearAlgebra.ishermitian(L::FunctionOperator) = L.traits.ishermitian
LinearAlgebra.isposdef(L::FunctionOperator) = L.traits.isposdef

has_adjoint(L::FunctionOperator) = !(L.op_adjoint isa Nothing)
has_mul(L::FunctionOperator{iip}) where{iip} = !iip
has_mul!(L::FunctionOperator{iip}) where{iip} = iip
has_ldiv(L::FunctionOperator{iip}) where{iip} = !iip & !(L.op_inverse isa Nothing)
has_ldiv!(L::FunctionOperator{iip}) where{iip} = iip & !(L.op_inverse isa Nothing)

# operator application
Base.:*(L::FunctionOperator, u::AbstractVector) = L.op(u, L.p, L.t)
Base.:\(L::FunctionOperator, u::AbstractVector) = L.op_inverse(u, L.p, L.t)

function cache_operator(L::FunctionOperator, u::AbstractVector)
    @set! L.cache = similar(u)
    L
end

function LinearAlgebra.mul!(v::AbstractVector, L::FunctionOperator, u::AbstractVector)
    L.op(v, u, L.p, L.t)
end

function LinearAlgebra.mul!(v::AbstractVector, L::FunctionOperator, u::AbstractVector, α, β)
    @assert L.isset "set up cache by calling cache_operator($L, $u)"
    copy!(L.cache, v)
    mul!(v, L, u)
    lmul!(α, v)
    axpy!(β, L.cache, v)
end

function LinearAlgebra.ldiv!(v::AbstractVector, L::FunctionOperator, u::AbstractVector)
    L.op_inverse(v, u, L.p, L.t)
end

function LinearAlgebra.ldiv!(L::FunctionOperator, u::AbstractVector)
    @assert L.isset "set up cache by calling cache_operator($L, $u)"
    copy!(L.cache, u)
    ldiv!(u, L, L.cache)
end

"""
    Lazy Tensor Product Operator

    (A ⊗ B)(u) = vec(A * U * transpose(B))

    where U is a lazy representation of the vector u as
    a matrix with the appropriate size.
"""
struct TensorProduct2DOperator{T,A,B,C} <: SciMLOperators.AbstractSciMLOperator{T}
    A::A
    B::B

    cache::C
    isset::Bool

    function TensorProduct2DOperator(A, B, cache, isset)
        T = promote_type(eltype.((A, B))...)
        isset = cache !== nothing
        new{T,
            typeof(A),
            typeof(B),
            typeof(cache)
           }(
             A, B, cache, isset
            )
    end
end
# make this multidimensional by using the multidimensional indexing
# trick in domains

function TensorProduct2DOperator(A::AbstractMatrix, B::AbstractMatrix; cache = nothing)
    isset = cache !== nothing
    TensorProduct2DOperator(A, B, cache, isset)
end

Base.size(L::TensorProduct2DOperator) = size(A.A) .* size(A.B)

for op in (
           :adjoint,
           :transpose,
          )
    @eval function Base.$op(L::TensorProduct2DOperator)
        TensorProduct2DOperator($op(L.A),
                                $op(L.B);
                                cache = issquare(A) ? L.cache : nothing
                               )
    end
end

function Base.:*(L::TensorProduct2DOperator, u::AbstractVector)
    sz = (size(L.A, 2), size(L.B, 2))
    u = _reshape(u, sz)
    v = L.A * u * transpose(L.B)
end
#
