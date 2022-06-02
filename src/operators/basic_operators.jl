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

getops(::DiffEqIdentity) = ()
isconstant(::DiffEqIdentity) = true
islinear(L::DiffEqIdentity) = true
iszero(::DiffEqIdentity) = false
has_adjoint(::DiffEqIdentity) = true
has_mul!(::DiffEqIdentity) = true
has_ldiv(::DiffEqIdentity) = true
has_ldiv!(::DiffEqIdentity) = true

# opeator application
for op in (
           :*, :\,
          )
    @eval function Base.$op(::DiffEqIdentity{N}, x::AbstractVector) where{N}
        @assert length(x) == N
        copy(x)
    end
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
for op in (
           :*, :∘, :/, :\,
          )
    @eval function Base.$op(::DiffEqIdentity{N}, A::AbstractSciMLOperator) where{N}
        @assert size(A, 1) == N
        DiffEqIdentity{N}()
    end

    @eval function Base.$op(A::AbstractSciMLOperator, ::DiffEqIdentity{N}) where{N}
        @assert size(A, 2) == N
        DiffEqIdentity{N}()
    end
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

getops(::DiffEqNullOperator) = ()
isconstant(::DiffEqNullOperator) = true
islinear(L::DiffEqNullOperator) = true
iszero(::DiffEqNullOperator) = true
has_adjoint(::DiffEqNullOperator) = true
has_mul!(::DiffEqNullOperator) = true

# opeator application
Base.:*(::DiffEqNullOperator{N}, x::AbstractVector) where{N} = (@assert length(x) == N; zero(x))
Base.:*(x::AbstractVector, ::DiffEqNullOperator{N}) where{N} = (@assert length(x) == N; zero(x))

function LinearAlgebra.mul!(v::AbstractVector, ::DiffEqNullOperator{N}, u::AbstractVector) where{N}
    @assert length(u) == length(v) == N
    lmul!(false, v)
end

# operator fusion, composition
for op in (:*, :∘)
    @eval function Base.$op(::DiffEqNullOperator{N}, A::AbstractSciMLOperator) where{N}
        @assert size(A, 1) == N
        DiffEqNullOperator{N}()
    end

    @eval function Base.$op(A::AbstractSciMLOperator, ::DiffEqNullOperator{N}) where{N}
        @assert size(A, 2) == N
        DiffEqNullOperator{N}()
    end
end

"""
    DiffEqScalar(val[; update_func])

    (α::DiffEqScalar)(a::Number) = α * a

Represents a time-dependent scalar/scaling operator. The update function
is called by `update_coefficients!` and is assumed to have the following
signature:

    update_func(oldval,u,p,t) -> newval
"""
struct DiffEqScalar{T<:Number,F} <: AbstractDiffEqLinearOperator{T}
    val::T
    update_func::F
    DiffEqScalar(val::T; update_func=DEFAULT_UPDATE_FUNC) where{T} =
        new{T,typeof(update_func)}(val, update_func)
end

# constructors
Base.convert(::Type{Number}, α::DiffEqScalar) = α.val
Base.convert(::Type{DiffEqScalar}, α::Number) = DiffEqScalar(α)

DiffEqScalar(α::DiffEqScalar) = α
DiffEqScalar(λ::UniformScaling) = DiffEqScalar(λ.λ)

# traits
Base.size(α::DiffEqScalar) = ()
function Base.adjoint(α::DiffEqScalar) # TODO - test
    val = α.val'
    update_func =  (oldval,u,p,t) -> α.update_func(oldval',u,p,t)'
    DiffEqScalar(val; update_func=update_func)
end

getops(α::DiffEqScalar) = (α.val)
islinear(L::DiffEqScalar) = true
isconstant(α::DiffEqScalar) = α.update_func == DEFAULT_UPDATE_FUNC
iszero(α::DiffEqScalar) = iszero(α.val)
has_adjoint(::DiffEqScalar) = true
has_mul(::DiffEqScalar) = true
has_ldiv(α::DiffEqScalar) = iszero(α.val)

for op in (
           :*, :/, :\,
          )
    for T in (
              :Number,
              :AbstractArray,
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
    ScaledDiffEqOperator

    (λ L)*(u) = λ * L(u)
"""
struct ScaledDiffEqOperator{T,
                            λType<:DiffEqScalar,
                            LType<:AbstractDiffEqOperator,
                           } <: AbstractDiffEqOperator{T}
    λ::λType
    L::LType

    function ScaledDiffEqOperator(λ::DiffEqScalar, L::AbstractDiffEqOperator)
        T = promote_type(eltype.((λ, L))...)
        new{T,typeof(λ),typeof(L)}(λ, L)
    end
end

ScalingNumberTypes = (
                      :DiffEqScalar,
                      :Number,
                      :UniformScaling,
                     )

# constructors
for T in ScalingNumberTypes[2:end]
    @eval ScaledDiffEqOperator(λ::$T, L::AbstractDiffEqOperator) = ScaledDiffEqOperator(DiffEqScalar(λ), L)
end

for T in ScalingNumberTypes
    @eval function ScaledDiffEqOperator(λ::$T, L::ScaledDiffEqOperator)
        λ = DiffEqScalar(λ) * L.λ
        ScaledDiffEqOperator(λ, L.L)
    end
    
    @eval Base.:*(λ::$T, L::AbstractDiffEqOperator) = ScaledDiffEqOperator(λ, L)
    @eval Base.:*(L::AbstractDiffEqOperator, λ::$T) = ScaledDiffEqOperator(λ, L)
    @eval Base.:\(λ::$T, L::AbstractDiffEqOperator) = ScaledDiffEqOperator(inv(λ), L)
    @eval Base.:\(L::AbstractDiffEqOperator, λ::$T) = ScaledDiffEqOperator(λ, inv(L))
    @eval Base.:/(L::AbstractDiffEqOperator, λ::$T) = ScaledDiffEqOperator(inv(λ), L)
    @eval Base.:/(λ::$T, L::AbstractDiffEqOperator) = ScaledDiffEqOperator(λ, inv(L))
end

Base.:-(L::AbstractDiffEqOperator) = ScaledDiffEqOperator(-true, L)
Base.:+(L::AbstractDiffEqOperator) = L

Base.convert(::Type{AbstractMatrix}, L::ScaledDiffEqOperator) = λ * convert(AbstractMatrix, L.L)
Base.Matrix(L::ScaledDiffEqOperator) = L.λ * Matrix(L.L)

# traits
Base.size(L::ScaledDiffEqOperator) = size(L.L)
Base.adjoint(L::ScaledDiffEqOperator) = ScaledDiffEqOperator(L.λ', L.op')
LinearAlgebra.opnorm(L::ScaledDiffEqOperator, p::Real=2) = abs(L.λ) * opnorm(L.L, p)

getops(L::ScaledDiffEqOperator) = (L.λ, L.A)
islinear(L::ScaledDiffEqOperator) = all(islinear, L.ops)
isconstant(L::ScaledDiffEqOperator) = isconstant(L.L) & isconstant(L.λ)
iszero(L::ScaledDiffEqOperator) = iszero(L.L) & iszero(L.λ)
has_adjoint(L::ScaledDiffEqOperator) = has_adjoint(L.L)
has_mul!(L::ScaledDiffEqOperator) = has_mul!(L.L)
has_ldiv(L::ScaledDiffEqOperator) = has_ldiv(L.L) & !iszero(L.λ)
has_ldiv!(L::ScaledDiffEqOperator) = has_ldiv!(L.L) & !iszero(L.λ)

# getindex
Base.getindex(L::ScaledDiffEqOperator, i::Int) = L.coeff * L.op[i]
Base.getindex(L::ScaledDiffEqOperator, I::Vararg{Int, N}) where {N} = L.λ * L.L[I...]
for fact in (
             :lu, :lu!,
             :qr, :qr!,
             :cholesky, :cholesky!,
             :ldlt, :ldlt!,
             :bunchkaufman, :bunchkaufman!,
             :lq, :lq!,
             :svd, :svd!,
            )
    @eval LinearAlgebra.$fact(L::ScaledDiffEqOperator, args...) = L.λ * fact(L.L, args...)
end

# operator application, inversion
for op in (
           :*, :\,
          )
    @eval Base.$op(L::ScaledDiffEqOperator, x::AbstractVector) = $op(L.λ, $op(L.L, x))
end

function LinearAlgebra.mul!(v::AbstractVector, L::ScaledDiffEqOperator, u::AbstractVector)
    mul!(v, L.L, u)
    lmul!(L.λ, v)
end

"""
Lazy operator addition (A + B)

    (A1 + A2 + A3...)u = A1*u + A2*u + A3*u ....
"""
struct AddedDiffEqOperator{T,
                           O<:Tuple{Vararg{AbstractDiffEqOperator}},
                           C,
                          } <: AbstractDiffEqOperator{T}
    ops::O
    cache::C
    isunset::Bool

    function AddedDiffEqOperator(ops...; cache = nothing)
        sz = size(first(ops))
        for op in ops[2:end]
            @assert size(op) == sz "Size mismatich in operators $ops"
        end

        T = promote_type(eltype.(ops)...)
        isunset = cache === nothing
        new{T,typeof(ops),typeof(cache)}(ops, cache, isunset)
    end
end

# constructors
Base.:+(ops::AbstractDiffEqOperator...) = AddedDiffEqOperator(ops...)

Base.:-(L::AddedDiffEqOperator) = AddedDiffEqOperator(.-(A.ops)...)
Base.:-(A::AbstractDiffEqOperator, B::AbstractDiffEqOperator) = AddedDiffEqOperator(A, -B)

for op in (
           :+, :-,
          )

    @eval Base.$op(A::AddedDiffEqOperator, B::AddedDiffEqOperator) = AddedDiffEqOperator(A.ops..., $op(B).ops...)
    @eval Base.$op(A::AbstractDiffEqOperator, B::AddedDiffEqOperator) = AddedDiffEqOperator(A, $op(B).ops...)
    @eval Base.$op(A::AddedDiffEqOperator, B::AbstractDiffEqOperator) = AddedDiffEqOperator(A.ops..., $op(B))

    for T in ScalingNumberTypes
        @eval function Base.$op(L::AbstractDiffEqOperator, λ::$T)
            @assert issquare(L)
            N  = size(L, 1)
            Id = DiffEqIdentity{N}()
            AddedDiffEqOperator(L, $op(λ)*Id)
        end

        @eval function Base.$op(λ::$T, L::AbstractDiffEqOperator)
            @assert issquare(L)
            N  = size(L, 1)
            Id = DiffEqIdentity{N}()
            AddedDiffEqOperator(λ*Id, $op(L))
        end
    end

    @eval function Base.$op(A::AbstractMatrix, L::AbstractDiffEqOperator)
        @assert size(A) == size(L)
        AddedDiffEqOperator(DiffEqArrayOperator(A), $op(L))
    end

    @eval function Base.$op(L::AbstractDiffEqOperator, A::AbstractMatrix)
        @assert size(A) == size(L)
        AddedDiffEqOperator(L, DiffEqArrayOperator($op(A)))
    end
end

Base.convert(::Type{AbstractMatrix}, L::AddedDiffEqOperator) = sum(convert.(AbstractMatrix, ops))
Base.Matrix(L::AddedDiffEqOperator) = sum(Matrix.(ops))
SparseArrays.sparse(L::AddedDiffEqOperator) = sum(_sparse, L.ops)

# traits
Base.size(L::AddedDiffEqOperator) = size(first(L.ops))
function Base.adjoint(L::AddedDiffEqOperator)
    if issquare(L) & !(L.isunset)
        AddedDiffEqOperator(adjoint.(L.ops)...,L.cache, L.isunset)
    else
        AddedDiffEqOperator(adjoint.(L.ops)...)
    end
end

getops(L::AddedDiffEqOperator) = L.ops
iszero(L::AddedDiffEqOperator) = all(iszero, getops(L))
has_adjoint(L::AddedDiffEqOperator) = all(has_adjoint, L.ops)

getindex(L::AddedDiffEqOperator, i::Int) = sum(op -> op[i], L.ops)
getindex(L::AddedDiffEqOperator, I::Vararg{Int, N}) where {N} = sum(op -> op[I...], L.ops)

function init_cache(A::AddedDiffEqOperator, u::AbstractVector)
    cache = A.B * u
end

function Base.:*(L::AddedDiffEqOperator, u::AbstractVector)
    sum(op -> iszero(op) ? similar(u, Bool) * false : op * u, L.ops)
end

function LinearAlgebra.mul!(v::AbstractVector, L::AddedDiffEqOperator, u::AbstractVector)
    mul!(v, first(L.ops), u)
    for op in L.ops[2:end]
        iszero(op) && continue
        mul!(L.cache, op, u)
        axpy!(true, L.cache, v)
    end
end

"""
    Matrix free operators (given by a function)
"""
struct DiffEqFunctionOperator{isinplace,T,F,Fa,P,Tr} <: AbstractDiffEqOperator{T}
    op::F
    op_adjoint::Fa
    p::P
    traits::Tr

    function DiffEqFunctionOperator(op;
                                    isinplace=false,
                                    adjoint=nothing,
                                    p=nothing,
                                    traits=nothing,
                                    kwargs...
                                   )
        traits = traits !== nothing ? traits : SciMLOperatorTraits(;kwargs...)
        T = eltype(op)
        new{isinplace,
            T,
            typeof(op),
            typeof(op_adjoint),
            typeof(p),
            typeof(traits)
           }(
             op, adjoint, p, traits
            )
    end
end

Base.:*(L::DiffEqFunctionOperator, u::AbstractVector) = L.op(u, p, t)
#Base.:\
#LinearAlgebra.mul!()
#LinearAlgebra.ldiv!()
#LinearAlgebra.ldiv!()

"""
    Lazy operator composition

    ∘(A, B, C)(u) = A(B(C(u)))

    ops = (A, B, C)
    cache = (B*C*u , C*u)
"""
struct ComposedDiffEqOperator{T,O,C} <: AbstractDiffEqOperator{T}
    """ Tuple of N operators to be applied in reverse"""
    ops::O
    """ Tuple of N-1 cache vectors. cache[N-1] = op[N] * u and so on """
    cache::C
    isunset::Bool
    function ComposedDiffEqOperator(ops::AbstractDiffEqOperator...)
        for i in 1:length(ops)-1
            opcurr = op[i]
            opnext = op[i+1]
            @assert size(opcurr, 1) == size(opnext, 2) "Cannot $opnext ∘ $opcurr. Size mismatich"
        end

        T = promote_type(eltype.(ops)...)
        isunset = cache === nothing
        new{T,typeof(ops),typeof(cache)}(ops, cache, isunset)
    end
end

function init_cache(L::ComposedDiffEqOperator, u::AbstractVector)
    cache = ()
    vec = u
    for i in reverse(2:length(L.ops))
        vec = op[i] * vec
        cache = push(cache, vec)
    end
    cache
end

# constructors
Base.:∘(ops::AbstractDiffEqOperator...) = ComposedDiffEqOperator(ops)

Base.:∘(A::ComposedDiffEqOperator, B::ComposedDiffEqOperator) = ComposedDiffEqOperator(A.ops..., B.ops...)
Base.:∘(A::AbstractDiffEqOperator, B::ComposedDiffEqOperator) = ComposedDiffEqOperator(A, B.ops...)
Base.:∘(A::ComposedDiffEqOperator, B::AbstractDiffEqOperator) = ComposedDiffEqOperator(A.ops..., B)

# operator fusion falls back on composition
Base.:*(ops::AbstractDiffEqOperator...) = ComposedDiffEqOperator(ops...)

Base.:*(A::AbstractDiffEqOperator, B::AbstractDiffEqOperator) = ∘(A, B)
Base.:*(A::ComposedDiffEqOperator, B::AbstractDiffEqOperator) = ∘(A.ops[1:end-1]..., A.ops[end] * B)
Base.:*(A::AbstractDiffEqOperator, B::ComposedDiffEqOperator) = ∘(A * B.ops[1], B.ops[2:end]...)


Base.Matrix(L::ComposedDiffEqOperator) = prod(Matrix, L.ops)
Base.convert(::Type{AbstractMatrix}, L::ComposedDiffEqOperator) = prod(op -> convert(AbstractMatrix, op), L.ops)

SparseArrays.sparse(L::ComposedDiffEqOperator) = prod(_sparse, L.ops)

# traits
Base.size(L::ComposedDiffEqOperator) = (size(first(L.ops), 1), size(last(L.ops),2))
Base.adjoint(L::ComposedDiffEqOperator) = ComposedDiffEqOperator(adjoint.(reverse(L.ops)))
LinearAlgebra.opnorm(L::ComposedDiffEqOperator) = prod(opnorm, L.ops)

getops(L::ComposedDiffEqOperator) = L.ops
islinear(L::ComposedDiffEqOperator) = all(islinear, L.ops)
iszero(L::ComposedDiffEqOperator) = all(iszero, getops(L))
has_adjoint(L::ComposedDiffEqOperator) = all(has_adjoint, L.ops)
has_mul!(L::ComposedDiffEqOperator) = all(has_mul!, L.ops)
has_ldiv(L::ComposedDiffEqOperator) = all(has_ldiv, L.ops)
has_ldiv!(L::ComposedDiffEqOperator) = all(has_mul!, L.ops)

factorize(L::ComposedDiffEqOperator) = prod(factorize, reverse(L.ops))
for fact in (
             :lu, :lu!,
             :qr, :qr!,
             :cholesky, :cholesky!,
             :ldlt, :ldlt!,
             :bunchkaufman, :bunchkaufman!,
             :lq, :lq!,
             :svd, :svd!,
            )
    @eval LinearAlgebra.$fact(L::ComposedDiffEqOperator, args...) = prod(op -> $fact(op, args...), reverse(L.ops))
end

# operator application
Base.:*(L::ComposedDiffEqOperator, u::AbstractVector) = foldl((acc, op) -> op * acc, reverse(L.ops); init=u)
Base.:\(L::ComposedDiffEqOperator, u::AbstractVector) = foldl((acc, op) -> op \ acc, L.ops; init=u)

function LinearAlgebra.mul!(v::AbstractVector, L::ComposedDiffEqOperator, u::AbstractVector)
    @assert L.isunset "cache needs to be set up to use LinearAlgebra.mul!"

    vecs = (v, L.cache..., u)
    for i in reverse(1:length(L.ops))
        mul!(vecs[i], L.ops[i], vecs[i+1])
    end
    v
end

function LinearAlgebra.ldiv!(v::AbstractVector, L::ComposedDiffEqOperator, u::AbstractVector)
    @assert L.isunset "cache needs to be set up to use LinearAlgebra.ldiv!"

    vecs = (u, reverse(L.cache)..., v)
    for i in 1:length(L.ops)
        ldiv!(vecs[i], L.ops[i], vecs[i+1])
    end
    v
end

function LinearAlgebra.ldiv!(L::ComposedDiffEqOperator, u::AbstractVector)
    @assert L.isunset "cache needs to be set up to use LinearAlgebra.ldiv!"

    for i in 1:length(L.ops)
        ldiv!(L.ops[i], vecs[i])
    end
    v
end
#
