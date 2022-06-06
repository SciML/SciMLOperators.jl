"""
$(TYPEDEF)
"""
struct IdentityOperator{N} <: AbstractSciMLLinearOperator{Bool} end

# constructors
IdentityOperator(u::AbstractVector) = IdentityOperator{length(u)}()

function Base.one(A::AbstractSciMLOperator)
    @assert issquare(A)
    N = size(A, 1)
    IdentityOperator{N}()
end

# TODO - sparse diagonal
Base.convert(::Type{AbstractMatrix}, ::IdentityOperator{N}) where{N} = Diagonal(ones(Bool, N))

# traits
Base.size(::IdentityOperator{N}) where{N} = (N, N)
Base.adjoint(A::IdentityOperator) = A
Base.transpose(A::IdentityOperator) = A
LinearAlgebra.opnorm(::IdentityOperator{N}, p::Real=2) where{N} = true
for pred in (
             :issymmetric, :ishermitian, :isposdef,
            )
    @eval LinearAlgebra.$pred(::IdentityOperator) = true
end

getops(::IdentityOperator) = ()
isconstant(::IdentityOperator) = true
islinear(L::IdentityOperator) = true
has_adjoint(::IdentityOperator) = true
has_mul!(::IdentityOperator) = true
has_ldiv(::IdentityOperator) = true
has_ldiv!(::IdentityOperator) = true

# opeator application
for op in (
           :*, :\,
          )
    @eval function Base.$op(::IdentityOperator{N}, x::AbstractVector) where{N}
        @assert length(x) == N
        copy(x)
    end
end

function LinearAlgebra.mul!(v::AbstractVector, ::IdentityOperator{N}, u::AbstractVector) where{N}
    @assert length(u) == N
    copy!(v, u)
end

function LinearAlgebra.mul!(v::AbstractVector, ::IdentityOperator{N}, u::AbstractVector, α, β) where{N}
    @assert length(u) == N
    mul!(v, I, u, α, β)
end

function LinearAlgebra.ldiv!(v::AbstractVector, ::IdentityOperator{N}, u::AbstractArray) where{N}
    @assert length(u) == N
    copy!(v, u)
end

function LinearAlgebra.ldiv!(::IdentityOperator{N}, u::AbstractArray) where{N}
    @assert length(u) == N
    u
end

# operator fusion, composition
for op in (
           :*, :∘, :/, :\,
          )
    @eval function Base.$op(::IdentityOperator{N}, A::AbstractSciMLOperator) where{N}
        @assert size(A, 1) == N
        A
    end

    @eval function Base.$op(A::AbstractSciMLOperator, ::IdentityOperator{N}) where{N}
        @assert size(A, 2) == N
        A
    end
end

"""
$(TYPEDEF)
"""
struct NullOperator{N} <: AbstractSciMLLinearOperator{Bool} end

# constructors
NullOperator(u::AbstractVector) = NullOperator{length(u)}()

function Base.zero(A::AbstractSciMLOperator)
    @assert issquare(A)
    N = size(A, 1)
    NullOperator{N}()
end

# TODO sparse diagonal
Base.convert(::Type{AbstractMatrix}, ::NullOperator{N}) where{N} = Diagonal(zeros(Bool, N))

# traits
Base.size(::NullOperator{N}) where{N} = (N, N)
Base.adjoint(A::NullOperator) = A
Base.transpose(A::NullOperator) = A
LinearAlgebra.opnorm(::NullOperator{N}, p::Real=2) where{N} = false
for pred in (
             :issymmetric, :ishermitian,
            )
    @eval LinearAlgebra.$pred(::NullOperator) = true
end
LinearAlgebra.isposdef(::NullOperator) = false

getops(::NullOperator) = ()
isconstant(::NullOperator) = true
islinear(L::NullOperator) = true
Base.iszero(::NullOperator) = true
has_adjoint(::NullOperator) = true
has_mul!(::NullOperator) = true

# opeator application
Base.:*(::NullOperator{N}, x::AbstractVector) where{N} = (@assert length(x) == N; zero(x))
Base.:*(x::AbstractVector, ::NullOperator{N}) where{N} = (@assert length(x) == N; zero(x))

function LinearAlgebra.mul!(v::AbstractVector, ::NullOperator{N}, u::AbstractVector) where{N}
    @assert length(u) == length(v) == N
    lmul!(false, v)
end

function LinearAlgebra.mul!(v::AbstractVector, ::NullOperator{N}, u::AbstractVector, α, β) where{N}
    @assert length(u) == length(v) == N
    lmul!(β, v)
end

# operator fusion, composition
for op in (:*, :∘)
    @eval function Base.$op(::NullOperator{N}, A::AbstractSciMLOperator) where{N}
        @assert size(A, 1) == N
        NullOperator{N}()
    end

    @eval function Base.$op(A::AbstractSciMLOperator, ::NullOperator{N}) where{N}
        @assert size(A, 2) == N
        NullOperator{N}()
    end
end

for op in (:+, :-)
    @eval function Base.$op(::NullOperator{N}, A::AbstractSciMLOperator) where{N}
        @assert size(A) == (N, N)
        A
    end

    @eval function Base.$op(A::AbstractSciMLOperator, ::NullOperator{N}) where{N}
        @assert size(A) == (N, N)
        A
    end
end

"""
    ScalarOperator(val[; update_func])

    (α::ScalarOperator)(a::Number) = α * a

Represents a time-dependent scalar/scaling operator. The update function
is called by `update_coefficients!` and is assumed to have the following
signature:

    update_func(oldval,u,p,t) -> newval
"""
struct ScalarOperator{T<:Number,F} <: AbstractSciMLLinearOperator{T}
    val::T
    update_func::F
    ScalarOperator(val::T; update_func=DEFAULT_UPDATE_FUNC) where{T} =
        new{T,typeof(update_func)}(val, update_func)
end

# constructors
Base.convert(::Type{Number}, α::ScalarOperator) = α.val
Base.convert(::Type{ScalarOperator}, α::Number) = ScalarOperator(α)

ScalarOperator(α::ScalarOperator) = α
ScalarOperator(λ::UniformScaling) = ScalarOperator(λ.λ)

# traits
Base.size(α::ScalarOperator) = ()
function Base.adjoint(α::ScalarOperator) # TODO - test
    val = α.val'
    update_func =  (oldval,u,p,t) -> α.update_func(oldval',u,p,t)'
    ScalarOperator(val; update_func=update_func)
end
Base.transpose(α::ScalarOperator) = α

getops(α::ScalarOperator) = (α.val,)
islinear(L::ScalarOperator) = true
issquare(L::ScalarOperator) = true
isconstant(α::ScalarOperator) = α.update_func == DEFAULT_UPDATE_FUNC
Base.iszero(α::ScalarOperator) = iszero(α.val)
has_adjoint(::ScalarOperator) = true
has_mul(::ScalarOperator) = true
has_mul!(::ScalarOperator) = true
has_ldiv(α::ScalarOperator) = iszero(α.val)
has_ldiv!(α::ScalarOperator) = iszero(α.val)

for op in (
           :*, :/, :\,
          )
    for T in (
              :Number,
              :AbstractVector,
             )
        @eval Base.$op(α::ScalarOperator, x::$T) = $op(α.val, x)
        @eval Base.$op(x::$T, α::ScalarOperator) = $op(x, α.val)
    end
    # TODO should result be Number or ScalarOperator
    @eval Base.$op(x::ScalarOperator, y::ScalarOperator) = $op(x.val, y.val)
    #@eval function Base.$op(x::ScalarOperator, y::ScalarOperator) # TODO - test
    #    val = $op(x.val, y.val)
    #    update_func = (oldval,u,p,t) -> x.update_func(oldval,u,p,t) * y.update_func(oldval,u,p,t)
    #    ScalarOperator(val; update_func=update_func)
    #end
end

for op in (:-, :+)
    @eval Base.$op(α::ScalarOperator, x::Number) = $op(α.val, x)
    @eval Base.$op(x::Number, α::ScalarOperator) = $op(x, α.val)
    # TODO - should result be Number or ScalarOperator?
    @eval Base.$op(x::ScalarOperator, y::ScalarOperator) = $op(x.val, y.val)
end

LinearAlgebra.lmul!(α::ScalarOperator, u::AbstractVector) = lmul!(α.val, u)
LinearAlgebra.rmul!(u::AbstractVector, α::ScalarOperator) = rmul!(u, α.val)
LinearAlgebra.mul!(v::AbstractVector, α::ScalarOperator, u::AbstractVector) = mul!(v, α.val, u)
LinearAlgebra.mul!(v::AbstractVector, α::ScalarOperator, u::AbstractVector, a, b) = mul!(v, α.val, u, a, b)
LinearAlgebra.axpy!(α::ScalarOperator, x::AbstractVector, y::AbstractVector) = axpy!(α.val, x, y)
LinearAlgebra.axpby!(α::ScalarOperator, x::AbstractVector, β::ScalarOperator, y::AbstractVector) = axpby!(α.val, x, β.val, y)
Base.abs(α::ScalarOperator) = abs(α.val)

LinearAlgebra.ldiv!(v::AbstractVector, α::ScalarOperator, u::AbstractVector) = ldiv!(v, α.val, u)
LinearAlgebra.ldiv!(α::ScalarOperator, u::AbstractVector) = ldiv!(α.val, u)

"""
    ScaledOperator

    (λ L)*(u) = λ * L(u)
"""
struct ScaledOperator{T,
                      λType<:ScalarOperator,
                      LType<:AbstractSciMLOperator,
                     } <: AbstractSciMLOperator{T}
    λ::λType
    L::LType
    cache::T

    function ScaledOperator(λ::ScalarOperator{Tλ},
                            L::AbstractSciMLOperator{TL},
                            cache = zero(promote_type(Tλ,TL))
                           ) where{Tλ,TL}
        T = promote_type(Tλ, TL)
        new{T,typeof(λ),typeof(L)}(λ, L, cache)
    end
end

ScalingNumberTypes = (
                      :ScalarOperator,
                      :Number,
                      :UniformScaling,
                     )

# constructors
for T in ScalingNumberTypes[2:end]
    @eval ScaledOperator(λ::$T, L::AbstractSciMLOperator) = ScaledOperator(ScalarOperator(λ), L)
end

for T in ScalingNumberTypes
    @eval function ScaledOperator(λ::$T, L::ScaledOperator)
        λ = ScalarOperator(λ) * L.λ
        ScaledOperator(λ, L.L)
    end
    
    @eval Base.:*(λ::$T, L::AbstractSciMLOperator) = ScaledOperator(λ, L)
    @eval Base.:*(L::AbstractSciMLOperator, λ::$T) = ScaledOperator(λ, L)
    @eval Base.:\(λ::$T, L::AbstractSciMLOperator) = ScaledOperator(inv(λ), L)
    @eval Base.:\(L::AbstractSciMLOperator, λ::$T) = ScaledOperator(λ, inv(L))
    @eval Base.:/(L::AbstractSciMLOperator, λ::$T) = ScaledOperator(inv(λ), L)
    @eval Base.:/(λ::$T, L::AbstractSciMLOperator) = ScaledOperator(λ, inv(L))
end

Base.:-(L::AbstractSciMLOperator) = ScaledOperator(-true, L)
Base.:+(L::AbstractSciMLOperator) = L

Base.convert(::Type{AbstractMatrix}, L::ScaledOperator) = L.λ.val * convert(AbstractMatrix, L.L)
SparseArrays.sparse(L::ScaledOperator) = L.λ * sparse(L.L)

# traits
Base.size(L::ScaledOperator) = size(L.L)
for op in (
           :adjoint,
           :transpose,
          )
    @eval Base.$op(L::ScaledOperator) = ScaledOperator($op(L.λ), $op(L.op))
end
LinearAlgebra.opnorm(L::ScaledOperator, p::Real=2) = abs(L.λ) * opnorm(L.L, p)

getops(L::ScaledOperator) = (L.λ, L.L,)
islinear(L::ScaledOperator) = all(islinear, L.ops)
isconstant(L::ScaledOperator) = isconstant(L.L) & isconstant(L.λ)
Base.iszero(L::ScaledOperator) = iszero(L.L) | iszero(L.λ)
has_adjoint(L::ScaledOperator) = has_adjoint(L.L)
has_mul!(L::ScaledOperator) = has_mul!(L.L)
has_ldiv(L::ScaledOperator) = has_ldiv(L.L) & !iszero(L.λ)
has_ldiv!(L::ScaledOperator) = has_ldiv!(L.L) & !iszero(L.λ)

function cache_internals(L::ScaledOperator, u::AbstractVector)
    @set! L.L = cache_operator(L.L, u)
    @set! L.λ = cache_operator(L.λ, u)
    L
end

# getindex
Base.getindex(L::ScaledOperator, i::Int) = L.coeff * L.op[i]
Base.getindex(L::ScaledOperator, I::Vararg{Int, N}) where {N} = L.λ * L.L[I...]
for fact in (
             :lu, :lu!,
             :qr, :qr!,
             :cholesky, :cholesky!,
             :ldlt, :ldlt!,
             :bunchkaufman, :bunchkaufman!,
             :lq, :lq!,
             :svd, :svd!,
            )
    @eval LinearAlgebra.$fact(L::ScaledOperator, args...) = L.λ * fact(L.L, args...)
end

# operator application, inversion
for op in (
           :*, :\,
          )
    @eval Base.$op(L::ScaledOperator, x::AbstractVector) = $op(L.λ, $op(L.L, x))
end

function LinearAlgebra.mul!(v::AbstractVector, L::ScaledOperator, u::AbstractVector)
    mul!(v, L.L, u, L.λ.val, false)
end

function LinearAlgebra.mul!(v::AbstractVector, L::ScaledOperator, u::AbstractVector, α, β)
    cache = L.λ.val * α # mul!(L.cache, L.λ.val, α) # TODO - set L.cache as 
    mul!(v, L.L, u, cache, β)
end

function LinearAlgebra.ldiv!(v::AbstractVector, L::ScaledOperator, u::AbstractVector)
    ldiv!(v, L.L, u)
    ldiv!(L.λ, v)
end

function LinearAlgebra.ldiv!(L::ScaledOperator, u::AbstractVector)
    ldiv!(L.λ, u)
    ldiv!(L.L, u)
end

"""
Lazy operator addition

    (A1 + A2 + A3...)u = A1*u + A2*u + A3*u ....
"""
struct AddedOperator{T,
                     O<:Tuple{Vararg{AbstractSciMLOperator}},
                    } <: AbstractSciMLOperator{T}
    ops::O

    function AddedOperator(ops)
        T = promote_type(eltype.(ops)...)
        new{T,typeof(ops)}(ops)
    end
end

function AddedOperator(ops::AbstractSciMLOperator...)
    sz = size(first(ops))
    for op in ops[2:end]
        @assert size(op) == sz "Size mismatich in operators $ops"
    end
    AddedOperator(ops)
end

AddedOperator(L::AbstractSciMLOperator) = L

# constructors
Base.:+(ops::AbstractSciMLOperator...) = AddedOperator(ops...)
Base.:-(A::AbstractSciMLOperator, B::AbstractSciMLOperator) = AddedOperator(A, -B)

for op in (
           :+, :-,
          )

    @eval Base.$op(A::AddedOperator, B::AddedOperator) = AddedOperator(A.ops..., $op(B).ops...)
    @eval Base.$op(A::AbstractSciMLOperator, B::AddedOperator) = AddedOperator(A, $op(B).ops...)
    @eval Base.$op(A::AddedOperator, B::AbstractSciMLOperator) = AddedOperator(A.ops..., $op(B))

    for T in ScalingNumberTypes
        @eval function Base.$op(L::AbstractSciMLOperator, λ::$T)
            @assert issquare(L)
            N  = size(L, 1)
            Id = IdentityOperator{N}()
            AddedOperator(L, $op(λ)*Id)
        end

        @eval function Base.$op(λ::$T, L::AbstractSciMLOperator)
            @assert issquare(L)
            N  = size(L, 1)
            Id = IdentityOperator{N}()
            AddedOperator(λ*Id, $op(L))
        end
    end

    @eval function Base.$op(A::AbstractMatrix, L::AbstractSciMLOperator)
        @assert size(A) == size(L)
        AddedOperator(MatrixOperator(A), $op(L))
    end

    @eval function Base.$op(L::AbstractSciMLOperator, A::AbstractMatrix)
        @assert size(A) == size(L)
        AddedOperator(L, MatrixOperator($op(A)))
    end
end

Base.convert(::Type{AbstractMatrix}, L::AddedOperator) = sum(op -> convert(AbstractMatrix, op), L.ops)
SparseArrays.sparse(L::AddedOperator) = sum(_sparse, L.ops)

# traits
Base.size(L::AddedOperator) = size(first(L.ops))
for op in (
           :adjoint,
           :transpose,
          )
    @eval Base.$op(L::AddedOperator) = AddedOperator($op.(L.ops)...)
end

getops(L::AddedOperator) = L.ops
Base.iszero(L::AddedOperator) = all(iszero, getops(L))
has_adjoint(L::AddedOperator) = all(has_adjoint, L.ops)

function cache_internals(L::AddedOperator, u::AbstractVector)
    for i=1:length(L.ops)
        @set! L.ops[i] = cache_operator(L.ops[i], u)
    end
    L
end

getindex(L::AddedOperator, i::Int) = sum(op -> op[i], L.ops)
getindex(L::AddedOperator, I::Vararg{Int, N}) where {N} = sum(op -> op[I...], L.ops)

function Base.:*(L::AddedOperator, u::AbstractVector)
    sum(op -> iszero(op) ? similar(u, Bool) * false : op * u, L.ops)
end

function LinearAlgebra.mul!(v::AbstractVector, L::AddedOperator, u::AbstractVector)
    mul!(v, first(L.ops), u)
    for op in L.ops[2:end]
        iszero(op) && continue
        mul!(v, op, u, true, true)
    end
    v
end

function LinearAlgebra.mul!(v::AbstractVector, L::AddedOperator, u::AbstractVector, α, β)
    lmul!(β, v)
    for op in L.ops
        iszero(op) && continue
        mul!(v, op, u, α, true)
    end
    v
end

"""
    Lazy operator composition

    ∘(A, B, C)(u) = A(B(C(u)))

    ops = (A, B, C)
    cache = (B*C*u , C*u)
"""
struct ComposedOperator{T,O,C} <: AbstractSciMLOperator{T}
    """ Tuple of N operators to be applied in reverse"""
    ops::O
    """ cache for 3 and 5 argument mul! """
    cache::C
    """ is cache set """
    isset::Bool

    function ComposedOperator(ops, cache, isset::Bool)
        for i in reverse(2:length(ops))
            opcurr = ops[i]
            opnext = ops[i-1]
            @assert size(opcurr, 1) == size(opnext, 2) "Cannot $opnext ∘ $opcurr. Size mismatich"
        end

        T = promote_type(eltype.(ops)...)
        isset = cache !== nothing
        new{T,typeof(ops),typeof(cache)}(ops, cache, isset)
    end
end

function ComposedOperator(ops::AbstractSciMLOperator...; cache = nothing)
    isset = cache !== nothing
    ComposedOperator(ops, cache, isset)
end

# constructors
Base.:∘(ops::AbstractSciMLOperator...) = ComposedOperator(ops...)
Base.:∘(A::ComposedOperator, B::ComposedOperator) = ComposedOperator(A.ops..., B.ops...)
Base.:∘(A::AbstractSciMLOperator, B::ComposedOperator) = ComposedOperator(A, B.ops...)
Base.:∘(A::ComposedOperator, B::AbstractSciMLOperator) = ComposedOperator(A.ops..., B)

# operator fusion falls back on composition
Base.:*(ops::AbstractSciMLOperator...) = reduce(*, ops) # pairwise fusion
Base.:*(A::AbstractSciMLOperator, B::AbstractSciMLOperator) = ∘(A, B)
Base.:*(A::ComposedOperator, B::AbstractSciMLOperator) = ∘(A.ops[1:end-1]..., A.ops[end] * B)
Base.:*(A::AbstractSciMLOperator, B::ComposedOperator) = ∘(A * B.ops[1], B.ops[2:end]...)

Base.convert(::Type{AbstractMatrix}, L::ComposedOperator) = prod(op -> convert(AbstractMatrix, op), L.ops)
SparseArrays.sparse(L::ComposedOperator) = prod(_sparse, L.ops)

# traits
Base.size(L::ComposedOperator) = (size(first(L.ops), 1), size(last(L.ops),2))
for op in (
           :adjoint,
           :transpose,
          )
    @eval Base.$op(L::ComposedOperator) = ComposedOperator($op.(reverse(L.ops))...)
end
LinearAlgebra.opnorm(L::ComposedOperator) = prod(opnorm, L.ops)

getops(L::ComposedOperator) = L.ops
islinear(L::ComposedOperator) = all(islinear, L.ops)
Base.iszero(L::ComposedOperator) = all(iszero, getops(L))
has_adjoint(L::ComposedOperator) = all(has_adjoint, L.ops)
has_mul!(L::ComposedOperator) = all(has_mul!, L.ops)
has_ldiv(L::ComposedOperator) = all(has_ldiv, L.ops)
has_ldiv!(L::ComposedOperator) = all(has_ldiv!, L.ops)

factorize(L::ComposedOperator) = prod(factorize, reverse(L.ops))
for fact in (
             :lu, :lu!,
             :qr, :qr!,
             :cholesky, :cholesky!,
             :ldlt, :ldlt!,
             :bunchkaufman, :bunchkaufman!,
             :lq, :lq!,
             :svd, :svd!,
            )
    @eval LinearAlgebra.$fact(L::ComposedOperator, args...) = prod(op -> $fact(op, args...), reverse(L.ops))
end

# operator application
Base.:*(L::ComposedOperator, u::AbstractVector) = foldl((acc, op) -> op * acc, reverse(L.ops); init=u)
Base.:\(L::ComposedOperator, u::AbstractVector) = foldl((acc, op) -> op \ acc, L.ops; init=u)

function cache_self(L::ComposedOperator, u::AbstractVector)
    vec = similar(u)
    cache = (vec,)
    for i in reverse(2:length(L.ops))
        vec   = L.ops[i] * vec
        cache = (vec, cache...)
    end

    @set! L.cache = cache
    L
end

function cache_internals(L::ComposedOperator, u::AbstractVector)
    if !(L.isset)
        L = cache_self(L, u)
    end

    vecs = L.cache
    for i in reverse(1:length(L.ops))
        @set! L.ops[i] = cache_operator(L.ops[i], vecs[i])
    end

    L
end

function LinearAlgebra.mul!(v::AbstractVector, L::ComposedOperator, u::AbstractVector)
    @assert L.isset "cache needs to be set up to use LinearAlgebra.mul!"

    vecs = (v, L.cache[1:end-1]..., u)
    for i in reverse(1:length(L.ops))
        mul!(vecs[i], L.ops[i], vecs[i+1])
    end
    v
end

function LinearAlgebra.mul!(v::AbstractVector, L::ComposedOperator, u::AbstractVector, α, β)
    @assert L.isset "cache needs to be set up to use LinearAlgebra.mul!"

    cache = L.cache[end]
    copy!(cache, v)

    mul!(v, L, u)
    lmul!(α, v)
    axpy!(β, cache, v)
end

function LinearAlgebra.ldiv!(v::AbstractVector, L::ComposedOperator, u::AbstractVector)
    @assert L.isset "cache needs to be set up to use 3 arg LinearAlgebra.ldiv!"

    vecs = (u, reverse(L.cache[1:end-1])..., v)
    for i in 1:length(L.ops)
        ldiv!(vecs[i+1], L.ops[i], vecs[i])
    end
    v
end

function LinearAlgebra.ldiv!(L::ComposedOperator, u::AbstractVector)

    for i in 1:length(L.ops)
        ldiv!(L.ops[i], u)
    end
    u
end

struct AdjointedOperator{T,LType} <: AbstractSciMLOperator{T}
    L::LType

    function AdjointedOperator(L::AbstractSciMLOperator{T}) where{T}
        new{T,typeof(L)}(L)
    end
end

struct TransposedOperator{T,LType} <: AbstractSciMLOperator{T}
    L::LType

    function TransposedOperator(L::AbstractSciMLOperator{T}) where{T}
        new{T,typeof(L)}(L)
    end
end

AbstractAdjointedVector  = Adjoint{  <:Number, <:AbstractVector}
AbstractTransposedVector = Transpose{<:Number, <:AbstractVector}

has_adjoint(::AdjointedOperator) = true

Base.transpose(L::AdjointedOperator) = conj(L.L)
Base.adjoint(L::TransposedOperator) = conj(L.L)

for (op, LType, VType) in (
                           (:adjoint,   :AdjointedOperator,  :AbstractAdjointedVector ),
                           (:transpose, :TransposedOperator, :AbstractTransposedVector),
                          )
    # constructor
    @eval Base.$op(L::AbstractSciMLOperator) = $LType(L)

    @eval Base.convert(::Type{AbstractMatrix}, L::$LType) = $op(convert(AbstractMatrix, L.L))

    # traits
    @eval Base.size(L::$LType) = size(L.L) |> reverse
    @eval Base.$op(L::$LType) = L.L

    @eval getops(L::$LType) = (L.L,)

    @eval @forward $LType.L (
                             # LinearAlgebra
                             LinearAlgebra.issymmetric,
                             LinearAlgebra.ishermitian,
                             LinearAlgebra.isposdef,
                             LinearAlgebra.opnorm,

                             # SciML
                             isconstant,
                             has_mul!,
                             has_ldiv,
                             has_ldiv!,
                            )

    # oeprator application
    @eval Base.:*(u::$VType, L::$LType) = $op(L.L * u.parent)
    @eval Base.:/(u::$VType, L::$LType) = $op(L.L \ u.parent)

    # v' ← u' * A'
    # v  ← A  * u
    @eval function LinearAlgebra.mul!(v::$VType, u::$VType, L::$LType)
        mul!(v.parent, L.L, u.parent)
        v
    end

    # v' ← α * (u' * A') + β * v'
    # v  ← α * (A  * u ) + β * v
    @eval function LinearAlgebra.mul!(v::$VType, u::$VType, L::$LType, α, β)
        mul!(v.parent, L.L, u.parent, α, β)
        v
    end

    # v' ← u' / A'
    # v  ← A  \ u
    @eval function LinearAlgebra.ldiv!(v::$VType, u::$VType, L::$LType)
        ldiv!(v.parent, L.L, u.parent)
        v
    end
    
    # u' ← u' / A'
    # u  ← A  \ u
    @eval function LinearAlgebra.ldiv!(u::$VType, L::$LType)
        ldiv!(L.L, u.parent)
        u
    end
end

"""
    Lazy Operator Inverse
"""
struct InvertedOperator{T, LType, C} <: AbstractSciMLOperator{T}
    L::LType
    cache::C
    isset::Bool

    function InvertedOperator(L::AbstractSciMLOperator{T}, cache, isset) where{T}
        isset = cache !== nothing
        new{T,typeof(L),typeof(cache)}(L, cache, isset)
    end
end

function InvertedOperator(L::AbstractSciMLOperator{T}; cache=nothing) where{T}
    isset = cache !== nothing
    InvertedOperator(L, cache, isset)
end

Base.inv(L::AbstractSciMLOperator) = InvertedOperator(L)
Base.convert(::Type{AbstractMatrix}, L::InvertedOperator) = inv(convert(AbstractMatrix, L.L))

Base.size(L::InvertedOperator) = size(L.L) |> reverse
Base.adjoint(L::InvertedOperator) = InvertedOperator(L.L')

getops(L::InvertedOperator) = (L.L,)

has_mul!(L::InvertedOperator) = has_ldiv!(L.L)
has_ldiv(L::InvertedOperator) = has_mul(L.L)
has_ldiv!(L::InvertedOperator) = has_mul!(L.L)

@forward InvertedOperator.L (
                             # LinearAlgebra
                             LinearAlgebra.issymmetric,
                             LinearAlgebra.ishermitian,
                             LinearAlgebra.isposdef,
                             LinearAlgebra.opnorm,

                             # SciML
                             isconstant,
                             has_adjoint,
                            )

Base.:*(L::InvertedOperator, u::AbstractVector) = L.L \ u
Base.:\(L::InvertedOperator, u::AbstractVector) = L.L * u

function cache_self(L::InvertedOperator, u::AbstractVector)
    cache = similar(u)
    @set! L.cache = cache
    L
end

function cache_internals(L::InvertedOperator, u::AbstractVector)
    @set! L.L = cache_operator(L.L, u)
    L
end

function LinearAlgebra.mul!(v::AbstractVector, L::InvertedOperator, u::AbstractVector)
    ldiv!(v, L.L, u)
end

function LinearAlgebra.mul!(v::AbstractVector, L::InvertedOperator, u::AbstractVector, α, β)
    @assert L.isset "cache needs to be set up to use 5 arg LinearAlgebra.ldiv!"
    copy!(L.cache, v)
    ldiv!(v, L.L, u)
    lmul!(α, v)
    axpy!(β, L.cache, v)
end

function LinearAlgebra.ldiv!(v::AbstractVector, L::InvertedOperator, u)
    mul!(v, L.L, u)
end

function LinearAlgebra.ldiv!(L::InvertedOperator, u::AbstractVector)
    @assert L.isset "cache needs to be set up to use 2 arg LinearAlgebra.ldiv!"
    copy!(L.cache, u)
    mul!(u, L.L, L.cache)
end
#
