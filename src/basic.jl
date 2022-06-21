"""
$(TYPEDEF)
"""
struct IdentityOperator{N} <: AbstractSciMLLinearOperator{Bool} end

# constructors
IdentityOperator(u::AbstractArray) = IdentityOperator{size(u,1)}()

function Base.one(A::AbstractSciMLOperator)
    @assert issquare(A)
    N = size(A, 1)
    IdentityOperator{N}()
end

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
    @eval function Base.$op(::IdentityOperator{N}, u::AbstractVecOrMat) where{N}
        @assert size(u, 1) == N
        copy(u)
    end
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, ::IdentityOperator{N}, u::AbstractVecOrMat) where{N}
    @assert size(u, 1) == N
    copy!(v, u)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, ::IdentityOperator{N}, u::AbstractVecOrMat, α, β) where{N}
    @assert size(u, 1) == N
    mul!(v, I, u, α, β)
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, ::IdentityOperator{N}, u::AbstractVecOrMat) where{N}
    @assert size(u, 1) == N
    copy!(v, u)
end

function LinearAlgebra.ldiv!(::IdentityOperator{N}, u::AbstractVecOrMat) where{N}
    @assert size(u, 1) == N
    u
end

# operator fusion with identity returns operator itself
for op in (
           :*, :∘,
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

function Base.:\(::IdentityOperator{N}, A::AbstractSciMLOperator) where{N}
    @assert size(A, 1) == N
    A
end

function Base.:/(A::AbstractSciMLOperator, ::IdentityOperator{N}) where{N}
    @assert size(A, 2) == N
    A
end

"""
$(TYPEDEF)
"""
struct NullOperator{N} <: AbstractSciMLLinearOperator{Bool} end

# constructors
NullOperator(u::AbstractArray) = NullOperator{size(u,1)}()

function Base.zero(A::AbstractSciMLOperator)
    @assert issquare(A)
    N = size(A, 1)
    NullOperator{N}()
end

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
Base.:*(::NullOperator{N}, u::AbstractVecOrMat) where{N} = (@assert size(u, 1) == N; zero(u))

function LinearAlgebra.mul!(v::AbstractVecOrMat, ::NullOperator{N}, u::AbstractVecOrMat) where{N}
    @assert size(u, 1) == size(v, 1) == N
    lmul!(false, v)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, ::NullOperator{N}, u::AbstractVecOrMat, α, β) where{N}
    @assert size(u, 1) == size(v, 1) == N
    lmul!(β, v)
end

# operator fusion, composition
for op in (
           :*, :∘,
          )
    @eval function Base.$op(::NullOperator{N}, A::AbstractSciMLOperator) where{N}
        @assert size(A, 1) == N
        NullOperator{N}()
    end

    @eval function Base.$op(A::AbstractSciMLOperator, ::NullOperator{N}) where{N}
        @assert size(A, 2) == N
        NullOperator{N}()
    end
end

# operator addition, subtraction with NullOperator returns operator itself
for op in (
           :+, :-,
          )
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
Base.one(::Type{<:AbstractSciMLOperator}) = ScalarOperator(true)
Base.zero(::Type{<:AbstractSciMLOperator}) = ScalarOperator(false)

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
              :AbstractVecOrMat,
             )
        @eval Base.$op(α::ScalarOperator, x::$T) = $op(α.val, x)
        @eval Base.$op(x::$T, α::ScalarOperator) = $op(x, α.val)
    end
    @eval Base.$op(x::ScalarOperator, y::ScalarOperator) = $op(x.val, y.val) # TODO - lazy compose instead?
end

for op in (:-, :+)
    @eval Base.$op(α::ScalarOperator, x::Number) = $op(α.val, x)
    @eval Base.$op(x::Number, α::ScalarOperator) = $op(x, α.val)
    @eval Base.$op(x::ScalarOperator, y::ScalarOperator) = $op(x.val, y.val) # TODO - lazy sum instead?
end

LinearAlgebra.lmul!(α::ScalarOperator, u::AbstractVecOrMat) = lmul!(α.val, u)
LinearAlgebra.rmul!(u::AbstractVecOrMat, α::ScalarOperator) = rmul!(u, α.val)
LinearAlgebra.mul!(v::AbstractVecOrMat, α::ScalarOperator, u::AbstractVecOrMat) = mul!(v, α.val, u)
LinearAlgebra.mul!(v::AbstractVecOrMat, α::ScalarOperator, u::AbstractVecOrMat, a, b) = mul!(v, α.val, u, a, b)
LinearAlgebra.axpy!(α::ScalarOperator, x::AbstractVecOrMat, y::AbstractVecOrMat) = axpy!(α.val, x, y)
LinearAlgebra.axpby!(α::ScalarOperator, x::AbstractVecOrMat, β::ScalarOperator, y::AbstractVecOrMat) = axpby!(α.val, x, β.val, y)
Base.abs(α::ScalarOperator) = abs(α.val)

LinearAlgebra.ldiv!(v::AbstractVecOrMat, α::ScalarOperator, u::AbstractVecOrMat) = ldiv!(v, α.val, u)
LinearAlgebra.ldiv!(α::ScalarOperator, u::AbstractVecOrMat) = ldiv!(α.val, u)

"""
    ScaledOperator

    (λ L)*(u) = λ * L(u)
"""
struct ScaledOperator{T,
                      λType<:ScalarOperator,
                      LType<:AbstractSciMLOperator,
                      C,
                     } <: AbstractSciMLOperator{T}
    λ::λType
    L::LType
    cache::C

    function ScaledOperator(λ::ScalarOperator{Tλ},
                            L::AbstractSciMLOperator{TL},
                            cache = zeros(promote_type(Tλ,TL), 1),
                           ) where{Tλ,TL}
        T = promote_type(Tλ, TL)
        new{T,typeof(λ),typeof(L),typeof(cache)}(λ, L, cache)
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
    @eval Base.$op(L::ScaledOperator) = ScaledOperator($op(L.λ), $op(L.L))
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

function cache_internals(L::ScaledOperator, u::AbstractVecOrMat)
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
    @eval Base.$op(L::ScaledOperator, x::AbstractVecOrMat) = $op(L.λ, $op(L.L, x))
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::ScaledOperator, u::AbstractVecOrMat)
    mul!(v, L.L, u, L.λ.val, false)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::ScaledOperator, u::AbstractVecOrMat, α, β)
    mul!(L.cache, [L.λ.val,], [α,])
    mul!(v, L.L, u, first(L.cache), β)
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::ScaledOperator, u::AbstractVecOrMat)
    ldiv!(v, L.L, u)
    ldiv!(L.λ, v)
end

function LinearAlgebra.ldiv!(L::ScaledOperator, u::AbstractVecOrMat)
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

function cache_internals(L::AddedOperator, u::AbstractVecOrMat)
    for i=1:length(L.ops)
        @set! L.ops[i] = cache_operator(L.ops[i], u)
    end
    L
end

getindex(L::AddedOperator, i::Int) = sum(op -> op[i], L.ops)
getindex(L::AddedOperator, I::Vararg{Int, N}) where {N} = sum(op -> op[I...], L.ops)

function Base.:*(L::AddedOperator, u::AbstractVecOrMat)
    sum(op -> iszero(op) ? similar(u, Bool) * false : op * u, L.ops)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::AddedOperator, u::AbstractVecOrMat)
    mul!(v, first(L.ops), u)
    for op in L.ops[2:end]
        iszero(op) && continue
        mul!(v, op, u, true, true)
    end
    v
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::AddedOperator, u::AbstractVecOrMat, α, β)
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

Base.:*(ops::AbstractSciMLOperator...) = ComposedOperator(ops...)
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
Base.:*(L::ComposedOperator, u::AbstractVecOrMat) = foldl((acc, op) -> op * acc, reverse(L.ops); init=u)
Base.:\(L::ComposedOperator, u::AbstractVecOrMat) = foldl((acc, op) -> op \ acc, L.ops; init=u)

function cache_self(L::ComposedOperator, u::AbstractVecOrMat)
    vec = similar(u)
    cache = (vec,)
    for i in reverse(2:length(L.ops))
        vec   = L.ops[i] * vec
        cache = (vec, cache...)
    end

    @set! L.cache = cache
    L
end

function cache_internals(L::ComposedOperator, u::AbstractVecOrMat)
    if !(L.isset)
        L = cache_self(L, u)
    end

    vecs = L.cache
    for i in reverse(1:length(L.ops))
        @set! L.ops[i] = cache_operator(L.ops[i], vecs[i])
    end

    L
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::ComposedOperator, u::AbstractVecOrMat)
    @assert L.isset "cache needs to be set up for operator of type $(typeof(L)).
    set up cache by calling cache_operator(L::AbstractSciMLOperator, u::AbstractArray)"

    vecs = (v, L.cache[1:end-1]..., u)
    for i in reverse(1:length(L.ops))
        mul!(vecs[i], L.ops[i], vecs[i+1])
    end
    v
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::ComposedOperator, u::AbstractVecOrMat, α, β)
    @assert L.isset "cache needs to be set up for operator of type $(typeof(L)).
    set up cache by calling cache_operator(L::AbstractSciMLOperator, u::AbstractArray)"

    cache = L.cache[end]
    copy!(cache, v)

    mul!(v, L, u)
    lmul!(α, v)
    axpy!(β, cache, v)
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::ComposedOperator, u::AbstractVecOrMat)
    @assert L.isset "cache needs to be set up for operator of type $(typeof(L)).
    set up cache by calling cache_operator(L::AbstractSciMLOperator, u::AbstractArray)"

    vecs = (u, reverse(L.cache[1:end-1])..., v)
    for i in 1:length(L.ops)
        ldiv!(vecs[i+1], L.ops[i], vecs[i])
    end
    v
end

function LinearAlgebra.ldiv!(L::ComposedOperator, u::AbstractVecOrMat)

    for i in 1:length(L.ops)
        ldiv!(L.ops[i], u)
    end
    u
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

Base.:*(L::InvertedOperator, u::AbstractVecOrMat) = L.L \ u
Base.:\(L::InvertedOperator, u::AbstractVecOrMat) = L.L * u

function cache_self(L::InvertedOperator, u::AbstractVecOrMat)
    cache = similar(u)
    @set! L.cache = cache
    L
end

function cache_internals(L::InvertedOperator, u::AbstractVecOrMat)
    @set! L.L = cache_operator(L.L, u)
    L
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::InvertedOperator, u::AbstractVecOrMat)
    ldiv!(v, L.L, u)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::InvertedOperator, u::AbstractVecOrMat, α, β)
    @assert L.isset "cache needs to be set up for operator of type $(typeof(L)).
    set up cache by calling cache_operator(L::AbstractSciMLOperator, u::AbstractArray)"

    copy!(L.cache, v)
    ldiv!(v, L.L, u)
    lmul!(α, v)
    axpy!(β, L.cache, v)
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::InvertedOperator, u)
    mul!(v, L.L, u)
end

function LinearAlgebra.ldiv!(L::InvertedOperator, u::AbstractVecOrMat)
    @assert L.isset "cache needs to be set up for operator of type $(typeof(L)).
    set up cache by calling cache_operator(L::AbstractSciMLOperator, u::AbstractArray)"

    copy!(L.cache, u)
    mul!(u, L.L, L.cache)
end
#
