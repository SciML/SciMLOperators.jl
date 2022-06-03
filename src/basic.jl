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

Base.convert(::Type{AbstractMatrix}, ::IdentityOperator{N}) where{N} = Diagonal(ones(Bool, N))

# traits
Base.size(::IdentityOperator{N}) where{N} = (N, N)
Base.adjoint(A::IdentityOperator) = A
LinearAlgebra.opnorm(::IdentityOperator{N}, p::Real=2) where{N} = true
for pred in (
             :isreal, :issymmetric, :ishermitian, :isposdef,
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

function LinearAlgebra.mul!(v::AbstractVector, ::IdentityOperator{N}, u::AbstractVector, α::Number, β::Number) where{N}
    @assert length(u) == N
    mul!(v, I, u, α, β)
end

function LinearAlgebra.ldiv!(v::AbstractVector, ::IdentityOperator{N}, u::AbstractArray) where{N}
    @assert length(u) == N
    copy!(v, u)
end

# operator fusion, composition
for op in (
           :*, :∘, :/, :\,
          )
    @eval function Base.$op(::IdentityOperator{N}, A::AbstractSciMLOperator) where{N}
        @assert size(A, 1) == N
        IdentityOperator{N}()
    end

    @eval function Base.$op(A::AbstractSciMLOperator, ::IdentityOperator{N}) where{N}
        @assert size(A, 2) == N
        IdentityOperator{N}()
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

Base.convert(::Type{AbstractMatrix}, ::NullOperator{N}) where{N} = Diagonal(zeros(Bool, N))

# traits
Base.size(::NullOperator{N}) where{N} = (N, N)
Base.adjoint(A::NullOperator) = A
LinearAlgebra.opnorm(::NullOperator{N}, p::Real=2) where{N} = false
for pred in (
             :isreal, :issymmetric, :ishermitian,
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

getops(α::ScalarOperator) = (α.val)
islinear(L::ScalarOperator) = true
isconstant(α::ScalarOperator) = α.update_func == DEFAULT_UPDATE_FUNC
Base.iszero(α::ScalarOperator) = iszero(α.val)
has_adjoint(::ScalarOperator) = true
has_mul(::ScalarOperator) = true
has_ldiv(α::ScalarOperator) = iszero(α.val)

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

LinearAlgebra.lmul!(α::ScalarOperator, B::AbstractVector) = lmul!(α.val, B)
LinearAlgebra.rmul!(B::AbstractVector, α::ScalarOperator) = rmul!(B, α.val)
LinearAlgebra.mul!(Y::AbstractVector, α::ScalarOperator, B::AbstractVector) = mul!(Y, α.val, B)
LinearAlgebra.axpy!(α::ScalarOperator, X::AbstractVector, Y::AbstractVector) = axpy!(α.val, X, Y)
Base.abs(α::ScalarOperator) = abs(α.val)

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

    function ScaledOperator(λ::ScalarOperator, L::AbstractSciMLOperator)
        T = promote_type(eltype.((λ, L))...)
        new{T,typeof(λ),typeof(L)}(λ, L)
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
Base.adjoint(L::ScaledOperator) = ScaledOperator(L.λ', L.op')
LinearAlgebra.opnorm(L::ScaledOperator, p::Real=2) = abs(L.λ) * opnorm(L.L, p)

getops(L::ScaledOperator) = (L.λ, L.L)
islinear(L::ScaledOperator) = all(islinear, L.ops)
isconstant(L::ScaledOperator) = isconstant(L.L) & isconstant(L.λ)
Base.iszero(L::ScaledOperator) = iszero(L.L) | iszero(L.λ)
has_adjoint(L::ScaledOperator) = has_adjoint(L.L)
has_mul!(L::ScaledOperator) = has_mul!(L.L)
has_ldiv(L::ScaledOperator) = has_ldiv(L.L) & !iszero(L.λ)
has_ldiv!(L::ScaledOperator) = has_ldiv!(L.L) & !iszero(L.λ)

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
    mul!(v, L.L, u)
    lmul!(L.λ, v)
end

"""
Lazy operator addition (A + B)

    (A1 + A2 + A3...)u = A1*u + A2*u + A3*u ....
"""
struct AddedOperator{T,
                     O<:Tuple{Vararg{AbstractSciMLOperator}},
                     C,
                    } <: AbstractSciMLOperator{T}
    ops::O
    cache::C
    isunset::Bool

    function AddedOperator(ops...; cache = nothing)
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
Base.:+(ops::AbstractSciMLOperator...) = AddedOperator(ops...)

Base.:-(L::AddedOperator) = AddedOperator(.-(A.ops)...)
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
function Base.adjoint(L::AddedOperator)
    if issquare(L) & !(L.isunset)
        AddedOperator(adjoint.(L.ops)...,L.cache, L.isunset)
    else
        AddedOperator(adjoint.(L.ops)...)
    end
end

getops(L::AddedOperator) = L.ops
Base.iszero(L::AddedOperator) = all(iszero, getops(L))
has_adjoint(L::AddedOperator) = all(has_adjoint, L.ops)

getindex(L::AddedOperator, i::Int) = sum(op -> op[i], L.ops)
getindex(L::AddedOperator, I::Vararg{Int, N}) where {N} = sum(op -> op[I...], L.ops)

function init_cache(A::AddedOperator, u::AbstractVector)
    cache = A.B * u
end

function Base.:*(L::AddedOperator, u::AbstractVector)
    sum(op -> iszero(op) ? similar(u, Bool) * false : op * u, L.ops)
end

function LinearAlgebra.mul!(v::AbstractVector, L::AddedOperator, u::AbstractVector)
    mul!(v, first(L.ops), u)
    for op in L.ops[2:end]
        iszero(op) && continue
        mul!(L.cache, op, u)
        axpy!(true, L.cache, v)
    end
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
    """ Tuple of N-1 cache vectors. cache[N-1] = op[N] * u and so on """
    cache::C
    isunset::Bool
    function ComposedOperator(ops::AbstractSciMLOperator...; cache = nothing)
        for i in reverse(2:length(ops))
            opcurr = ops[i]
            opnext = ops[i-1]
            @assert size(opcurr, 1) == size(opnext, 2) "Cannot $opnext ∘ $opcurr. Size mismatich"
        end

        T = promote_type(eltype.(ops)...)
        isunset = cache === nothing
        new{T,typeof(ops),typeof(cache)}(ops, cache, isunset)
    end
end

function init_cache(L::ComposedOperator, u::AbstractVector)
    cache = ()
    vec = u
    for i in reverse(2:length(L.ops))
        vec = op[i] * vec
        cache = push(cache, vec)
    end
    cache
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
Base.adjoint(L::ComposedOperator) = ComposedOperator(adjoint.(reverse(L.ops)))
LinearAlgebra.opnorm(L::ComposedOperator) = prod(opnorm, L.ops)

getops(L::ComposedOperator) = L.ops
islinear(L::ComposedOperator) = all(islinear, L.ops)
Base.iszero(L::ComposedOperator) = all(iszero, getops(L))
has_adjoint(L::ComposedOperator) = all(has_adjoint, L.ops)
has_mul!(L::ComposedOperator) = all(has_mul!, L.ops)
has_ldiv(L::ComposedOperator) = all(has_ldiv, L.ops)
has_ldiv!(L::ComposedOperator) = all(has_mul!, L.ops)

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

function LinearAlgebra.mul!(v::AbstractVector, L::ComposedOperator, u::AbstractVector)
    @assert L.isunset "cache needs to be set up to use LinearAlgebra.mul!"

    vecs = (v, L.cache..., u)
    for i in reverse(1:length(L.ops))
        mul!(vecs[i], L.ops[i], vecs[i+1])
    end
    v
end

function LinearAlgebra.ldiv!(v::AbstractVector, L::ComposedOperator, u::AbstractVector)
    @assert L.isunset "cache needs to be set up to use LinearAlgebra.ldiv!"

    vecs = (u, reverse(L.cache)..., v)
    for i in 1:length(L.ops)
        ldiv!(vecs[i], L.ops[i], vecs[i+1])
    end
    v
end

function LinearAlgebra.ldiv!(L::ComposedOperator, u::AbstractVector)
    @assert L.isunset "cache needs to be set up to use LinearAlgebra.ldiv!"

    for i in 1:length(L.ops)
        ldiv!(L.ops[i], vecs[i])
    end
    v
end
#
