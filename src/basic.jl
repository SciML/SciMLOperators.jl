"""
$(TYPEDEF)
"""
struct SciMLIdentity{N} <: AbstractSciMLLinearOperator{Bool} end

# constructors
SciMLIdentity(u::AbstractVector) = SciMLIdentity{length(u)}()

function Base.one(A::AbstractSciMLOperator)
    @assert issquare(A)
    N = size(A, 1)
    SciMLIdentity{N}()
end

Base.convert(::Type{AbstractMatrix}, ::SciMLIdentity{N}) where{N} = Diagonal(ones(Bool, N))

# traits
Base.size(::SciMLIdentity{N}) where{N} = (N, N)
Base.adjoint(A::SciMLIdentity) = A
LinearAlgebra.opnorm(::SciMLIdentity{N}, p::Real=2) where{N} = true
for pred in (
             :isreal, :issymmetric, :ishermitian, :isposdef,
            )
    @eval LinearAlgebra.$pred(::SciMLIdentity) = true
end

getops(::SciMLIdentity) = ()
isconstant(::SciMLIdentity) = true
islinear(L::SciMLIdentity) = true
has_adjoint(::SciMLIdentity) = true
has_mul!(::SciMLIdentity) = true
has_ldiv(::SciMLIdentity) = true
has_ldiv!(::SciMLIdentity) = true

# opeator application
for op in (
           :*, :\,
          )
    @eval function Base.$op(::SciMLIdentity{N}, x::AbstractVector) where{N}
        @assert length(x) == N
        copy(x)
    end
end

function LinearAlgebra.mul!(v::AbstractVector, ::SciMLIdentity{N}, u::AbstractVector) where{N}
    @assert length(u) == N
    copy!(v, u)
end

function LinearAlgebra.ldiv!(v::AbstractVector, ::SciMLIdentity{N}, u::AbstractArray) where{N}
    @assert length(u) == N
    copy!(v, u)
end

# operator fusion, composition
for op in (
           :*, :∘, :/, :\,
          )
    @eval function Base.$op(::SciMLIdentity{N}, A::AbstractSciMLOperator) where{N}
        @assert size(A, 1) == N
        SciMLIdentity{N}()
    end

    @eval function Base.$op(A::AbstractSciMLOperator, ::SciMLIdentity{N}) where{N}
        @assert size(A, 2) == N
        SciMLIdentity{N}()
    end
end

"""
$(TYPEDEF)
"""
struct SciMLNullOperator{N} <: AbstractSciMLLinearOperator{Bool} end

# constructors
SciMLNullOperator(u::AbstractVector) = SciMLNullOperator{length(u)}()

function Base.zero(A::AbstractSciMLOperator)
    @assert issquare(A)
    N = size(A, 1)
    SciMLNullOperator{N}()
end

Base.convert(::Type{AbstractMatrix}, ::SciMLNullOperator{N}) where{N} = Diagonal(zeros(Bool, N))

# traits
Base.size(::SciMLNullOperator{N}) where{N} = (N, N)
Base.adjoint(A::SciMLNullOperator) = A
LinearAlgebra.opnorm(::SciMLNullOperator{N}, p::Real=2) where{N} = false
for pred in (
             :isreal, :issymmetric, :ishermitian,
            )
    @eval LinearAlgebra.$pred(::SciMLNullOperator) = true
end
LinearAlgebra.isposdef(::SciMLNullOperator) = false

getops(::SciMLNullOperator) = ()
isconstant(::SciMLNullOperator) = true
islinear(L::SciMLNullOperator) = true
Base.iszero(::SciMLNullOperator) = true
has_adjoint(::SciMLNullOperator) = true
has_mul!(::SciMLNullOperator) = true

# opeator application
Base.:*(::SciMLNullOperator{N}, x::AbstractVector) where{N} = (@assert length(x) == N; zero(x))
Base.:*(x::AbstractVector, ::SciMLNullOperator{N}) where{N} = (@assert length(x) == N; zero(x))

function LinearAlgebra.mul!(v::AbstractVector, ::SciMLNullOperator{N}, u::AbstractVector) where{N}
    @assert length(u) == length(v) == N
    lmul!(false, v)
end

# operator fusion, composition
for op in (:*, :∘)
    @eval function Base.$op(::SciMLNullOperator{N}, A::AbstractSciMLOperator) where{N}
        @assert size(A, 1) == N
        SciMLNullOperator{N}()
    end

    @eval function Base.$op(A::AbstractSciMLOperator, ::SciMLNullOperator{N}) where{N}
        @assert size(A, 2) == N
        SciMLNullOperator{N}()
    end
end

"""
    SciMLScalar(val[; update_func])

    (α::SciMLScalar)(a::Number) = α * a

Represents a time-dependent scalar/scaling operator. The update function
is called by `update_coefficients!` and is assumed to have the following
signature:

    update_func(oldval,u,p,t) -> newval
"""
struct SciMLScalar{T<:Number,F} <: AbstractSciMLLinearOperator{T}
    val::T
    update_func::F
    SciMLScalar(val::T; update_func=DEFAULT_UPDATE_FUNC) where{T} =
        new{T,typeof(update_func)}(val, update_func)
end

# constructors
Base.convert(::Type{Number}, α::SciMLScalar) = α.val
Base.convert(::Type{SciMLScalar}, α::Number) = SciMLScalar(α)

SciMLScalar(α::SciMLScalar) = α
SciMLScalar(λ::UniformScaling) = SciMLScalar(λ.λ)

# traits
Base.size(α::SciMLScalar) = ()
function Base.adjoint(α::SciMLScalar) # TODO - test
    val = α.val'
    update_func =  (oldval,u,p,t) -> α.update_func(oldval',u,p,t)'
    SciMLScalar(val; update_func=update_func)
end

getops(α::SciMLScalar) = (α.val)
islinear(L::SciMLScalar) = true
isconstant(α::SciMLScalar) = α.update_func == DEFAULT_UPDATE_FUNC
Base.iszero(α::SciMLScalar) = iszero(α.val)
has_adjoint(::SciMLScalar) = true
has_mul(::SciMLScalar) = true
has_ldiv(α::SciMLScalar) = iszero(α.val)

for op in (
           :*, :/, :\,
          )
    for T in (
              :Number,
              :AbstractVector,
             )
        @eval Base.$op(α::SciMLScalar, x::$T) = $op(α.val, x)
        @eval Base.$op(x::$T, α::SciMLScalar) = $op(x, α.val)
    end
    # TODO should result be Number or SciMLScalar
    @eval Base.$op(x::SciMLScalar, y::SciMLScalar) = $op(x.val, y.val)
    #@eval function Base.$op(x::SciMLScalar, y::SciMLScalar) # TODO - test
    #    val = $op(x.val, y.val)
    #    update_func = (oldval,u,p,t) -> x.update_func(oldval,u,p,t) * y.update_func(oldval,u,p,t)
    #    SciMLScalar(val; update_func=update_func)
    #end
end

for op in (:-, :+)
    @eval Base.$op(α::SciMLScalar, x::Number) = $op(α.val, x)
    @eval Base.$op(x::Number, α::SciMLScalar) = $op(x, α.val)
    # TODO - should result be Number or SciMLScalar?
    @eval Base.$op(x::SciMLScalar, y::SciMLScalar) = $op(x.val, y.val)
end

LinearAlgebra.lmul!(α::SciMLScalar, B::AbstractVector) = lmul!(α.val, B)
LinearAlgebra.rmul!(B::AbstractVector, α::SciMLScalar) = rmul!(B, α.val)
LinearAlgebra.mul!(Y::AbstractVector, α::SciMLScalar, B::AbstractVector) = mul!(Y, α.val, B)
LinearAlgebra.axpy!(α::SciMLScalar, X::AbstractVector, Y::AbstractVector) = axpy!(α.val, X, Y)
Base.abs(α::SciMLScalar) = abs(α.val)

"""
    SciMLScaledOperator

    (λ L)*(u) = λ * L(u)
"""
struct SciMLScaledOperator{T,
                            λType<:SciMLScalar,
                            LType<:AbstractSciMLOperator,
                           } <: AbstractSciMLOperator{T}
    λ::λType
    L::LType

    function SciMLScaledOperator(λ::SciMLScalar, L::AbstractSciMLOperator)
        T = promote_type(eltype.((λ, L))...)
        new{T,typeof(λ),typeof(L)}(λ, L)
    end
end

ScalingNumberTypes = (
                      :SciMLScalar,
                      :Number,
                      :UniformScaling,
                     )

# constructors
for T in ScalingNumberTypes[2:end]
    @eval SciMLScaledOperator(λ::$T, L::AbstractSciMLOperator) = SciMLScaledOperator(SciMLScalar(λ), L)
end

for T in ScalingNumberTypes
    @eval function SciMLScaledOperator(λ::$T, L::SciMLScaledOperator)
        λ = SciMLScalar(λ) * L.λ
        SciMLScaledOperator(λ, L.L)
    end
    
    @eval Base.:*(λ::$T, L::AbstractSciMLOperator) = SciMLScaledOperator(λ, L)
    @eval Base.:*(L::AbstractSciMLOperator, λ::$T) = SciMLScaledOperator(λ, L)
    @eval Base.:\(λ::$T, L::AbstractSciMLOperator) = SciMLScaledOperator(inv(λ), L)
    @eval Base.:\(L::AbstractSciMLOperator, λ::$T) = SciMLScaledOperator(λ, inv(L))
    @eval Base.:/(L::AbstractSciMLOperator, λ::$T) = SciMLScaledOperator(inv(λ), L)
    @eval Base.:/(λ::$T, L::AbstractSciMLOperator) = SciMLScaledOperator(λ, inv(L))
end

Base.:-(L::AbstractSciMLOperator) = SciMLScaledOperator(-true, L)
Base.:+(L::AbstractSciMLOperator) = L

Base.convert(::Type{AbstractMatrix}, L::SciMLScaledOperator) = L.λ.val * convert(AbstractMatrix, L.L)
SparseArrays.sparse(L::SciMLScaledOperator) = L.λ * sparse(L.L)

# traits
Base.size(L::SciMLScaledOperator) = size(L.L)
Base.adjoint(L::SciMLScaledOperator) = SciMLScaledOperator(L.λ', L.op')
LinearAlgebra.opnorm(L::SciMLScaledOperator, p::Real=2) = abs(L.λ) * opnorm(L.L, p)

getops(L::SciMLScaledOperator) = (L.λ, L.L)
islinear(L::SciMLScaledOperator) = all(islinear, L.ops)
isconstant(L::SciMLScaledOperator) = isconstant(L.L) & isconstant(L.λ)
Base.iszero(L::SciMLScaledOperator) = iszero(L.L) | iszero(L.λ)
has_adjoint(L::SciMLScaledOperator) = has_adjoint(L.L)
has_mul!(L::SciMLScaledOperator) = has_mul!(L.L)
has_ldiv(L::SciMLScaledOperator) = has_ldiv(L.L) & !iszero(L.λ)
has_ldiv!(L::SciMLScaledOperator) = has_ldiv!(L.L) & !iszero(L.λ)

# getindex
Base.getindex(L::SciMLScaledOperator, i::Int) = L.coeff * L.op[i]
Base.getindex(L::SciMLScaledOperator, I::Vararg{Int, N}) where {N} = L.λ * L.L[I...]
for fact in (
             :lu, :lu!,
             :qr, :qr!,
             :cholesky, :cholesky!,
             :ldlt, :ldlt!,
             :bunchkaufman, :bunchkaufman!,
             :lq, :lq!,
             :svd, :svd!,
            )
    @eval LinearAlgebra.$fact(L::SciMLScaledOperator, args...) = L.λ * fact(L.L, args...)
end

# operator application, inversion
for op in (
           :*, :\,
          )
    @eval Base.$op(L::SciMLScaledOperator, x::AbstractVector) = $op(L.λ, $op(L.L, x))
end

function LinearAlgebra.mul!(v::AbstractVector, L::SciMLScaledOperator, u::AbstractVector)
    mul!(v, L.L, u)
    lmul!(L.λ, v)
end

"""
Lazy operator addition (A + B)

    (A1 + A2 + A3...)u = A1*u + A2*u + A3*u ....
"""
struct SciMLAddedOperator{T,
                          O<:Tuple{Vararg{AbstractSciMLOperator}},
                          C,
                         } <: AbstractSciMLOperator{T}
    ops::O
    cache::C
    isunset::Bool

    function SciMLAddedOperator(ops...; cache = nothing)
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
Base.:+(ops::AbstractSciMLOperator...) = SciMLAddedOperator(ops...)

Base.:-(L::SciMLAddedOperator) = SciMLAddedOperator(.-(A.ops)...)
Base.:-(A::AbstractSciMLOperator, B::AbstractSciMLOperator) = SciMLAddedOperator(A, -B)

for op in (
           :+, :-,
          )

    @eval Base.$op(A::SciMLAddedOperator, B::SciMLAddedOperator) = SciMLAddedOperator(A.ops..., $op(B).ops...)
    @eval Base.$op(A::AbstractSciMLOperator, B::SciMLAddedOperator) = SciMLAddedOperator(A, $op(B).ops...)
    @eval Base.$op(A::SciMLAddedOperator, B::AbstractSciMLOperator) = SciMLAddedOperator(A.ops..., $op(B))

    for T in ScalingNumberTypes
        @eval function Base.$op(L::AbstractSciMLOperator, λ::$T)
            @assert issquare(L)
            N  = size(L, 1)
            Id = SciMLIdentity{N}()
            SciMLAddedOperator(L, $op(λ)*Id)
        end

        @eval function Base.$op(λ::$T, L::AbstractSciMLOperator)
            @assert issquare(L)
            N  = size(L, 1)
            Id = SciMLIdentity{N}()
            SciMLAddedOperator(λ*Id, $op(L))
        end
    end

    @eval function Base.$op(A::AbstractMatrix, L::AbstractSciMLOperator)
        @assert size(A) == size(L)
        SciMLAddedOperator(MatrixOperator(A), $op(L))
    end

    @eval function Base.$op(L::AbstractSciMLOperator, A::AbstractMatrix)
        @assert size(A) == size(L)
        SciMLAddedOperator(L, MatrixOperator($op(A)))
    end
end

Base.convert(::Type{AbstractMatrix}, L::SciMLAddedOperator) = sum(op -> convert(AbstractMatrix, op), L.ops)
SparseArrays.sparse(L::SciMLAddedOperator) = sum(_sparse, L.ops)

# traits
Base.size(L::SciMLAddedOperator) = size(first(L.ops))
function Base.adjoint(L::SciMLAddedOperator)
    if issquare(L) & !(L.isunset)
        SciMLAddedOperator(adjoint.(L.ops)...,L.cache, L.isunset)
    else
        SciMLAddedOperator(adjoint.(L.ops)...)
    end
end

getops(L::SciMLAddedOperator) = L.ops
Base.iszero(L::SciMLAddedOperator) = all(iszero, getops(L))
has_adjoint(L::SciMLAddedOperator) = all(has_adjoint, L.ops)

getindex(L::SciMLAddedOperator, i::Int) = sum(op -> op[i], L.ops)
getindex(L::SciMLAddedOperator, I::Vararg{Int, N}) where {N} = sum(op -> op[I...], L.ops)

function init_cache(A::SciMLAddedOperator, u::AbstractVector)
    cache = A.B * u
end

function Base.:*(L::SciMLAddedOperator, u::AbstractVector)
    sum(op -> iszero(op) ? similar(u, Bool) * false : op * u, L.ops)
end

function LinearAlgebra.mul!(v::AbstractVector, L::SciMLAddedOperator, u::AbstractVector)
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
struct SciMLComposedOperator{T,O,C} <: AbstractSciMLOperator{T}
    """ Tuple of N operators to be applied in reverse"""
    ops::O
    """ Tuple of N-1 cache vectors. cache[N-1] = op[N] * u and so on """
    cache::C
    isunset::Bool
    function SciMLComposedOperator(ops::AbstractSciMLOperator...; cache = nothing)
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

function init_cache(L::SciMLComposedOperator, u::AbstractVector)
    cache = ()
    vec = u
    for i in reverse(2:length(L.ops))
        vec = op[i] * vec
        cache = push(cache, vec)
    end
    cache
end

# constructors
Base.:∘(ops::AbstractSciMLOperator...) = SciMLComposedOperator(ops...)
Base.:∘(A::SciMLComposedOperator, B::SciMLComposedOperator) = SciMLComposedOperator(A.ops..., B.ops...)
Base.:∘(A::AbstractSciMLOperator, B::SciMLComposedOperator) = SciMLComposedOperator(A, B.ops...)
Base.:∘(A::SciMLComposedOperator, B::AbstractSciMLOperator) = SciMLComposedOperator(A.ops..., B)

# operator fusion falls back on composition
Base.:*(ops::AbstractSciMLOperator...) = reduce(*, ops) # pairwise fusion
Base.:*(A::AbstractSciMLOperator, B::AbstractSciMLOperator) = ∘(A, B)
Base.:*(A::SciMLComposedOperator, B::AbstractSciMLOperator) = ∘(A.ops[1:end-1]..., A.ops[end] * B)
Base.:*(A::AbstractSciMLOperator, B::SciMLComposedOperator) = ∘(A * B.ops[1], B.ops[2:end]...)

Base.convert(::Type{AbstractMatrix}, L::SciMLComposedOperator) = prod(op -> convert(AbstractMatrix, op), L.ops)
SparseArrays.sparse(L::SciMLComposedOperator) = prod(_sparse, L.ops)

# traits
Base.size(L::SciMLComposedOperator) = (size(first(L.ops), 1), size(last(L.ops),2))
Base.adjoint(L::SciMLComposedOperator) = SciMLComposedOperator(adjoint.(reverse(L.ops)))
LinearAlgebra.opnorm(L::SciMLComposedOperator) = prod(opnorm, L.ops)

getops(L::SciMLComposedOperator) = L.ops
islinear(L::SciMLComposedOperator) = all(islinear, L.ops)
Base.iszero(L::SciMLComposedOperator) = all(iszero, getops(L))
has_adjoint(L::SciMLComposedOperator) = all(has_adjoint, L.ops)
has_mul!(L::SciMLComposedOperator) = all(has_mul!, L.ops)
has_ldiv(L::SciMLComposedOperator) = all(has_ldiv, L.ops)
has_ldiv!(L::SciMLComposedOperator) = all(has_mul!, L.ops)

factorize(L::SciMLComposedOperator) = prod(factorize, reverse(L.ops))
for fact in (
             :lu, :lu!,
             :qr, :qr!,
             :cholesky, :cholesky!,
             :ldlt, :ldlt!,
             :bunchkaufman, :bunchkaufman!,
             :lq, :lq!,
             :svd, :svd!,
            )
    @eval LinearAlgebra.$fact(L::SciMLComposedOperator, args...) = prod(op -> $fact(op, args...), reverse(L.ops))
end

# operator application
Base.:*(L::SciMLComposedOperator, u::AbstractVector) = foldl((acc, op) -> op * acc, reverse(L.ops); init=u)
Base.:\(L::SciMLComposedOperator, u::AbstractVector) = foldl((acc, op) -> op \ acc, L.ops; init=u)

function LinearAlgebra.mul!(v::AbstractVector, L::SciMLComposedOperator, u::AbstractVector)
    @assert L.isunset "cache needs to be set up to use LinearAlgebra.mul!"

    vecs = (v, L.cache..., u)
    for i in reverse(1:length(L.ops))
        mul!(vecs[i], L.ops[i], vecs[i+1])
    end
    v
end

function LinearAlgebra.ldiv!(v::AbstractVector, L::SciMLComposedOperator, u::AbstractVector)
    @assert L.isunset "cache needs to be set up to use LinearAlgebra.ldiv!"

    vecs = (u, reverse(L.cache)..., v)
    for i in 1:length(L.ops)
        ldiv!(vecs[i], L.ops[i], vecs[i+1])
    end
    v
end

function LinearAlgebra.ldiv!(L::SciMLComposedOperator, u::AbstractVector)
    @assert L.isunset "cache needs to be set up to use LinearAlgebra.ldiv!"

    for i in 1:length(L.ops)
        ldiv!(L.ops[i], vecs[i])
    end
    v
end
#
