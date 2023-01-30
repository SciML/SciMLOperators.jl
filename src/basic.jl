"""
$(TYPEDEF)
"""
struct IdentityOperator{N} <: AbstractSciMLOperator{Bool} end

# constructors
IdentityOperator(u::AbstractArray) = IdentityOperator{size(u,1)}()

function Base.one(L::AbstractSciMLOperator)
    @assert issquare(L)
    N = size(L, 1)
    IdentityOperator{N}()
end

Base.convert(::Type{AbstractMatrix}, ::IdentityOperator{N}) where{N} = Diagonal(ones(Bool, N))

# traits
Base.size(::IdentityOperator{N}) where{N} = (N, N)
Base.adjoint(A::IdentityOperator) = A
Base.transpose(A::IdentityOperator) = A
Base.conj(A::IdentityOperator) = A
LinearAlgebra.opnorm(::IdentityOperator{N}, p::Real=2) where{N} = true
for pred in (
             :issymmetric, :ishermitian, :isposdef,
            )
    @eval LinearAlgebra.$pred(::IdentityOperator) = true
end

getops(::IdentityOperator) = ()
isconstant(::IdentityOperator) = true
islinear(::IdentityOperator) = true
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
struct NullOperator{N} <: AbstractSciMLOperator{Bool} end

# constructors
NullOperator(u::AbstractArray) = NullOperator{size(u,1)}()

function Base.zero(L::AbstractSciMLOperator)
    @assert issquare(L)
    N = size(L, 1)
    NullOperator{N}()
end

Base.convert(::Type{AbstractMatrix}, ::NullOperator{N}) where{N} = Diagonal(zeros(Bool, N))

# traits
Base.size(::NullOperator{N}) where{N} = (N, N)
Base.adjoint(A::NullOperator) = A
Base.transpose(A::NullOperator) = A
Base.conj(A::NullOperator) = A
LinearAlgebra.opnorm(::NullOperator{N}, p::Real=2) where{N} = false
for pred in (
             :issymmetric, :ishermitian,
            )
    @eval LinearAlgebra.$pred(::NullOperator) = true
end
LinearAlgebra.isposdef(::NullOperator) = false

getops(::NullOperator) = ()
isconstant(::NullOperator) = true
islinear(::NullOperator) = true
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
    ScaledOperator

    (λ L)*(u) = λ * L(u)
"""
struct ScaledOperator{T,
                      λType,
                      LType,
                     } <: AbstractSciMLOperator{T}
    λ::λType
    L::LType

    function ScaledOperator(λ::AbstractSciMLScalarOperator{Tλ},
                            L::AbstractSciMLOperator{TL},
                           ) where{Tλ,TL}
        T = promote_type(Tλ, TL)
        new{T,typeof(λ),typeof(L)}(λ, L)
    end
end

# constructors
for T in SCALINGNUMBERTYPES[2:end]
    @eval ScaledOperator(λ::$T, L::AbstractSciMLOperator) = ScaledOperator(ScalarOperator(λ), L)
end

for T in SCALINGNUMBERTYPES
    @eval function ScaledOperator(λ::$T, L::ScaledOperator)
        λ = ScalarOperator(λ) * L.λ
        ScaledOperator(λ, L.L)
    end
    
    for LT in SCALINGCOMBINETYPES
        @eval Base.:*(λ::$T, L::$LT) = ScaledOperator(λ, L)
        @eval Base.:*(L::$LT, λ::$T) = ScaledOperator(λ, L)

        @eval Base.:\(λ::$T, L::$LT) = ScaledOperator(inv(λ), L)
        @eval Base.:\(L::$LT, λ::$T) = ScaledOperator(λ, inv(L))

        @eval Base.:/(L::$LT, λ::$T) = ScaledOperator(inv(λ), L)
        @eval Base.:/(λ::$T, L::$LT) = ScaledOperator(λ, inv(L))
    end
end

Base.:-(L::AbstractSciMLOperator) = ScaledOperator(-true, L)
Base.:+(L::AbstractSciMLOperator) = L

Base.convert(::Type{AbstractMatrix}, L::ScaledOperator) = convert(Number,L.λ) * convert(AbstractMatrix, L.L)
SparseArrays.sparse(L::ScaledOperator) = L.λ * sparse(L.L)

# traits
Base.size(L::ScaledOperator) = size(L.L)
for op in (
           :adjoint,
           :transpose,
          )
    @eval Base.$op(L::ScaledOperator) = ScaledOperator($op(L.λ), $op(L.L))
end
Base.conj(L::ScaledOperator) = conj(L.λ) * conj(L.L)
LinearAlgebra.opnorm(L::ScaledOperator, p::Real=2) = abs(L.λ) * opnorm(L.L, p)

getops(L::ScaledOperator) = (L.λ, L.L,)
isconstant(L::ScaledOperator) = isconstant(L.L) & isconstant(L.λ)
islinear(L::ScaledOperator) = islinear(L.L)
Base.iszero(L::ScaledOperator) = iszero(L.L) | iszero(L.λ)
has_adjoint(L::ScaledOperator) = has_adjoint(L.L)
has_mul(L::ScaledOperator) = has_mul(L.L)
has_mul!(L::ScaledOperator) = has_mul!(L.L)
has_ldiv(L::ScaledOperator) = has_ldiv(L.L) & !iszero(L.λ)
has_ldiv!(L::ScaledOperator) = has_ldiv!(L.L) & !iszero(L.λ)

function cache_internals(L::ScaledOperator, u::AbstractVecOrMat)
    @set! L.L = cache_operator(L.L, u)
    @set! L.λ = cache_operator(L.λ, u)
    L
end

# getindex
Base.getindex(L::ScaledOperator, i::Int) = L.coeff * L.L[i]
Base.getindex(L::ScaledOperator, I::Vararg{Int, N}) where {N} = L.λ * L.L[I...]

factorize(L::ScaledOperator) = L.λ * factorize(L.L)
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
Base.:*(L::ScaledOperator, u::AbstractVecOrMat) = L.λ * (L.L * u)
Base.:\(L::ScaledOperator, u::AbstractVecOrMat) = L.λ \ (L.L \ u)

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::ScaledOperator, u::AbstractVecOrMat)
    iszero(L.λ) && return lmul!(false, v)
    a = convert(Number, L.λ)
    mul!(v, L.L, u, a, false)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::ScaledOperator, u::AbstractVecOrMat, α, β)
    iszero(L.λ) && return lmul!(β, v)
    a = convert(Number, L.λ*α)
    mul!(v, L.L, u, a, β)
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
        @assert !isempty(ops)
        T = promote_type(eltype.(ops)...)
        new{T,typeof(ops)}(ops)
    end
end

function AddedOperator(ops::AbstractSciMLOperator...)
    sz = size(first(ops))
    for op in ops[2:end]
        @assert size(op) == sz "Dimension mismatch: cannot add operators of
        sizes $(sz), and $(size(op))."
    end
    AddedOperator(ops)
end

AddedOperator(L::AbstractSciMLOperator) = L

# constructors
Base.:+(A::AbstractSciMLOperator, B::AbstractMatrix) = A + MatrixOperator(B)
Base.:+(A::AbstractMatrix, B::AbstractSciMLOperator) = MatrixOperator(A) + B
Base.:+(ops::AbstractSciMLOperator...) = AddedOperator(ops...)

Base.:+(A::AbstractSciMLOperator, B::AddedOperator) = AddedOperator(A, B.ops...)
Base.:+(A::AddedOperator, B::AbstractSciMLOperator) = AddedOperator(A.ops..., B)
Base.:+(A::AddedOperator, B::AddedOperator) = AddedOperator(A.ops..., B.ops...)

function Base.:+(A::AddedOperator, Z::NullOperator)
    @assert size(A) == size(Z)
    A
end

function Base.:+(Z::NullOperator, A::AddedOperator)
    @assert size(A) == size(Z)
    A
end

Base.:-(A::AbstractSciMLOperator, B::AbstractSciMLOperator) = AddedOperator(A, -B)
Base.:-(A::AbstractSciMLOperator, B::AbstractMatrix) = A - MatrixOperator(B)
Base.:-(A::AbstractMatrix, B::AbstractSciMLOperator) = MatrixOperator(A) - B

for op in (
           :+, :-,
          )

    for T in SCALINGNUMBERTYPES
        for LT in SCALINGCOMBINETYPES
            @eval function Base.$op(L::$LT, λ::$T)
                @assert issquare(L)
                N  = size(L, 1)
                Id = IdentityOperator{N}()
                AddedOperator(L, $op(λ)*Id)
            end

            @eval function Base.$op(λ::$T, L::$LT)
                @assert issquare(L)
                N  = size(L, 1)
                Id = IdentityOperator{N}()
                AddedOperator(λ*Id, $op(L))
            end
        end
    end
end

Base.convert(::Type{AbstractMatrix}, L::AddedOperator) = sum(op -> convert(AbstractMatrix, op), L.ops)
SparseArrays.sparse(L::AddedOperator) = sum(sparse, L.ops)

# traits
Base.size(L::AddedOperator) = size(first(L.ops))
for op in (
           :adjoint,
           :transpose,
          )
    @eval Base.$op(L::AddedOperator) = AddedOperator($op.(L.ops)...)
end
Base.conj(L::AddedOperator) = AddedOperator(conj.(L.ops))

getops(L::AddedOperator) = L.ops
islinear(L::AddedOperator) = all(islinear, getops(L))
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
    sum(op -> iszero(op) ? zero(u) : op * u, L.ops)
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
        @assert !isempty(ops)
        for i in reverse(2:length(ops))
            opcurr = ops[i]
            opnext = ops[i-1]
            @assert size(opcurr, 1) == size(opnext, 2) "Dimension mismatch: cannot compose
            operators of sizes $(size(opnext)), and $(size(opcurr))."
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
Base.:*(A::ComposedOperator, B::ComposedOperator) = ComposedOperator(A.ops..., B.ops...)

for op in (
           :*, :∘,
          )
    # identity
    @eval function Base.$op(::IdentityOperator{N}, A::ComposedOperator) where{N}
        @assert size(A, 1) == N
        A
    end

    @eval function Base.$op(A::ComposedOperator, ::IdentityOperator{N}) where{N}
        @assert size(A, 2) == N
        A
    end

    # null operator
    @eval function Base.$op(::NullOperator{N}, A::ComposedOperator) where{N}
        @assert size(A, 1) == N
        zero(A)
    end

    @eval function Base.$op(A::ComposedOperator, ::NullOperator{N}) where{N}
        @assert size(A, 2) == N
        zero(A)
    end

    # scalar operator
    @eval function Base.$op(λ::AbstractSciMLScalarOperator, L::ComposedOperator)
        ScaledOperator(λ, L)
    end

    @eval function Base.$op(L::ComposedOperator, λ::AbstractSciMLScalarOperator)
        ScaledOperator(λ, L)
    end
end

Base.convert(::Type{AbstractMatrix}, L::ComposedOperator) = prod(op -> convert(AbstractMatrix, op), L.ops)
SparseArrays.sparse(L::ComposedOperator) = prod(sparse, L.ops)

# traits
Base.size(L::ComposedOperator) = (size(first(L.ops), 1), size(last(L.ops),2))
for op in (
           :adjoint,
           :transpose,
          )
    @eval Base.$op(L::ComposedOperator) = ComposedOperator(
                                                           $op.(reverse(L.ops))...;
                                                           cache=L.isset ? reverse(L.cache) : nothing,
                                                          )
end
Base.conj(L::ComposedOperator) = ComposedOperator(conj.(L.ops); cache=L.cache)
LinearAlgebra.opnorm(L::ComposedOperator) = prod(opnorm, L.ops)

getops(L::ComposedOperator) = L.ops
islinear(L::ComposedOperator) = all(islinear, L.ops)
Base.iszero(L::ComposedOperator) = all(iszero, getops(L))
has_adjoint(L::ComposedOperator) = all(has_adjoint, L.ops)
has_mul(L::ComposedOperator) = all(has_mul, L.ops)
has_mul!(L::ComposedOperator) = all(has_mul!, L.ops)
has_ldiv(L::ComposedOperator) = all(has_ldiv, L.ops)
has_ldiv!(L::ComposedOperator) = all(has_ldiv!, L.ops)

factorize(L::ComposedOperator) = prod(factorize, L.ops)
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
# https://github.com/SciML/SciMLOperators.jl/pull/94
#Base.:*(L::ComposedOperator, u::AbstractVecOrMat) = foldl((acc, op) -> op * acc, reverse(L.ops); init=u)
#Base.:\(L::ComposedOperator, u::AbstractVecOrMat) = foldl((acc, op) -> op \ acc, L.ops; init=u)

function Base.:\(L::ComposedOperator, u::AbstractVecOrMat)
    v = u
    for op in L.ops
        v = op \ v
    end

    v
end

function Base.:*(L::ComposedOperator, u::AbstractVecOrMat)
    v = u
    for op in reverse(L.ops)
        v = op * v
    end

    v
end

function cache_self(L::ComposedOperator, u::AbstractVecOrMat)
    if has_mul(L)
        vec = zero(u)
        cache = (vec,)
        for i in reverse(2:length(L.ops))
            vec   = L.ops[i] * vec
            cache = (vec, cache...)
        end
    elseif has_ldiv(L)
        m = size(L, 1) 
        k = size(u, 2)
        vec = u isa AbstractMatrix ? similar(u, (m, k)) : similar(u, (m,))
        cache = ()
        for i in 1:length(L.ops)
            vec   = L.ops[i] \ vec
            cache = (cache..., vec)
        end
    else
        error("ComposedOperator cannot be cached without supporting either mul or ldiv.")
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

Base.:\(A::AbstractSciMLOperator, B::AbstractSciMLOperator) = inv(A) * B
Base.:/(A::AbstractSciMLOperator, B::AbstractSciMLOperator) = A * inv(B)

Base.convert(::Type{AbstractMatrix}, L::InvertedOperator) = inv(convert(AbstractMatrix, L.L))

Base.size(L::InvertedOperator) = size(L.L) |> reverse
Base.transpose(L::InvertedOperator) = InvertedOperator(transpose(L.L); cache = L.isset ? L.cache' : nothing)
Base.adjoint(L::InvertedOperator) = InvertedOperator(adjoint(L.L); cache = L.isset ? L.cache' : nothing)
Base.conj(L::InvertedOperator) = InvertedOperator(conj(L.L); cache=L.cache)

getops(L::InvertedOperator) = (L.L,)
islinear(L::InvertedOperator) = islinear(L.L)

has_mul(L::InvertedOperator) = has_ldiv(L.L)
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
    cache = zero(u)
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

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::InvertedOperator, u::AbstractVecOrMat)
    mul!(v, L.L, u)
end

function LinearAlgebra.ldiv!(L::InvertedOperator, u::AbstractVecOrMat)
    @assert L.isset "cache needs to be set up for operator of type $(typeof(L)).
    set up cache by calling cache_operator(L::AbstractSciMLOperator, u::AbstractArray)"

    copy!(L.cache, u)
    mul!(u, L.L, L.cache)
end
#
