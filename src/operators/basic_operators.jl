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
iszero(::DiffEqIdentity) = false
issquare(::DiffEqIdentity) = true
has_adjoint(::DiffEqIdentity) = true
has_mul!(::DiffEqIdentity) = true
has_ldiv(::DiffEqIdentity) = true
has_ldiv!(::DiffEqIdentity) = true

# opeator application
for op in (
           :*, :\,
          )
    @eval Base.$op(::DiffEqIdentity{N}, x::AbstractVector) where{N} = (@assert length(x) == N; copy(x))
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
for op in (:*, :∘, :/, :\)
    @eval Base.$op(::DiffEqIdentity{N}, A::AbstractSciMLOperator) where{N} = (@assert size(A, 1) == N; DiffEqIdentity{N}())
    @eval Base.$op(A::AbstractSciMLOperator, ::DiffEqIdentity{N}) where{N} = (@assert size(A, 2) == N; DiffEqIdentity{N}())
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
issquare(::DiffEqNullOperator) = true
iszero(::DiffEqNullOperator) = true
has_adjoint(::DiffEqNullOperator) = true
has_mul!(::DiffEqNullOperator) = true

# opeator application
Base.:*(::DiffEqNullOperator{N}, x::AbstractVector) where{N} = (@assert length(x) == N; zero(x))
Base.:*(x::AbstractVector, ::DiffEqNullOperator{N}) where{N} = (@assert length(x) == N; zero(x))

function LinearAlgebra.mul!(v::AbstractVector, ::DiffEqNullOperator{N}, u::AbstractArray) where{N}
    @assert length(u) == length(v) == N
    lmul!(false, v)
end

# operator fusion, composition
for op in (:*, :∘)
    @eval Base.$op(::DiffEqNullOperator{N}, A::AbstractSciMLOperator) where{N} = (@assert size(A, 1) == N; DiffEqNullOperator{N}())
    @eval Base.$op(A::AbstractSciMLOperator, ::DiffEqNullOperator{N}) where{N} = (@assert size(A, 2) == N; DiffEqNullOperator{N}())
end

"""
    DiffEqScalar(val[; update_func])

    (α::DiffEqScalar)(a::Number) = α * a

Represents a time-dependent scalar/scaling operator. The update function
is called by `update_coefficients!` and is assumed to have the following
signature:

    update_func(oldval,u,p,t) -> newval
"""
mutable struct DiffEqScalar{T<:Number,F} <: AbstractDiffEqLinearOperator{T}
  val::T
  update_func::F
  DiffEqScalar(val::T; update_func=DEFAULT_UPDATE_FUNC) where{T} =
    new{T,typeof(update_func)}(val, update_func)
end

update_coefficients!(α::DiffEqScalar,u,p,t) = (α.val = α.update_func(α.val,u,p,t); α)

# constructors
Base.convert(::Type{Number}, α::DiffEqScalar) = α.val
Base.convert(::Type{DiffEqScalar}, α::Number) = DiffEqScalar(α)

# traits
Base.size(α::DiffEqScalar) = ()
function Base.adjoint(α::DiffEqScalar) # TODO - test
    val = α.val'
    update_func =  (oldval,u,p,t) -> α.update_func(oldval',u,p,t)'
    DiffEqScalar(val; update_func=update_func)
end

getops(α::DiffEqScalar) = (α.val)
isconstant(α::DiffEqScalar) = α.update_func == DEFAULT_UPDATE_FUNC
iszero(α::DiffEqScalar) = iszero(α.val)
has_adjoint(::DiffEqScalar) = true
has_mul(::DiffEqScalar) = true
has_ldiv(α::DiffEqScalar) = iszero(α.val)

for op in (:*, :/, :\)
    for T in (
              :Number,
              :AbstractArray, # TODO - to act on AbstractArrays, it needs a notion of size?
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

    function ScaledDiffEqOperator(λ::Union{Number,DiffEqScalar}, L::AbstractDiffEqOperator)
        λ = λ isa DiffEqScalar ? λ : DiffEqScalar(λ)
        T = promote_type(eltype.(λ, L)...)
        new{T,typeof(λ),typeof(op)}(λ, L)
    end
end

# constructor
Base.:*(λ::Union{Number,DiffEqScalar}, L::AbstractDiffEqOperator) = ScaledDiffEqOperator(λ, L)
Base.:*(L::AbstractDiffEqOperator, λ::Union{Number,DiffEqScalar}) = ScaledDiffEqOperator(λ, L)
Base.:\(λ::Union{Number,DiffEqScalar}, L::AbstractDiffEqOperator) = ScaledDiffEqOperator(inv(λ), L)
Base.:\(L::AbstractDiffEqOperator, λ::Union{Number,DiffEqScalar}) = ScaledDiffEqOperator(λ, inv(L))
Base.:/(L::AbstractDiffEqOperator, λ::Union{Number,DiffEqScalar}) = ScaledDiffEqOperator(inv(λ), L)
Base.:/(λ::Union{Number,DiffEqScalar}, L::AbstractDiffEqOperator) = ScaledDiffEqOperator(λ, inv(L))

Base.:-(L::AbstractDiffEqOperator) = ScaledDiffEqOperator(-true, L)
Base.:+(L::AbstractDiffEqOperator) = L

Base.convert(::Type{AbstractMatrix}, L::ScaledDiffEqOperator) = λ * convert(AbstractMatrix, L.L)
Base.Matrix(L::ScaledDiffEqOperator) = L.λ * Matrix(L.L)

# traits
Base.size(L::ScaledDiffEqOperator) = size(L.L)
Base.adjoint(L::ScaledDiffEqOperator) = ScaledDiffEqOperator(L.λ', L.op')
LinearAlgebra.opnorm(L::ScaledDiffEqOperator, p::Real=2) = abs(L.λ) * opnorm(L.L, p)

getops(L::ScaledDiffEqOperator) = (L.λ, L.A)
isconstant(L::ScaledDiffEqOperator) = isconstant(L.L) & isconstant(L.λ)
iszero(L::ScaledDiffEqOperator) = iszero(L.L) & iszero(L.λ)
issquare(L::ScaledDiffEqOperator) = issquare(L.L)
has_adjoint(L::ScaledDiffEqOperator) = has_adjoint(L.L)
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
             :svd, :svd!
            )
    @eval LinearAlgebra.$fact(L::ScaledDiffEqOperator, args...) = L.λ * fact(L.L, args...)
end

# operator application
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

    (A + B)u = Au + Bu
"""
struct AddedDiffEqOperator{T,
                           Ta<:AbstractDiffEqOperator,
                           Tb<:AbstractDiffEqOperator,
                           Tc,
                          } <: AbstractDiffEqOperator{T}
    A::Ta
    B::Tb

    cache::Tc
    isunset::Bool

    function AddedDiffEqOperator(A::AbstractDiffEqOperator,
                                 B::AbstractDiffEqOperator;
                                 cache = nothing,
                                )
        @assert size(A) == size(B)
        T = promote_type(eltype.((A,B))...)

        isunset = cache === nothing,
        new{T,
            typeof(A),
            typeof(B),
            typeof(cache),
           }(
             A, B, cache, isunset,
            )
    end
end

Base.convert(::Type{AbstractMatrix}, L::AddedDiffEqOperator) = convert(AbstractMatrix, L.A) + convert(AbstractMatrix, L.B)
Base.Matrix(L::AddedDiffEqOperator) = Matrix(L.A) + Matrix(L.B)

# traits
Base.size(A::AddedDiffEqOperator) = size(A.A)
function Base.adjoint(A::AddedDiffEqOperator)
    if issquare(A) & !(A.isunset)
        AddedDiffEqOperator(A.A',A.B',A.cache, A.isunset)
    else
        AddedDiffEqOperator(A.A',A.B')
    end
end

getops(L::AddedDiffEqOperator) = (L.A, L.B)
iszero(L::AddedDiffEqOperator) = all(iszero, getops(L))
issquare(L::AddedDiffEqOperator) = issquare(L.A)
has_adjoint(L::AddedDiffEqOperator) = has_adjoint(L.A) & has_adjoint(L.B)

function init_cache(A::AddedDiffEqOperator, u::AbstractVector)
    cache = A.B * u
end

function Base.:*(L::AddedDiffEqOperator, u::AbstractVector)
    @unpack A, B = L
    if iszero(A)
        B * u
    elseif iszero(B)
        A * u
    else
        (A * u) + (B * u)
    end
end

function LinearAlgebra.mul!(v::AbstractVector, L::AddedDiffEqOperator, u::AbstractVector)
    @unpack A, B, cache, isunset = Op

    if iszero(A)
        return mul!(v, B, u)
    elseif iszero(B)
        return mul!(v, A, u)
    end

    mul!(v, A, u)

    if isunset
        cache = init_cache(Op, u)
        Op = set_cache(Op, cache)
    end

    mul!(cache, B, u)
    axpy!(true, cache, v)
end

# operator fusion
for op in (
           :+, :-,
          )

    @eval function Base.$op(A::AbstractDiffEqOperator, B::AbstractDiffEqOperator)
        AddedDiffEqOperator(A, $op(B))
    end

    @eval function Base.$op(A::AbstractDiffEqOperator, λ::Union{DiffEqScalar,Number})
        @assert issquare(A)
        N  = size(A, 1)
        Id = DiffEqIdentity{N}()
        AddedDiffEqOperator(A, $op(λ)*Id)
    end

    @eval function Base.$op(λ::Union{DiffEqScalar,Number}, A::AbstractDiffEqOperator)
        @assert issquare(A)
        N  = size(A, 1)
        Id = DiffEqIdentity{N}()
        AddedDiffEqOperator(λ*Id, $op(A))
    end
end
