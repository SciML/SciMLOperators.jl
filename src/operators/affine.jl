#
"""
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

Base.convert(::Type{AbstractMatrix}, L::ScaledDiffEqOperator) = λ * convert(AbstractMatrix, L.L)
Base.Matrix(L::ScaledDiffEqOperator) = L.λ * Matrix(L.L)

# traits
Base.size(L::ScaledDiffEqOperator) = size(L.L)
Base.adjoint(L::ScaledDiffEqOperator) = ScaledDiffEqOperator(L.λ', L.op')
LinearAlgebra.opnorm(L::ScaledDiffEqOperator, p::Real=2) = abs(L.λ) * opnorm(L.L, p)

# getindex
Base.getindex(L::ScaledDiffEqOperator, i::Int) = L.coeff * L.op[i]
Base.getindex(L::ScaledDiffEqOperator, I::Vararg{Int, N}) where {N} = L.λ * L.L[I...]

getops(L::ScaledDiffEqOperator) = (L.λ, L.A)
isconstant(L::ScaledDiffEqOperator) = isconstant(L.L) & isconstant(L.λ)
iszero(L::ScaledDiffEqOperator) = iszero(L.L) & iszero(L.λ)
issquare(L::ScaledDiffEqOperator) = issquare(L.L)
has_adjoint(L::ScaledDiffEqOperator) = has_adjoint(L.L)
has_ldiv(L::ScaledDiffEqOperator) = has_ldiv(L.L) & !iszero(L.λ)
has_ldiv!(L::ScaledDiffEqOperator) = has_ldiv!(L.L) & !iszero(L.λ)

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
                                 isunset = cache === nothing,
                                )
        @assert size(A) == size(B)
        T = promote_type(eltype.((A,B))...)

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
isconstant(L::AddedDiffEqOperator) = isconstant(L.A) & isconstant(L.B)
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
#
