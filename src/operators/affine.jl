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

function update_coefficients!(L::ScaledDiffEqOperator, u, p, t)
    update_coefficients!(L.L, u, p, t)
    update_coefficients!(L.λ, u, p, t)
    L
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

Base.convert(::Type{AbstractMatrix}, L::DiffEqScaledOperator) = λ * convert(AbstractMatrix, L.L)

# traits
Base.size(L::ScaledDiffEqOperator) = size(L.L)
Base.adjoint(L::ScaledDiffEqOperator) = ScaledDiffEqOperator(L.λ', L.op')

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
    @eval Base.$op(L::DiffEqScaledOperator, x::AbstractVector) = $op(L.λ, $op(L.L, x))
end

function LinearAlgebra.mul!(v::AbstractVector, L::DiffEqScaledOperator, u::AbstractVector)
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

function update_coefficients!(L::AddedDiffEqOperator, u, p, t)
    update_coefficients!(L.A, u, p, t)
    update_coefficients!(L.B, u, p, t)
    L
end

# traits
Base.size(A::AddedDiffEqOperator) = size(A.A)
function Base.adjoint(A::AddedDiffEqOperator)
    if issquare(A) & !(A.isunset)
        AddedDiffEqOperator(A.A',A.B',A.cache, A.isunset)
    else
        AddedDiffEqOperator(A.A',A.B')
    end
end

issquare(L::AddedDiffEqOperator) = issquare(L.A)
isconstant(L::AddedDiffEqOperator) = isconstant(L.A) & isconstant(L.B)
iszero(L::AddedDiffEqOperator) = iszero(L.A) & iszero(L.B)
has_adjoint(L::AddedDiffEqOperator) = has_adjoint(L.A) & has_adjoint(L.B)

function init_cache(A::AddedDiffEqOperator, u::AbstractField)
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

function LinearAlgebra.mul!(v::AbstractField, L::AddedDiffEqOperator, u::AbstractField)
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

function Base.:/(A::AbstractOperator, λ::Number)
    N = size(A, 1)
    Z = DiffEqNullOperator{N}()
    AddedDiffEqOperator(A, Z, -true, λ)
end

