#
###
# Operator interface
###

###
# Utilities for update functions
###

DEFAULT_UPDATE_FUNC(A,u,p,t) = A

struct NoKwargFilter end

function preprocess_update_func(update_func, accepted_kwargs)
    _update_func = (update_func === nothing) ? DEFAULT_UPDATE_FUNC : update_func
    _accepted_kwargs = (accepted_kwargs === nothing) ? () : accepted_kwargs 
    # accepted_kwargs can be passed as nothing to indicate that we should not filter 
    # (e.g. if the function already accepts all kwargs...). 
    return (_accepted_kwargs isa NoKwargFilter) ? _update_func : FilterKwargs(_update_func, _accepted_kwargs)
end
function update_func_isconstant(update_func)
    if update_func isa FilterKwargs
        return update_func.f == DEFAULT_UPDATE_FUNC
    else
        return update_func == DEFAULT_UPDATE_FUNC
    end
end

update_coefficients!(L,u,p,t; kwargs...) = nothing
update_coefficients(L,u,p,t; kwargs...) = L

function update_coefficients!(L::AbstractSciMLOperator, u, p, t; kwargs...)
    for op in getops(L)
        update_coefficients!(op, u, p, t; kwargs...)
    end
    L
end

(L::AbstractSciMLOperator)(u, p, t; kwargs...) = update_coefficients(L, u, p, t; kwargs...) * u
(L::AbstractSciMLOperator)(du, u, p, t; kwargs...) = (update_coefficients!(L, u, p, t; kwargs...); mul!(du, L, u))
(L::AbstractSciMLOperator)(du, u, p, t, α, β; kwargs...) = (update_coefficients!(L, u, p, t; kwargs...); mul!(du, L, u, α, β))

function (L::AbstractSciMLOperator)(du::Number, u::Number, p, t, args...; kwargs...)
    msg = """Nonallocating L(v, u, p, t) type methods are not available for
    subtypes of `Number`."""
    throw(ArgumentError(msg))
end

###
# caching interface
###

getops(L) = ()

function iscached(L::AbstractSciMLOperator)

    has_cache = hasfield(typeof(L), :cache) # TODO - confirm this is static
    isset = has_cache ? !isnothing(L.cache) : true

    return isset & all(iscached, getops(L))
end

iscached(L) = true

iscached(::Union{
                 # LinearAlgebra
                 AbstractMatrix,
                 UniformScaling,
                 Factorization,

                 # Base
                 Number,

                }
        ) = true

"""
Allocate caches for a SciMLOperator for fast evaluation

arguments:
    L :: AbstractSciMLOperator
    in :: AbstractVecOrMat input prototype to L
    out :: (optional) AbstractVecOrMat output prototype to L
"""
function cache_operator end

cache_operator(L, u) = L
cache_operator(L, u, v) = L
cache_self(L::AbstractSciMLOperator, ::AbstractVecOrMat...) = L
cache_internals(L::AbstractSciMLOperator, ::AbstractVecOrMat...) = L

function cache_operator(L::AbstractSciMLOperator, u::AbstractVecOrMat, v::AbstractVecOrMat)

    L = cache_self(L, u, v)
    L = cache_internals(L, u, v)
    L
end

function cache_operator(L::AbstractSciMLOperator, u::AbstractVecOrMat)

    L = cache_self(L, u)
    L = cache_internals(L, u)
    L
end

function cache_operator(L::AbstractSciMLOperator, u::AbstractArray)
    u isa AbstractVecOrMat && @error "cache_operator not defined for $(typeof(L)), $(typeof(u))."

    n = size(L, 2)
    s = size(u)
    k = prod(s[2:end])

    @assert s[1] == n "Dimension mismatch"

    U = reshape(u, (n, k))
    L = cache_operator(L, U)
    L
end

###
# Operator Traits
###

Base.size(A::AbstractSciMLOperator, d::Integer) = d <= 2 ? size(A)[d] : 1
Base.eltype(::Type{AbstractSciMLOperator{T}}) where T = T
Base.eltype(::AbstractSciMLOperator{T}) where T = T

Base.oneunit(L::AbstractSciMLOperator) = one(L)
Base.oneunit(LType::Type{<:AbstractSciMLOperator}) = one(LType)

Base.iszero(::AbstractSciMLOperator) = false # TODO

has_adjoint(L::AbstractSciMLOperator) = false # L', adjoint(L)
has_expmv!(L::AbstractSciMLOperator) = false # expmv!(v, L, t, u)
has_expmv(L::AbstractSciMLOperator) = false # v = exp(L, t, u)
has_exp(L::AbstractSciMLOperator) = islinear(L)
has_mul(L::AbstractSciMLOperator) = true # du = L*u
has_mul!(L::AbstractSciMLOperator) = true # mul!(du, L, u)
has_ldiv(L::AbstractSciMLOperator) = false # du = L\u
has_ldiv!(L::AbstractSciMLOperator) = false # ldiv!(du, L, u)

### Extra standard assumptions

isconstant(::Union{
                   # LinearAlgebra
                   AbstractMatrix,
                   UniformScaling,
                   Factorization,

                   # Base
                   Number,

                  }
          ) = true
isconstant(L::AbstractSciMLOperator) = all(isconstant, getops(L))

#islinear(L) = false
islinear(::AbstractSciMLOperator) = false

islinear(::Union{
                 # LinearAlgebra
                 AbstractMatrix,
                 UniformScaling,
                 Factorization,

                 # Base
                 Number,
                }
        ) = true

has_mul(L) = false
has_mul(::Union{
                # LinearAlgebra
                AbstractVecOrMat,
                AbstractMatrix,
                UniformScaling,

                # Base
                Number,
               }
       ) = true

has_mul!(L) = false
has_mul!(::Union{
                 # LinearAlgebra
                 AbstractVecOrMat,
                 AbstractMatrix,
                 UniformScaling,

                 # Base
                 Number,
                }
        ) = true

has_ldiv(L) = false
has_ldiv(::Union{
                 AbstractMatrix,
                 Factorization,
                 Number,
                }
        ) = true

has_ldiv!(L) = false
has_ldiv!(::Union{
                  Diagonal,
                  Bidiagonal,
                  Factorization
                 }
         ) = true

has_adjoint(L) = islinear(L)
has_adjoint(::Union{
                    # LinearAlgebra
                    AbstractMatrix,
                    UniformScaling,
                    Factorization,

                    # Base
                    Number,
                   }
           ) = true

issquare(L) = ndims(L) >= 2 && size(L, 1) == size(L, 2)
issquare(::AbstractVector) = false
issquare(::Union{
                 # LinearAlgebra
                 UniformScaling,

                 # SciMLOperators
                 AbstractSciMLScalarOperator,

                 # Base
                 Number,
                }
        ) = true
issquare(A...) = @. (&)(issquare(A)...)

Base.length(L::AbstractSciMLOperator) = prod(size(L))
Base.ndims(L::AbstractSciMLOperator) = length(size(L))
Base.isreal(L::AbstractSciMLOperator{T}) where{T} = T <: Real
Base.Matrix(L::AbstractSciMLOperator) = Matrix(convert(AbstractMatrix, L))

LinearAlgebra.exp(L::AbstractSciMLOperator,t) = exp(t*L)
expmv(L::AbstractSciMLOperator,u,p,t) = exp(L,t)*u
expmv!(v,L::AbstractSciMLOperator,u,p,t) = mul!(v,exp(L,t),u)

###
# fallback implementations
###

function Base.conj(L::AbstractSciMLOperator)
    isreal(L) && return L
    convert(AbstractMatrix, L) |> conj
end

function Base.:(==)(L1::AbstractSciMLOperator, L2::AbstractSciMLOperator)
    size(L1) != size(L2) && return false
    convert(AbstractMatrix, L1) == convert(AbstractMatrix, L1)
end

Base.@propagate_inbounds function Base.getindex(L::AbstractSciMLOperator, I::Vararg{Any,N}) where {N}
    convert(AbstractMatrix, L)[I...]
end
function Base.getindex(L::AbstractSciMLOperator, I::Vararg{Int, N}) where {N}
    convert(AbstractMatrix,L)[I...]
end

function Base.resize!(L::AbstractSciMLOperator, n::Integer)
    throw(MethodError(resize!, typeof.((L, n))))
end

LinearAlgebra.exp(L::AbstractSciMLOperator) = exp(Matrix(L))
LinearAlgebra.opnorm(L::AbstractSciMLOperator, p::Real=2) = opnorm(convert(AbstractMatrix,L), p)
for pred in (
             :issymmetric,
             :ishermitian,
             :isposdef,
            )
    @eval function LinearAlgebra.$pred(L::AbstractSciMLOperator)
        $pred(convert(AbstractMatrix, L))
    end
end
for op in (
           :sum,:prod
          )
  @eval function LinearAlgebra.$op(L::AbstractSciMLOperator; kwargs...)
      $op(convert(AbstractMatrix, L); kwargs...)
  end
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::AbstractSciMLOperator, u::AbstractVecOrMat)
    mul!(v, convert(AbstractMatrix,L), u)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::AbstractSciMLOperator, u::AbstractVecOrMat, α, β)
    mul!(v, convert(AbstractMatrix,L), u, α, β)
end
#
