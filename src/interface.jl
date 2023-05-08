#
###
# Operator interface
###

"""
Function call and multiplication:
    - L(du, u, p, t) for in-place operator evaluation,
    - du = L(u, p, t) for out-of-place operator evaluation

If the operator is not a constant, update it with (u,p,t). A mutating form, i.e.
update_coefficients!(A,u,p,t) that changes the internal coefficients, and a
out-of-place form B = update_coefficients(A,u,p,t).

"""
function (::AbstractSciMLOperator) end

DEFAULT_UPDATE_FUNC(A,u,p,t) = A

update_coefficients(L,u,p,t) = L
update_coefficients!(L,u,p,t) = L

function update_coefficients(L::AbstractSciMLOperator, u, p, t)
    @error "Vedant hasn't implemented OOP update_coeffs yet for $L"
end

function update_coefficients!(L::AbstractSciMLOperator, u, p, t)
    for op in getops(L)
        update_coefficients!(op, u, p, t)
    end
    L
end

(L::AbstractSciMLOperator)(u, p, t) = update_coefficients(L, u, p, t) * u
(L::AbstractSciMLOperator)(du, u, p, t) = (update_coefficients!(L, u, p, t); mul!(du, L, u))

###
# caching interface
###

function iscached(L::AbstractSciMLOperator)
    has_cache = hasfield(typeof(L), :cache) # TODO - confirm this is static
    isset = has_cache ? L.cache !== nothing : true

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
cache_operator

cache_operator(L, u) = L
cache_operatro(L, u, v) = L
cache_self(L::AbstractSciMLOperator, uv::AbstractVecOrMat...) = L
cache_internals(L::AbstractSciMLOperator, uv::AbstractVecOrMat...) = L

function cache_operator(L::AbstractSciMLOperator,
                        u::AbstractVecOrMat,
                        v::AbstractVecOrMat)
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
has_mul!(L::AbstractSciMLOperator) = false # mul!(du, L, u)
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
