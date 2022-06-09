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

update_coefficients!(L,u,p,t) = nothing
update_coefficients(L,u,p,t) = L
function update_coefficients!(L::AbstractSciMLOperator, u, p, t)
    for op in getops(L)
        update_coefficients!(op, u, p, t)
    end
    L
end

(L::AbstractSciMLOperator)(u, p, t) = (update_coefficients!(L, u, p, t); L * u)
(L::AbstractSciMLOperator)(du, u, p, t) = (update_coefficients!(L, u, p, t); mul!(du, L, u))

###
# caching interface
###

"""
Allocate caches for a SciMLOperator for fast evaluation

arguments:
    L :: AbstractSciMLOperator
    u :: AbstractVecOrMat argument to L
"""
cache_operator(L, u) = L
cache_self(L, u) = L
cache_internals(L, u) = L

function cache_operator(L::AbstractSciMLOperator, u::AbstractVecOrMat)
    L = cache_self(L, u)
    L = cache_internals(L, u)
    L
end

###
# Operator Traits
###

Base.size(A::AbstractSciMLOperator, d::Integer) = d <= 2 ? size(A)[d] : 1
Base.eltype(::Type{AbstractSciMLOperator{T}}) where T = T
Base.eltype(::AbstractSciMLOperator{T}) where T = T

issquare(L::AbstractSciMLOperator) = isequal(size(L)...)

Base.iszero(::AbstractSciMLOperator) = false

islinear(::AbstractSciMLOperator) = false
islinear(::AbstractSciMLLinearOperator) = true

has_adjoint(L::AbstractSciMLOperator) = false # L', adjoint(L)
has_expmv!(L::AbstractSciMLOperator) = false # expmv!(v, L, t, u)
has_expmv(L::AbstractSciMLOperator) = false # v = exp(L, t, u)
has_exp(L::AbstractSciMLOperator) = false # v = exp(L, t)*u
has_mul(L::AbstractSciMLOperator) = true # du = L*u
has_mul!(L::AbstractSciMLOperator) = false # mul!(du, L, u)
has_ldiv(L::AbstractSciMLOperator) = false # du = L\u
has_ldiv!(L::AbstractSciMLOperator) = false # ldiv!(du, L, u)

### AbstractSciMLLinearOperator Interface

#=
1. AbstractSciMLLinearOperator <: AbstractSciMLOperator
2. Can absorb under multiplication by a scalar. In all algorithms things like
   dt*L show up all the time, so the linear operator must be able to absorb
   such constants.
4. isconstant(A) trait for whether the operator is constant or not.
5. Optional: diagonal, symmetric, etc traits from LinearMaps.jl.
6. Optional: exp(A). Required for simple exponential integration.
7. Optional: expmv(A,u,p,t) = exp(t*A)*u and expmv!(v,A::SciMLOperator,u,p,t)
   Required for sparse-saving exponential integration.
8. Optional: factorizations. A_ldiv_B, factorize et. al. This is only required
   for algorithms which use the factorization of the operator (Crank-Nicholson),
   and only for when the default linear solve is used.
=#

# Extra standard assumptions
isconstant(L) = true
isconstant(L::AbstractSciMLOperator) = all(isconstant, getops(L))
#isconstant(L::AbstractSciMLOperator) = L.update_func = DEFAULT_UPDATE_FUNC

islinear(L) = false
islinear(::Union{
                 # LinearAlgebra
                 AbstractMatrix,
                 UniformScaling,
                 Factorization,

                 # Base
                 Number,

                 # SciMLOperators
                 AbstractSciMLLinearOperator,
                }
        ) = true

has_mul(L) = true

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

has_adjoint(L) = false
has_adjoint(::Union{
                    # LinearAlgebra
                    AbstractMatrix,
                    UniformScaling,
                    Factorization,

                    # Base
                    Number,

                    # SciMLOperators
                    AbstractSciMLLinearOperator,
                   }
           ) = true

issquare(A) = size(A,1) === size(A,2)
issquare(::Union{
                 # LinearAlgebra
                 UniformScaling,

                 # Base
                 Number,
                }
        ) = true
issquare(A...) = @. (&)(issquare(A)...)

###
# default linear operator traits
###

Base.isreal(L::AbstractSciMLOperator{T}) where{T} = T <: Real
function Base.conj(L::AbstractSciMLOperator)
    isreal(L) && return L
    convert(AbstractMatrix, L) |> conj
end
function Base.:(==)(L1::AbstractSciMLOperator, L2::AbstractSciMLOperator)
    size(L1) != size(L2) && return false
    convert(AbstractMatrix, L1) == convert(AbstractMatrix, L1)
end

LinearAlgebra.exp(L::AbstractSciMLLinearOperator,t) = exp(t*L)
has_exp(L::AbstractSciMLLinearOperator) = true
expmv(L::AbstractSciMLLinearOperator,u,p,t) = exp(L,t)*u
expmv!(v,L::AbstractSciMLLinearOperator,u,p,t) = mul!(v,exp(L,t),u)

Base.Matrix(L::AbstractSciMLLinearOperator) = Matrix(convert(AbstractMatrix, L))

Base.@propagate_inbounds function Base.getindex(L::AbstractSciMLLinearOperator, I::Vararg{Any,N}) where {N}
    convert(AbstractMatrix, L)[I...]
end
function Base.getindex(L::AbstractSciMLLinearOperator, I::Vararg{Int, N}) where {N}
    convert(AbstractMatrix,L)[I...]
end

LinearAlgebra.exp(L::AbstractSciMLLinearOperator) = exp(Matrix(L))
LinearAlgebra.opnorm(L::AbstractSciMLLinearOperator, p::Real=2) = opnorm(convert(AbstractMatrix,L), p)
for pred in (
             :issymmetric,
             :ishermitian,
             :isposdef,
            )
    @eval function LinearAlgebra.$pred(L::AbstractSciMLLinearOperator)
        $pred(convert(AbstractMatrix, L))
    end
end
for op in (
           :sum,:prod
          )
  @eval function LinearAlgebra.$op(L::AbstractSciMLLinearOperator; kwargs...)
      $op(convert(AbstractMatrix, L); kwargs...)
  end
end

for op in (
           :*, :\,
          )
    @eval Base.$op(L::AbstractSciMLLinearOperator, x::AbstractVecOrMat) = $op(convert(AbstractMatrix,L), x)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::AbstractSciMLLinearOperator, u::AbstractVecOrMat)
    mul!(v, convert(AbstractMatrix,L), u)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::AbstractSciMLLinearOperator, u::AbstractVecOrMat, α, β)
    mul!(v, convert(AbstractMatrix,L), u, α, β)
end
#
