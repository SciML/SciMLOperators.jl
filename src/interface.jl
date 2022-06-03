#
###
# AbstractSciMLOperator (u,p,t) and (du,u,p,t) interface
###

#=
1. Function call and multiplication: L(du, u, p, t) for inplace and du = L(u, p, t) for
   out-of-place, meaning L*u and mul!(du, L, u).
2. If the operator is not a constant, update it with (u,p,t). A mutating form, i.e.
   update_coefficients!(A,u,p,t) that changes the internal coefficients, and a
   out-of-place form B = update_coefficients(A,u,p,t).
3. isconstant(A) trait for whether the operator is constant or not.
4. islinear(A) trait for whether the operator is linear or not.
=#

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
# AbstractSciMLOperator Traits
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

islinear(::Union{
                 # LinearAlgebra
                 AbstractMatrix,
                 UniformScaling,
                 Factorization,

                 # Base
                 Number,

                 # SciMLOperator
                 AbstractSciMLLinearOperator,
                }
        ) = true

has_mul(L) = true
has_mul!(L) = false

has_ldiv(::Union{
                 AbstractMatrix,
                 Factorization,
                }
        ) = true

has_ldiv!(L) = false
has_ldiv!(::Union{
                  Diagonal,
                  Bidiagonal,
                  Factorization
                 }
         ) = true

has_adjoint(::Union{
                    # LinearAlgebra
                    AbstractMatrix,
                    UniformScaling,
                    Factorization,

                    # Base
                    Number,

                    # SciMLOperator
                    AbstractSciMLLinearOperator,
                   }
           ) = true

issquare(A) = size(A,1) === size(A,2)
issquare(::Union{
                 UniformScaling,
                }
        ) = true
issquare(A...) = @. (&)(issquare(A)...)

###
# default linear operator traits
###

LinearAlgebra.exp(L::AbstractSciMLLinearOperator,t) = exp(t*L)
has_exp(L::AbstractSciMLLinearOperator) = true
expmv(L::AbstractSciMLLinearOperator,u,p,t) = exp(L,t)*u
expmv!(v,L::AbstractSciMLLinearOperator,u,p,t) = mul!(v,exp(L,t),u)

Base.Matrix(L::AbstractSciMLLinearOperator) = Matrix(convert(AbstractMatrix, L))
Base.adjoint(A::AbstractSciMLLinearOperator) = Adjoint(A) # TODO write lazy adjoint operator interface here

Base.@propagate_inbounds function Base.getindex(L::AbstractSciMLLinearOperator, I::Vararg{Any,N}) where {N}
    convert(AbstractMatrix, L)[I...]
end
function Base.getindex(L::AbstractSciMLLinearOperator, I::Vararg{Int, N}) where {N}
    convert(AbstractMatrix,L)[I...]
end

LinearAlgebra.exp(L::AbstractSciMLLinearOperator) = exp(Matrix(L))
LinearAlgebra.opnorm(L::AbstractSciMLLinearOperator, p::Real=2) = opnorm(convert(AbstractMatrix,L), p)
for pred in (
             :isreal,
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
    @eval Base.$op(L::AbstractSciMLLinearOperator, x::AbstractVector) = $op(convert(AbstractMatrix,L), x)
end

function LinearAlgebra.mul!(v::AbstractVector, L::AbstractSciMLLinearOperator, u::AbstractVector)
    mul!(v, convert(AbstractMatrix,L), u)
end
function LinearAlgebra.mul!(v::AbstractVector, L::AbstractSciMLLinearOperator, u::AbstractVector, α, β)
    mul!(v, convert(AbstractMatrix,L), u, α, β)
end
#
