#
###
# AbstractSciMLOperator Traits
###

# SciML
#isconstant = false
#islinear = false
#iszero = false

#has_adjoint = false
#has_mul = false
#has_mul! = false
#has_ldiv = false
#has_ldiv! = false

#LinearAlgebra.isreal
#LinearAlgebra.issymmetric
#LinearAlgebra.ishermitian
#LinearAlgbera.isposdef

Base.size(A::AbstractSciMLOperator, d::Integer) = d <= 2 ? size(A)[d] : 1
Base.eltype(::Type{AbstractSciMLOperator{T}}) where T = T
Base.eltype(::AbstractSciMLOperator{T}) where T = T

isconstant(L::AbstractSciMLOperator) = all(isconstant, getops(L))
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
isconstant(::AbstractSciMLLinearOperator) = true
isconstant(L::AbstractSciMLOperator) = L.update_func = DEFAULT_UPDATE_FUNC

isconstant(::AbstractMatrix) = true
islinear(::AbstractMatrix) = true
has_adjoint(::AbstractMatrix) = true
has_mul(::AbstractMatrix) = true
has_mul!(::AbstractMatrix) = true
has_ldiv(::AbstractMatrix) = true
has_ldiv!(::AbstractMatrix) = false
has_ldiv!(::Union{Diagonal, Bidiagonal, Factorization}) = true

issquare(::UniformScaling) = true
issquare(A) = size(A,1) === size(A,2)
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
