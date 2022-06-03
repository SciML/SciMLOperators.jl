###
# common interface
###

DEFAULT_UPDATE_FUNC(A,u,p,t) = A # no-op used by the basic operators
function update_coefficients!(L::AbstractSciMLOperator, u, p, t)
    for op in getops(L)
        update_coefficients!(op, u, p, t)
    end
    L
end

Base.size(A::AbstractSciMLOperator, d::Integer) = d <= 2 ? size(A)[d] : 1

###
# fallback traits
###

LinearAlgebra.opnorm(L::AbstractSciMLLinearOperator, p::Real=2) = opnorm(convert(AbstractMatrix,L), p)
Base.@propagate_inbounds Base.getindex(L::AbstractSciMLLinearOperator, I::Vararg{Any,N}) where {N} = convert(AbstractMatrix,L)[I...]
Base.getindex(L::AbstractSciMLLinearOperator, I::Vararg{Int, N}) where {N} = convert(AbstractMatrix,L)[I...]

for pred in (
             :isreal, :issymmetric, :ishermitian, :isposdef
            )
    @eval LinearAlgebra.$pred(L::AbstractSciMLLinearOperator) = $pred(convert(AbstractMatrix, L))
end
for op in (
           :sum,:prod
          )
  @eval LinearAlgebra.$op(L::AbstractSciMLLinearOperator; kwargs...) = $op(convert(AbstractMatrix, L); kwargs...)
end

# fallback operator application
for op in (
           :*, :\,
          )
    @eval Base.$op(L::AbstractSciMLLinearOperator, x::AbstractVector) = $op(convert(AbstractMatrix,L), x)
end

LinearAlgebra.mul!(Y::AbstractVector, L::AbstractSciMLLinearOperator, B::AbstractVector) = mul!(Y, convert(AbstractMatrix,L), B)
LinearAlgebra.mul!(Y::AbstractVector, L::AbstractSciMLLinearOperator, B::AbstractVector, α, β) = mul!(Y, convert(AbstractMatrix,L), B, α, β)

# Routines that use the full matrix representation
Base.Matrix(L::AbstractSciMLLinearOperator) = Matrix(convert(AbstractMatrix, L))
LinearAlgebra.exp(L::AbstractSciMLLinearOperator) = exp(Matrix(L))

# TODO write fallback interfae for adjoint operator here
Base.adjoint(A::AbstractSciMLLinearOperator) = Adjoint(A)
