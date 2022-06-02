# The `update_coefficients!` interface
DEFAULT_UPDATE_FUNC(A,u,p,t) = A # no-op used by the basic operators
# isconstant(::AbstractSciMLLinearOperator) = true # already defined in SciMLBase
function update_coefficients!(L::AbstractSciMLOperator, u, p, t)
    for op in getops(L)
        update_coefficients!(op, u, p, t)
    end
    L
end
#@deprecate is_constant(L::AbstractSciMLOperator) isconstant(L)
#iszero(L::AbstractSciMLOperator) = all(iszero, getops(L))

Base.size(A::AbstractSciMLOperator, d::Integer) = d <= 2 ? size(A)[d] : 1

# Routines that use the AbstractMatrix representation
Base.convert(::Type{AbstractArray}, L::AbstractSciMLLinearOperator) = convert(AbstractMatrix, L)
LinearAlgebra.opnorm(L::AbstractSciMLLinearOperator, p::Real=2) = opnorm(convert(AbstractMatrix,L), p)
Base.@propagate_inbounds Base.getindex(L::AbstractSciMLLinearOperator, I::Vararg{Any,N}) where {N} = convert(AbstractMatrix,L)[I...]
Base.getindex(L::AbstractSciMLLinearOperator, I::Vararg{Int, N}) where {N} =
  convert(AbstractMatrix,L)[I...]
for op in (:*, :/, :\)
    @eval Base.$op(L::AbstractSciMLLinearOperator, x::AbstractArray) = $op(convert(AbstractMatrix,L), x)
    @eval Base.$op(x::AbstractArray, L::AbstractSciMLLinearOperator) = $op(x, convert(AbstractMatrix,L))
end
LinearAlgebra.mul!(Y::AbstractArray, L::AbstractSciMLLinearOperator, B::AbstractArray) =
  mul!(Y, convert(AbstractMatrix,L), B)
LinearAlgebra.mul!(Y::AbstractArray, L::AbstractSciMLLinearOperator, B::AbstractArray, α::Number, β::Number) =
  mul!(Y, convert(AbstractMatrix,L), B, α, β)
for pred in (:isreal, :issymmetric, :ishermitian, :isposdef)
    @eval LinearAlgebra.$pred(L::AbstractSciMLLinearOperator) = $pred(convert(AbstractArray, L))
end
for op in (:sum,:prod)
  @eval LinearAlgebra.$op(L::AbstractSciMLLinearOperator; kwargs...) = $op(convert(AbstractArray, L); kwargs...)
end

# Routines that use the full matrix representation
Base.Matrix(L::AbstractSciMLLinearOperator) = Matrix(convert(AbstractMatrix, L))
LinearAlgebra.exp(L::AbstractSciMLLinearOperator) = exp(Matrix(L))

# TODO write fallback interfae for adjoint operator here
Base.adjoint(A::AbstractSciMLLinearOperator) = Adjoint(A)
