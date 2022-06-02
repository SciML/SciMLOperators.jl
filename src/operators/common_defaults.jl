# The `update_coefficients!` interface
DEFAULT_UPDATE_FUNC(A,u,p,t) = A # no-op used by the basic operators
# isconstant(::AbstractDiffEqLinearOperator) = true # already defined in DiffEqBase
function update_coefficients!(L::AbstractDiffEqOperator, u, p, t)
    for op in getops(L)
        update_coefficients!(op, u, p, t)
    end
    L
end
isconstant(L::AbstractDiffEqOperator) = all(isconstant, getops(L))
#@deprecate is_constant(L::AbstractDiffEqOperator) isconstant(L)
#iszero(L::AbstractDiffEqOperator) = all(iszero, getops(L))
issquare(L::AbstractDiffEqOperator) = size(L, 1) == size(L, 2)

Base.size(A::AbstractDiffEqOperator, d::Integer) = d <= 2 ? size(A)[d] : 1

# Routines that use the AbstractMatrix representation
Base.convert(::Type{AbstractArray}, L::AbstractDiffEqLinearOperator) = convert(AbstractMatrix, L)
LinearAlgebra.opnorm(L::AbstractDiffEqLinearOperator, p::Real=2) = opnorm(convert(AbstractMatrix,L), p)
Base.@propagate_inbounds Base.getindex(L::AbstractDiffEqLinearOperator, I::Vararg{Any,N}) where {N} = convert(AbstractMatrix,L)[I...]
Base.getindex(L::AbstractDiffEqLinearOperator, I::Vararg{Int, N}) where {N} =
  convert(AbstractMatrix,L)[I...]
for op in (:*, :/, :\)
    @eval Base.$op(L::AbstractDiffEqLinearOperator, x::AbstractArray) = $op(convert(AbstractMatrix,L), x)
    @eval Base.$op(x::AbstractArray, L::AbstractDiffEqLinearOperator) = $op(x, convert(AbstractMatrix,L))
end
LinearAlgebra.mul!(Y::AbstractArray, L::AbstractDiffEqLinearOperator, B::AbstractArray) =
  mul!(Y, convert(AbstractMatrix,L), B)
LinearAlgebra.mul!(Y::AbstractArray, L::AbstractDiffEqLinearOperator, B::AbstractArray, α::Number, β::Number) =
  mul!(Y, convert(AbstractMatrix,L), B, α, β)
for pred in (:isreal, :issymmetric, :ishermitian, :isposdef)
    @eval LinearAlgebra.$pred(L::AbstractDiffEqLinearOperator) = $pred(convert(AbstractArray, L))
end
for op in (:sum,:prod)
  @eval LinearAlgebra.$op(L::AbstractDiffEqLinearOperator; kwargs...) = $op(convert(AbstractArray, L); kwargs...)
end

# Routines that use the full matrix representation
Base.Matrix(L::AbstractDiffEqLinearOperator) = Matrix(convert(AbstractMatrix, L))
LinearAlgebra.exp(L::AbstractDiffEqLinearOperator) = exp(Matrix(L))

# write fallback interfae for adjoint operator here
Base.adjoint(A::AbstractDiffEqLinearOperator) = Adjoint(A)
