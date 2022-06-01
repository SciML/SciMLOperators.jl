# Composition (A ∘ B)
struct DiffEqOperatorComposition{T,O<:Tuple{Vararg{AbstractDiffEqLinearOperator{T}}},
  C<:Tuple{Vararg{AbstractVector{T}}}} <: AbstractDiffEqCompositeOperator{T}
  ops::O # stored in the order of application
  caches::C
  function DiffEqOperatorComposition(ops; caches=nothing)
    T = eltype(ops[1])
    for i in 2:length(ops)
      @assert size(ops[i-1], 1) == size(ops[i], 2) "Operations do not have compatible sizes! Mismatch between $(ops[i]) and $(ops[i-1]), which are operators $i and $(i-1) respectively."
    end

    if caches === nothing
      # Construct a list of caches to be used by mul! and ldiv!
      caches = []
      for op in ops[1:end-1]
        tmp = Vector{T}(undef, size(op, 1))
        fill!(tmp,0)
        push!(caches,tmp)
      end
      caches = tuple(caches...)
    end
    new{T,typeof(ops),typeof(caches)}(ops, caches)
  end
end
# this is needed to not break dispatch in MethodOfLines
function Base.:*(ops::AbstractDiffEqLinearOperator...) 
  try 
    return DiffEqOperatorComposition(reverse(ops))
  catch e
    return 1
  end
end
∘(L1::AbstractDiffEqLinearOperator, L2::AbstractDiffEqLinearOperator) = DiffEqOperatorComposition((L2, L1))
Base.:*(L1::DiffEqOperatorComposition, L2::AbstractDiffEqLinearOperator) = DiffEqOperatorComposition((L2, L1.ops...))
∘(L1::DiffEqOperatorComposition, L2::AbstractDiffEqLinearOperator) = DiffEqOperatorComposition((L2, L1.ops...))
Base.:*(L1::AbstractDiffEqLinearOperator, L2::DiffEqOperatorComposition) = DiffEqOperatorComposition((L2.ops..., L1))
∘(L1::AbstractDiffEqLinearOperator, L2::DiffEqOperatorComposition) = DiffEqOperatorComposition((L2.ops..., L1))
Base.:*(L1::DiffEqOperatorComposition, L2::DiffEqOperatorComposition) = DiffEqOperatorComposition((L2.ops..., L1.ops...))
∘(L1::DiffEqOperatorComposition, L2::DiffEqOperatorComposition) = DiffEqOperatorComposition((L2.ops..., L1.ops...))
getops(L::DiffEqOperatorComposition) = L.ops
Matrix(L::DiffEqOperatorComposition) = prod(Matrix, reverse(L.ops))
convert(::Type{AbstractMatrix}, L::DiffEqOperatorComposition) =
  prod(op -> convert(AbstractMatrix, op), reverse(L.ops))
SparseArrays.sparse(L::DiffEqOperatorComposition) = prod(sparse1, reverse(L.ops))

size(L::DiffEqOperatorComposition) = (size(L.ops[end], 1), size(L.ops[1], 2))
size(L::DiffEqOperatorComposition, m::Integer) = size(L)[m]
opnorm(L::DiffEqOperatorComposition) = prod(opnorm, L.ops)
Base.:*(L::DiffEqOperatorComposition, x::AbstractArray) = foldl((acc, op) -> op*acc, L.ops; init=x)
Base.:*(x::AbstractArray, L::DiffEqOperatorComposition) = foldl((acc, op) -> acc*op, reverse(L.ops); init=x)
/(L::DiffEqOperatorComposition, x::AbstractArray) = foldl((acc, op) -> op/acc, L.ops; init=x)
/(x::AbstractArray, L::DiffEqOperatorComposition) = foldl((acc, op) -> acc/op, L.ops; init=x)
\(L::DiffEqOperatorComposition, x::AbstractArray) = foldl((acc, op) -> op\acc, reverse(L.ops); init=x)
\(x::AbstractArray, L::DiffEqOperatorComposition) = foldl((acc, op) -> acc\op, reverse(L.ops); init=x)
function mul!(y::AbstractVector, L::DiffEqOperatorComposition, b::AbstractVector)
  mul!(L.caches[1], L.ops[1], b)
  for i in 2:length(L.ops) - 1
    mul!(L.caches[i], L.ops[i], L.caches[i-1])
  end
  mul!(y, L.ops[end], L.caches[end])
end
function ldiv!(y::AbstractVector, L::DiffEqOperatorComposition, b::AbstractVector)
  ldiv!(L.caches[end], L.ops[end], b)
  for i in length(L.ops) - 1:-1:2
    ldiv!(L.caches[i-1], L.ops[i], L.caches[i])
  end
  ldiv!(y, L.ops[1], L.caches[1])
end
factorize(L::DiffEqOperatorComposition) = prod(factorize, reverse(L.ops))
for fact in (:lu, :lu!, :qr, :qr!, :cholesky, :cholesky!, :ldlt, :ldlt!,
  :bunchkaufman, :bunchkaufman!, :lq, :lq!, :svd, :svd!)
  @eval LinearAlgebra.$fact(L::DiffEqOperatorComposition, args...) =
    prod(op -> $fact(op, args...), reverse(L.ops))
end
