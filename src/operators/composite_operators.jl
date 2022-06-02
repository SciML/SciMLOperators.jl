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
