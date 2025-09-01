module SciMLOperatorsChainRulesCoreExt

using SciMLOperators
using ChainRulesCore
import SciMLOperators: ScaledOperator, ScalarOperator, AbstractSciMLOperator

"""
Fix for gradient double-counting issue in ScaledOperator constructor.

The issue: When creating ScaledOperator(λ, L) where λ is a ScalarOperator with parameter 
dependencies, Zygote was double-counting gradients because:
1. Gradient flows through the ScalarOperator's creation/value
2. Gradient also flows through the ScalarOperator being stored as a struct field

This rrule ensures gradients are only counted once by carefully managing the pullback
to avoid the structural dependency double-counting.

Fixes issue: https://github.com/SciML/SciMLOperators.jl/issues/305
"""
function ChainRulesCore.rrule(::Type{ScaledOperator}, λ::ScalarOperator, L::AbstractSciMLOperator)
    # Forward pass - same as original constructor
    result = ScaledOperator(λ, L)
    
    function ScaledOperator_pullback(Ȳ)
        # Handle gradients carefully to avoid double-counting for ScalarOperator
        # The key insight: gradients should flow through ScalarOperator creation
        # but NOT through struct field access
        
        if hasfield(typeof(Ȳ), :λ) && getfield(Ȳ, :λ) isa ChainRulesCore.AbstractTangent
            λ_tangent = getfield(Ȳ, :λ)
            # For ScalarOperator, only propagate through the value to avoid double-counting
            if hasfield(typeof(λ_tangent), :val)
                ∂λ = ChainRulesCore.Tangent{typeof(λ)}(val=getfield(λ_tangent, :val))
            else
                ∂λ = λ_tangent
            end
        else
            ∂λ = NoTangent()
        end
        
        if hasfield(typeof(Ȳ), :L) && getfield(Ȳ, :L) isa ChainRulesCore.AbstractTangent
            ∂L = getfield(Ȳ, :L)
        else
            ∂L = NoTangent()
        end
        
        return (NoTangent(), ∂λ, ∂L)
    end
    
    return result, ScaledOperator_pullback
end

end # module