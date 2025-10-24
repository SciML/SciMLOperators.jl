module SciMLOperatorsEnzymeExt

using SciMLOperators
using Enzyme
using LinearAlgebra

# The issue with Enzyme and SciMLOperators is that operators have mutable state
# (like ScalarOperator.val and MatrixOperator.A) and update functions stored as closures.
# Enzyme needs special handling for these cases.

# Mark utility function types as inactive since they're just code, not data to differentiate
function Enzyme.EnzymeRules.inactive(::typeof(SciMLOperators.DEFAULT_UPDATE_FUNC), args...)
    return true
end

function Enzyme.EnzymeRules.inactive(::typeof(SciMLOperators.preprocess_update_func), args...)
    return true
end

function Enzyme.EnzymeRules.inactive_type(::Type{SciMLOperators.NoKwargFilter})
    return true
end

# The key insight: Function-typed fields in operators are code (update functions),
# not differentiable data. Tell Enzyme to treat them as inactive.
# This prevents Enzyme from trying to differentiate through closure captures.
function Enzyme.EnzymeRules.inactive_type(::Type{F}) where {F <: Function}
    return true
end

# Note: The actual differentiation will happen through the mathematical operations
# (mul!, *, +, etc.) which Enzyme knows how to handle natively. The operator
# structures just orchestrate these operations.

end # module
