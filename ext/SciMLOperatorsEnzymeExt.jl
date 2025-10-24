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

# For operator types with function fields, we need to tell Enzyme that the operators
# themselves are inactive during forward/reverse passes - the differentiation happens
# through the mathematical operations (mul!, ldiv!, etc.) not through the operator structures.
# The function fields (update_func) are just code that computes coefficients.

# Mark specific scalar and matrix operator types that have function fields as inactive
function Enzyme.EnzymeRules.inactive_type(::Type{<:SciMLOperators.AbstractSciMLScalarOperator})
    true
end
Enzyme.EnzymeRules.inactive_type(::Type{<:SciMLOperators.AbstractSciMLOperator}) = true

# Note: The actual differentiation will happen through the mathematical operations
# (mul!, *, +, etc.) which Enzyme knows how to handle natively.

end # module
