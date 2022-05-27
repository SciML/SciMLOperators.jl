module SciMLOperators

# Misc
"""
$(TYPEDEF)
"""
abstract type AbstractSciMLOperator{T} end

"""
$(TYPEDEF)
"""
abstract type AbstractDiffEqOperator{T} <: AbstractSciMLOperator{T} end

"""
$(TYPEDEF)
"""
abstract type AbstractDiffEqLinearOperator{T} <: AbstractDiffEqOperator{T} end

"""
$(TYPEDEF)
"""
abstract type AbstractDiffEqCompositeOperator{T} <: AbstractDiffEqLinearOperator{T} end

include("interface.jl")
include("operators/basic_operators.jl")
include("operators/diffeq_operator.jl")
include("operators/common_defaults.jl")


export AffineDiffEqOperator, DiffEqScaledOperator

export DiffEqScalar, DiffEqArrayOperator, DiffEqIdentity

export update_coefficients!, update_coefficients,
    has_adjoint, has_expmv!, has_expmv, has_exp, has_mul, has_mul!, has_ldiv, has_ldiv!

end # module
