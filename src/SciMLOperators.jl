module SciMLOperators

using LinearAlgebra
using DiffEqBase
import StaticArrays
import SparseArrays
import ArrayInterfaceCore

## caching
#import UnPack: @unpack
#import Setfield: @set!

# overloads
import Lazy: @forward
import Base: size, +, -, *, /, \, adjoint, ∘, inv, one, convert, ==
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!, factorize

# Misc

#"""
#$(TYPEDEF)
#"""
abstract type AbstractSciMLOperator{T} end

#"""
#$(TYPEDEF)
#"""
abstract type AbstractDiffEqOperator{T} <: AbstractSciMLOperator{T} end

#"""
#$(TYPEDEF)
#"""
abstract type AbstractDiffEqLinearOperator{T} <: AbstractDiffEqOperator{T} end

#"""
#$(TYPEDEF)
#"""
abstract type AbstractDiffEqCompositeOperator{T} <: AbstractDiffEqLinearOperator{T} end
#"""
#$(TYPEDEF)
#"""
abstract type AbstractMatrixFreeOperator{T} <: AbstractDiffEqLinearOperator{T} end

include("interface.jl")
include("operators/basic_operators.jl")
include("operators/common_defaults.jl")
include("operators/diffeq_operator.jl")
include("operators/matrixfree_operators.jl")
include("operators/composite_operators.jl")

# The (u,p,t) and (du,u,p,t) interface
for T in (
          DiffEqIdentity, DiffEqNullOperator,
          DiffEqScalar,
          DiffEqArrayOperator, FactorizedDiffEqArrayOperator,
          DiffEqScaledOperator, DiffEqOperatorCombination, DiffEqOperatorComposition,
         )
    (L::T)(u, p, t) = (update_coefficients!(L, u, p, t); L * u)
    (L::T)(du, u, p, t) = (update_coefficients!(L, u, p, t); mul!(du, L, u))
end

export DiffEqIdentity, DiffEqNullOperator,
       DiffEqScalar, DiffEqArrayOperator, FactorizedDiffEqArrayOperator,
       AffineDiffEqOperator, DiffEqScaledOperator,
       MatrixFreeOperator

export update_coefficients!, update_coefficients,
       issquare, isconstant, islinear,
       has_adjoint, has_expmv!, has_expmv, has_exp, has_mul, has_mul!, has_ldiv, has_ldiv!

end # module
