module SciMLOperators

using LinearAlgebra
using DiffEqBase
using DocStringExtensions

import StaticArrays
import SparseArrays
import ArrayInterfaceCore

# caching
import UnPack: @unpack
import Setfield: @set!

# overloads
import Lazy: @forward
import Base: size, +, -, *, /, \, adjoint, ∘, inv, one, convert, Matrix, iszero, ==
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!, factorize, exp

using DocStringExtensions

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

"""
$(TYPEDEF)
"""
abstract type AbstractMatrixFreeOperator{T} <: AbstractDiffEqLinearOperator{T} end

include("interface.jl")
include("operators/basic_operators.jl")
include("operators/diffeq_operator.jl")
include("operators/matrixfree_operators.jl")
#include("operators/composite_operators.jl")
include("operators/common_defaults.jl")

# Define a helper function `sparse1` that handles
# `DiffEqArrayOperator` and `ScaledDiffEqOperator`.
# We should define `sparse` for these types in `SciMLBase` instead,
# but that package doesn't know anything about sparse arrays yet, so
# we'll introduce a temporary work-around here.
_sparse(L) = sparse(L)
_sparse(L::DiffEqArrayOperator) = _sparse(L.A)
_sparse(L::ScaledDiffEqOperator) = L.λ * _sparse(L.L)

# (u,p,t) and (du,u,p,t) interface
for T in (
          DiffEqIdentity,
          DiffEqNullOperator,
          DiffEqScalar,
          ScaledDiffEqOperator,
          AddedDiffEqOperator,

          DiffEqArrayOperator,
          FactorizedDiffEqArrayOperator,

#         DiffEqOperatorCombination,
#         DiffEqOperatorComposition,
         )

    (L::T)(u, p, t) = (update_coefficients!(L, u, p, t); L * u)
    (L::T)(du, u, p, t) = (update_coefficients!(L, u, p, t); mul!(du, L, u))
end

export DiffEqScalar,
       DiffEqArrayOperator,
       FactorizedDiffEqArrayOperator,
       AffineDiffEqOperator,
       MatrixFreeOperator

export update_coefficients!,
       update_coefficients,

       has_adjoint,
       has_expmv!,
       has_expmv,
       has_exp,
       has_mul,
       has_mul!,
       has_ldiv,
       has_ldiv!

end # module
