module SciMLOperators

using LinearAlgebra
using DocStringExtensions

import StaticArrays
import SparseArrays
import ArrayInterfaceCore

import Lazy: @forward
import Setfield: @set!
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
abstract type AbstractSciMLLinearOperator{T} <: AbstractSciMLOperator{T} end

"""
$(TYPEDEF)
"""
abstract type AbstractSciMLCompositeOperator{T} <: AbstractSciMLOperator{T} end

"""
$(TYPEDEF)
"""
abstract type AbstractMatrixFreeOperator{T} <: AbstractSciMLOperator{T} end

include("interface.jl")
include("basic.jl")
include("sciml.jl")
include("common.jl")

# Define a helper function `sparse1` that handles
# `SciMLMatrixOperator` and `SciMLScaledOperator`.
# We should define `sparse` for these types in `SciMLBase` instead,
# but that package doesn't know anything about sparse arrays yet, so
# we'll introduce a temporary work-around here.
_sparse(L) = sparse(L)
_sparse(L::SciMLMatrixOperator) = _sparse(L.A)
_sparse(L::SciMLScaledOperator) = L.λ * _sparse(L.L)

# (u,p,t) and (du,u,p,t) interface
for T in (
          SciMLIdentity,
          SciMLNullOperator,
          SciMLScalar,
          SciMLScaledOperator,
          SciMLAddedOperator,
          SciMLComposedOperator,

          SciMLMatrixOperator,
          SciMLFactorizedOperator,
#         SciMLFunctionOperator,
         )

    (L::T)(u, p, t) = (update_coefficients!(L, u, p, t); L * u)
    (L::T)(du, u, p, t) = (update_coefficients!(L, u, p, t); mul!(du, L, u))
end

export SciMLScalar,
       SciMLMatrixOperator,
       SciMLFactorizedOperator,
       AffineSciMLOperator,
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
