module SciMLOperators

using DocStringExtensions

using LinearAlgebra
import StaticArrays
import SparseArrays
import ArrayInterfaceCore

import Lazy: @forward
import Setfield: @set!
import Base: size, +, -, *, /, \, adjoint, ∘, inv, one, convert, Matrix, iszero, ==
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!, factorize, exp

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

include("traits.jl")
include("basic.jl")
include("sciml.jl")
include("interface.jl")
include("utils.jl")

export SciMLScalar,
       SciMLMatrixOperator,
       SciMLFactorizedOperator,
       AffineSciMLOperator,
       SciMLFunctionOperator

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
