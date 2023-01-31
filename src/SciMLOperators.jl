module SciMLOperators

using DocStringExtensions

using LinearAlgebra
import SparseArrays
import StaticArraysCore
import ArrayInterfaceCore
import Tricks: static_hasmethod
import Lazy: @forward
import Setfield: @set!

# overload
import Base: zero, one, oneunit
import Base: +, -, *, /, \, ∘, ==, conj, exp, kron
import Base: iszero, inv, adjoint, transpose, size, convert
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!, factorize
import LinearAlgebra: Matrix, Diagonal
import SparseArrays: sparse, issparse

"""
$(TYPEDEF)
"""
abstract type AbstractSciMLOperator{T} end

"""
$(TYPEDEF)
"""
abstract type AbstractSciMLScalarOperator{T} <: AbstractSciMLOperator{T} end

include("utils.jl")
include("interface.jl")
include("left.jl")
include("multidim.jl")

include("scalar.jl")
include("matrix.jl")
include("basic.jl")
include("batch.jl")
include("func.jl")
include("tensor.jl")

export ScalarOperator,
       MatrixOperator,
       DiagonalOperator,
       AffineOperator,
       AddVector,
       FunctionOperator,
       TensorProductOperator

export update_coefficients!,
       update_coefficients,

       isconstant,

       cache_operator,

       issquare,
       islinear,

       has_adjoint,
       has_expmv,
       has_expmv!,
       has_exp,
       has_mul,
       has_mul!,
       has_ldiv,
       has_ldiv!

end # module
