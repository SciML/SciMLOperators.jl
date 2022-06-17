module SciMLOperators

using DocStringExtensions

using LinearAlgebra
import StaticArrays
import SparseArrays
import ArrayInterfaceCore
import Base: ReshapedArray
import Lazy: @forward
import Setfield: @set!

# overload
import Base: +, -, *, /, \, ∘, ==, one, zero, conj, exp, kron
import Base: iszero, inv, adjoint, transpose, size, convert
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!, factorize
import LinearAlgebra: Matrix, Diagonal
import SparseArrays: sparse

"""
$(TYPEDEF)
"""
abstract type AbstractSciMLOperator{T} end

"""
$(TYPEDEF)
"""
abstract type AbstractSciMLLinearOperator{T} <: AbstractSciMLOperator{T} end

include("interface.jl")
include("utils.jl")
include("left.jl")
include("multidim.jl")
include("basic.jl")
include("sciml.jl")

export ScalarOperator,
       MatrixOperator,
       DiagonalOperator,
       AffineOperator,
       FunctionOperator,
       TensorProductOperator

export update_coefficients!,
       update_coefficients,

       cache_operator,

       has_adjoint,
       has_expmv,
       has_expmv!,
       has_exp,
       has_mul,
       has_mul!,
       has_ldiv,
       has_ldiv!

end # module
