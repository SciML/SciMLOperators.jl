module SciMLOperators

using DocStringExtensions

using LinearAlgebra
import SparseArrays
import StaticArraysCore
import ArrayInterface
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

## Interface

An `L::AbstractSciMLOperator` can be called like a function. This behaves
like multiplication by the linear operator represented by the
`AbstractSciMLOperator`. Possible signatures are

- `L(du, u, p, t)` for in-place operator evaluation
- `du = L(u, p, t)` for out-of-place operator evaluation

If the operator is not a constant, update it with `(u, p, t)`.
A mutating form, i.e. `update_coefficients!(L, u, p, t)` that changes the
internal coefficients, and an out-of-place form
`L_new = update_coefficients(L, u, p, t)`.
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
       iscached,
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
