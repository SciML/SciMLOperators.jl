"""
$(README)
"""
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
import Base: show
import Base: zero, one, oneunit
import Base: +, -, *, /, \, ∘, ==, conj, exp, kron
import Base: iszero, inv, adjoint, transpose, size, convert
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!, factorize
import LinearAlgebra: Matrix, Diagonal
import SparseArrays: sparse, issparse

"""
$(TYPEDEF)

Subtypes of `AbstractSciMLOperator` represent linear, nonlinear,
time-dependent operators acting on vectors, or matrix column-vectors.
A lazy operator algebra is also defined for `AbstractSciMLOperator`s.

# Interface

An `AbstractSciMLOperator` can be called like a function. This behaves
like multiplication by the linear operator represented by the
`AbstractSciMLOperator`. Possible signatures are

- `L(du, u, p, t)` for in-place operator evaluation
- `du = L(u, p, t)` for out-of-place operator evaluation

Operator evaluation methods update its coefficients with `(u, p, t)`
information using the `update_coefficients(!)` method. The methods
are exported and can be called as follows:

- `update_coefficients!(L, u, p, t)` for out-of-place operator update
- `L = update_coefficients(L, u, p, t)` for in-place operator update

SciMLOperators also overloads `Base.*`, `LinearAlgebra.mul!`,
`LinearAlgebra.ldiv!` for operator evaluation without updating operator state.
An `AbstractSciMLOperator` behaves like a matrix in these methods.
Allocation-free methods, suffixed with a `!` often need cache arrays.
To precache an `AbstractSciMLOperator`, call the function
`L = cache_operator(L, input_vector)`.

# Methods
"""
abstract type AbstractSciMLOperator{T} end

"""
$(TYPEDEF)

An `AbstractSciMLScalarOperator` represents a linear scaling operation
that may be applied to `Number`, `AbstractVecOrMat` subtypes. Addition,
multiplication, division of `AbstractSciMLScalarOperator`s is defined
lazily so operator state may be updated.
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
