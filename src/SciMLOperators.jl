"""
$(README)
"""
module SciMLOperators

using DocStringExtensions

using LinearAlgebra

import ArrayInterface
import MacroTools: @forward
import Accessors: @reset

# overload
import Base: show
import Base: zero, one, oneunit
import Base: +, -, *, /, \, ∘, ==, conj, exp, kron
import Base: iszero, inv, adjoint, transpose, size, convert
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!, factorize
import LinearAlgebra: Matrix, Diagonal

"""
$(TYPEDEF)

Subtypes of `AbstractSciMLOperator` represent linear, nonlinear,
time-dependent operators acting on vectors, or matrix column-vectors.
A lazy operator algebra is also defined for `AbstractSciMLOperator`s.

# Mathematical Notation

An `AbstractSciMLOperator` ``L`` is an operator which is used to represent
the following type of equation:

```math
w = L(u,p,t)[v]
```

where `L[v]` is the operator application of ``L`` on the vector ``v``. 

# Interface

An `AbstractSciMLOperator` can be called  like a function in the following ways:

- `L(v, u, p, t)` - Out-of-place application where `v` is the action vector and `u` is the update vector
- `L(w, v, u, p, t)` - In-place application where `w` is the destination, `v` is the action vector, and `u` is the update vector
- `L(w, v, u, p, t, α, β)` - In-place application with scaling: `w = α*(L*v) + β*w`

Operator state can be updated separately from application:

- `update_coefficients!(L, u, p, t)` for in-place operator update
- `L = update_coefficients(L, u, p, t)` for out-of-place operator update

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

include("scalar.jl")
include("matrix.jl")
include("basic.jl")
include("batch.jl")
include("func.jl")
include("tensor.jl")

export
       IdentityOperator,
       NullOperator,
       ScalarOperator,
       MatrixOperator,
       DiagonalOperator,
       InvertibleOperator,
       AffineOperator,
       AddVector,
       FunctionOperator,
       TensorProductOperator

export update_coefficients!,
       update_coefficients, isconstant,
       iscached,
       cache_operator, issquare,
       islinear,
       concretize,
       isconvertible, has_adjoint,
       has_expmv,
       has_expmv!,
       has_exp,
       has_mul,
       has_mul!,
       has_ldiv,
       has_ldiv!

end # module
