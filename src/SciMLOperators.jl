"""
SciMLOperators provides a common interface and lazy algebra for matrix-like,
matrix-free, state-dependent, and time-dependent operators used throughout
the SciML ecosystem.
"""
module SciMLOperators

using DocStringExtensions: DocStringExtensions, FIELDS, SIGNATURES, TYPEDEF
using SciMLPublic: @public

using LinearAlgebra: LinearAlgebra, Adjoint, Bidiagonal, Factorization, I,
    Transpose, UniformScaling, axpby!, axpy!, ishermitian,
    isposdef, issuccess, issymmetric, lu, opnorm

import ArrayInterface
# MacroTools dependency removed - using explicit method forwarding instead
import Accessors: @reset

using Adapt: Adapt

# overload
import Base: show
import Base: zero, one
import Base: +, -, *, /, \, ==, conj, exp, kron
import Base: iszero, inv, adjoint, transpose, size, convert
import Base: Matrix
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!, factorize
import LinearAlgebra: Diagonal

# Used for downstream checking
const isv1 = true

"""
$(TYPEDEF)

`AbstractSciMLOperator` is the extension point for matrix-like and
matrix-free operators. A subtype represents an operator ``L(u,p,t)`` whose
action on an array ``v`` is written ``L(u,p,t)v``. The subtype may be
constant, state-dependent, or time-dependent, and may be composed with other
SciML operators through the lazy algebra.

This is an interface type, not a constructor. The concrete type should be
public only when users are expected to construct or extend it; otherwise use
the qualified developer-facing API documented in this section.

## Mathematical Notation

An `AbstractSciMLOperator` ``L`` is an operator which is used to represent
the following type of equation:

```math
w = L(u,p,t)[v]
```

where `L[v]` is the operator application of ``L`` on the vector ``v``.

## Construction and Extension Rules

`AbstractSciMLOperator` is an interface, not a concrete constructor. New
operator types should subtype it with the scalar element type `T` and must
preserve their mathematical action when they participate in lazy algebra.

## Required Interface

A concrete subtype must implement `size(L) -> (m, n)`, `*(L, v)`, and
`mul!(w, L, v)`. The returned action has leading size `m` for an input whose
leading size is `n`, and the in-place method must return `w` after writing the
result. The scaling form `mul!(w, L, v, α, β)` is required when
`has_mul!(L) == true`; it must compute ``w \\leftarrow α(Lv) + βw``.

`has_mul(L)`, `has_mul!(L)`, `has_ldiv(L)`, and `has_ldiv!(L)` are promises,
not capability probes: return `true` only when the corresponding operation is
valid for all compatible inputs. `convert(AbstractMatrix, L)` is optional and
should be defined only when `isconvertible(L) == true`; its result must have
the same size and action as `L` in its current state.

An `AbstractSciMLOperator` can be called like a function in the following ways:

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

## Required Interface For Subtypes

A concrete subtype must define `Base.size(L)` and one of the following
application paths:

  - `Base.:*(L, v)` for out-of-place matrix-like application.
  - `LinearAlgebra.mul!(w, L, v)` and, when `has_mul!(L)` is `true`,
    `LinearAlgebra.mul!(w, L, v, α, β)` for in-place application.
  - `Base.convert(AbstractMatrix, L)` when `isconvertible(L)` is `true`.

If the operator state depends on `(u, p, t)` or accepted keyword arguments,
the subtype must implement `update_coefficients` for out-of-place state
updates or `update_coefficients!` for in-place state updates. The out-of-place
form returns a new operator and leaves `L` unchanged; the in-place form
returns `nothing`. Composite operators assume these update methods may be
called recursively on every operator returned by `getops(L)`.

The positional arguments are forwarded unchanged through a composite
operator. `u` is the state supplied by the caller and is not necessarily the
same shape as the action vector `v`; an operator whose action is nonlinear in
`v` should generally use `FunctionOperator` and report `islinear(L) == false`.
For a constant leaf, the default update is a no-op. A stateful leaf must
override `isconstant` rather than inheriting the empty-child default.

Subtypes that need preallocated work arrays for allocation-free application
must implement `cache_self(L, v)` for their own caches, `cache_internals(L, v)`
for child-operator caches, or both. `cache_operator(L, v)` calls these hooks
and downstream solvers may call it before repeated `mul!` evaluations.

## Caching Rules

`cache_operator(L, v)` may return either `L` or a cached replacement. A
subtype that advertises `has_mul!(L) == true` must ensure the cached result is
ready for repeated `mul!` calls with compatible vectors. Cache hooks may not
change the mathematical action, dimensions, or trait values of the operator.
A cached operator's scratch is mutable and is not safe to use concurrently
unless the subtype explicitly provides that guarantee.

Composite types expose their children through the developer-facing `getops`
method. A new composite must forward state updates, caching, and traits to
every child that contributes to its action. The public action must remain
unchanged by flattening, caching, or updating the composition.

## Trait Rules

Trait functions such as `isconstant`, `islinear`, `isconvertible`,
`has_concretization`, `has_mul`, `has_mul!`, `has_ldiv`, and `has_ldiv!`
are part of the public operator interface. A trait returning `true` is a
promise that the corresponding operation is valid for inputs with compatible
sizes. For example, `has_mul!(L)` means `mul!(w, L, v)` is available, and
`has_concretization(L)` means either `convert(AbstractMatrix, L)` or
`convert(Number, L)` can materialize the operator state without changing its
mathematical action.

`isconstant(L)` means repeated calls to `update_coefficients[!]` are not
required to keep `L` current. `islinear(L)` means the action is linear in
the vector being multiplied; state dependence on `(u, p, t)` is still allowed
for a linear operator.

## Keyword Arguments

When an operator accepts keywords during updates, its constructor must record
the accepted names with `accepted_kwargs`, normally as
`Val((:name1, :name2))`. Composite operators forward only those accepted
keywords to each component. Extension authors must therefore accept
`(u, p, t; kwargs...)` consistently in every update and application method
they advertise. An unlisted keyword must not be silently passed to a leaf
update function.

## Standard Actions

The behavior of a `SciMLOperator` is
indistinguishable from an `AbstractMatrix`. These operators can be
passed to linear solver packages, and even to ordinary differential
equation solvers. The list of overloads to the `AbstractMatrix`
interface includes, but is not limited to, the following:

  - `Base: size, zero, one, +, -, *, /, \\, ∘, inv, adjoint, transpose, convert`
  - `LinearAlgebra: mul!, ldiv!, lmul!, rmul!, factorize, issymmetric, ishermitian, isposdef`
  - `SparseArrays: sparse, issparse`

## Multidimensional arrays and batching

SciMLOperator can also be applied to `AbstractMatrix` subtypes where
operator-evaluation is done column-wise.

```julia
using LinearAlgebra, SciMLOperators

N = 4
K = 10
L = MatrixOperator(Matrix(I, N, N))
u_mat = rand(N, K)

v_mat = L(u_mat, nothing, nothing, 0.0)
size(v_mat) == (N, K) # true
```

`L` can also be applied to `AbstractArray`s that are not
`AbstractVecOrMat`s so long as their size in the first dimension is appropriate
for matrix-multiplication. Internally, `SciMLOperator`s reshapes an
`N`-dimensional array to an `AbstractMatrix`, and applies the operator via
matrix-multiplication.

## Operator update

This package can also be used to write state-dependent, time-dependent, and
parameter-dependent operators, whose state can be updated per
a user-defined function.
The updates can be done in-place, i.e. by mutating the object,
or out-of-place, i.e. in a non-mutating, `Zygote`-compatible way.

For example,

```julia
using LinearAlgebra, SciMLOperators

n = 4
v = rand(n)
u = rand(n)
p = rand(n)
t = rand()

# out-of-place update
mat_update_func = (A, u, p, t) -> t * (p * u')
sca_update_func = (a, u, p, t) -> t * sum(p)

M = MatrixOperator(zeros(n, n); update_func = mat_update_func)
α = ScalarOperator(0.0; update_func = sca_update_func)

L = α * M
L = cache_operator(L, v)

# L is initialized with zero state
L * v == zeros(n) # true

# update operator state with `(u, p, t)`
L = update_coefficients(L, u, p, t)
# and multiply
L * v != zeros(n) # true

# updates state and evaluates L*v at (u, p, t)
L(v, u, p, t) != zeros(n) # true
```

The out-of-place evaluation function `L(v, u, p, t)` calls
`update_coefficients` under the hood, which recursively calls
the `update_func` for each component `SciMLOperator`.
Therefore, the out-of-place evaluation function is equivalent to
calling `update_coefficients` followed by `Base.*`. Notice that
the out-of-place evaluation does not return the updated operator.

On the other hand, the in-place evaluation function, `L(w, v, u, p, t)`,
mutates `L`, and is equivalent to calling `update_coefficients!`
followed by `mul!`. The in-place update behavior works the same way,
with a few `<!>`s appended here and there. For example,

```julia
using LinearAlgebra, SciMLOperators

n = 4
w = rand(n)
v = rand(n)
u = rand(n)
p = rand(n)
t = rand()

# in-place update
_A = rand(n, n)
mat_update_func! = (A, u, p, t) -> (copy!(A, _A); lmul!(t, A); nothing)

M = MatrixOperator(zeros(n, n); update_func! = mat_update_func!)

L = M
L = cache_operator(L, v)

# L is initialized with zero state
L * v == zeros(n) # true

# update L in-place
update_coefficients!(L, v, p, t)
# and multiply
mul!(w, L, v) != zeros(n) # true

# updates L in-place, and evaluates w=L*v at (u, p, t)
L(w, v, u, p, t) != zeros(n) # true
```

The update behavior makes this package flexible enough to be used
in `OrdinaryDiffEq`. As the parameter object `p` is often reserved
for sensitivity computation via automatic-differentiation, a user may
prefer to pass in state information via other arguments. For that
reason, we allow update functions with arbitrary keyword arguments.

```julia
using SciMLOperators

n = 4
v = rand(n)
u = rand(n)
p = rand(n)
t = 0.0
mat_update_func = (A, u, p, t; scale = 0.0) -> scale * (p * u')

M = MatrixOperator(zeros(n, n); update_func = mat_update_func,
    accepted_kwargs = Val((:scale,)))

M(v, u, p, t) == zeros(n) # true
M(v, u, p, t; scale = 1.0) != zeros(n)
```
"""
abstract type AbstractSciMLOperator{T} end

"""
    AbstractSciMLScalarOperator{T} <: AbstractSciMLOperator{T}

Abstract interface for a scalar-valued linear scaling operator.

# Interface Rules

Subtypes must provide `convert(Number, operator)` for their current scalar
value and `eltype` through the type parameter `T`. Scalar application to an
array must preserve the array shape. If the subtype is stateful, its update
methods use the same `(u, p, t; kwargs...)` contract as
[`AbstractSciMLOperator`](@ref): the out-of-place form returns a new scalar
operator, while the in-place form mutates the operator and returns `nothing`.

`islinear` describes linearity in the array being scaled. `has_ldiv` and
`has_ldiv!` may be true only when the current scalar value is invertible.
Scalar addition, multiplication, division, and inversion remain lazy so that
later updates affect the composed expression. Use `ScalarOperator` when a
premade implementation is sufficient.

# Examples

```julia
using SciMLOperators

struct MutableScale <: AbstractSciMLScalarOperator{Float64}
    value::Float64
end

Base.convert(::Type{Number}, L::MutableScale) = L.value
Base.:*(L::MutableScale, v::AbstractArray) = L.value .* v
SciMLOperators.islinear(::MutableScale) = true

L = MutableScale(2.0)
L * [3.0, 4.0]
concretize(L) == 2.0
```

Use `ScalarOperator` to construct a concrete scalar operator. Custom scalar
operator types should only claim traits such as `has_ldiv` when the converted
scalar has the corresponding operation for the current state.
"""
abstract type AbstractSciMLScalarOperator{T} <: AbstractSciMLOperator{T} end

include("utils.jl")
include("interface.jl")
include("left.jl")

include("scalar.jl")
include("matrix.jl")
include("basic.jl")
include("block.jl")
include("batch.jl")
include("func.jl")
include("tensor.jl")
include("woperator.jl")
include("adapt.jl")

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
    BlockDiagonalOperator,
    TensorProductOperator,
    TensorSumOperator,
    WOperator,
    StaticWOperator,
    jacobian_stale,
    mark_jacobian_updated!,
    mark_jacobian_current!

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
    has_ldiv!,
    has_concretization,
    kronsum

# Documented but not exported: core abstract types, lazy-algebra result types,
# and developer extension hooks used through qualified access downstream.
@public AbstractSciMLOperator, AbstractSciMLScalarOperator,
    ScaledOperator, AddedOperator, ComposedOperator,
    InvertedOperator, AdjointOperator, TransposedOperator,
    AddedScalarOperator, ComposedScalarOperator, InvertedScalarOperator,
    has_tensor_outer_mul_fast, tensor_outer_mul_fast!,
    getcache, update_cache, adopt_cache

end # module
