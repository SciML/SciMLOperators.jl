"""
    AbstractWOperator{T} <: AbstractSciMLOperator{T}

Developer-facing interface for operators representing the implicit-solver
matrix ``W = J - MM / gamma``. This type is intentionally not exported and
should not be used as a user-facing construction target.

# Interface Rules

A concrete subtype must provide `size`, matrix-like `*`, and `mul!` with the
same shape and return-value rules as [`AbstractSciMLOperator`](@ref). It must
also expose enough state for the implicit solver that owns it to update the
Jacobian and mass-matrix actions. Define `isconvertible` and
`convert(AbstractMatrix, W)` only when materialization is supported.

If a consumer caches a Jacobian-dependent factorization, it should implement
the [`jacobian_stale`](@ref), [`mark_jacobian_updated!`](@ref), and
[`mark_jacobian_current!`](@ref) protocol or keep an equivalent private
generation for its own subtype.
"""
abstract type AbstractWOperator{T} <: AbstractSciMLOperator{T} end

"""
$(TYPEDEF)

Small dense factorization helper for repeated solves with a fixed `W` matrix.

# Arguments

  - `W`: Concrete square matrix to solve against.
  - `callinv`: Whether very small matrices may store `inv(W)` for direct
    multiplication during `\\`.

# Fields

$(FIELDS)

# Interface Rules

`StaticWOperator` is a specialized helper for solver internals that need a
fixed W-operator solve. It supports `Wstatic \\ v`; it does not participate in
coefficient updates and should be reconstructed when the underlying matrix
changes. This is a developer-facing helper, not a general-purpose
factorization type; user code should normally use `factorize` or a solver's
documented linear-operator interface instead.

# Returns

`StaticWOperator(W, callinv) \\ v` returns the solution of `W * x = v`.

# Errors

The constructor and solve throw the same errors as the selected inverse or
factorization for nonsquare or singular input.

# Examples

```julia
using LinearAlgebra, SciMLOperators

W = StaticWOperator([2.0 0.0; 0.0 4.0])
W \\ [2.0, 8.0]
```
"""
struct StaticWOperator{isinv, T, F} <: AbstractWOperator{T}
    W::T
    F::F
    function StaticWOperator(W::T, callinv = true) where {T}
        n = size(W, 1)
        isinv = n <= 7 # cutoff where LU has noticeable overhead

        F = if isinv && callinv
            ArrayInterface.lu_instance(W)
        else
            lu(W, check = false)
        end
        # when constructing W for the first time for the type
        # inv(W) can be singular
        _W = if isinv && callinv
            inv(W)
        else
            W
        end
        return new{isinv, T, typeof(F)}(_W, F)
    end
end
Base.:\(W::StaticWOperator{isinv}, v::AbstractArray) where {isinv} = isinv ? W.W * v : W.F \ v

"""
$TYPEDEF

    WOperator{IIP}(mass_matrix, gamma, J, u[, jacvec])

A linear operator that represents the W matrix used by implicit ODE solvers,
defined as

```math
W = J - \\frac{1}{\\gamma}MM
```

where `MM` is the mass matrix, `gamma` is a scalar, and `J` is the Jacobian
operator. The sign convention matches the matrix action implemented by
`WOperator`: `W * x == J * x - (MM * x) / gamma`.

# Arguments

  - `mass_matrix`: A matrix-like object, `UniformScaling`, or `MatrixOperator`
    representing `MM`.
  - `gamma`: Scalar coefficient in the W-operator definition.
  - `J`: Jacobian represented as a number, matrix, or `AbstractSciMLOperator`.
  - `u`: Prototype state used to allocate the internal multiplication cache.
  - `jacvec`: Optional operator used for Jacobian-vector products in `mul!`.

# Fields

$(FIELDS)

The fields have the following interface-relevant meanings:

  - `mass_matrix`: The current mass-matrix action `MM`.
  - `gamma`: The scalar in the mass-matrix term.
  - `J`: The current Jacobian action.
  - `jacvec`: Optional replacement for `J` in matrix-free `mul!` calls.
  - `jac_stale`: Whether a consumer has been told that the Jacobian changed
    since its last factorization.

`_func_cache` and `_concrete_form` are implementation storage and are not
part of the extension contract.

# Interface Rules

`WOperator` is part of the public solver-developer interface used by implicit
ODE solvers. It supports matrix-like `*`, `\\`, `mul!`, indexing, sizing, and
concretization. Calling `update_coefficients!(W, u, p, t; gamma)` updates the
Jacobian, mass matrix, optional Jacobian-vector operator, and stored `gamma`.
Omitting `(u, p, t)` leaves those operators unchanged and only updates `gamma`
when it is supplied. The caller must call
[`mark_jacobian_updated!`](@ref) after mutating the contents of `J` in place;
the operator cannot observe that mutation automatically.

`IIP` controls whether conversion reuses the internally stored concrete form
as an in-place operator. The public contract is the mathematical action of
`W`; downstream code should not depend on `_func_cache` or `_concrete_form`.
This type is intended for solver developers extending the implicit-solver
interface. End users should construct it only when a solver's documentation
explicitly requests a `WOperator`.

# Returns

`WOperator{IIP}(mass_matrix, gamma, J, u, jacvec)` returns a mutable operator
with the size of `J`. `*`, `mul!`, `\\`, and `concretize` use the current
values of its fields and preserve the matrix action above.

# Errors

Construction or application can throw the dimension, conversion, or
factorization errors raised by the supplied mass matrix and Jacobian. A
matrix-free `J` or mass matrix cannot be materialized by
`convert(AbstractMatrix, W)` unless both operands support that conversion.

# Examples

```julia
using LinearAlgebra, SciMLOperators

J = MatrixOperator([1.0 2.0; 3.0 4.0])
W = WOperator{true}(I, 0.5, J, zeros(2))

v = [1.0, 2.0]
W * v == (Matrix(W) * v)
```
"""
mutable struct WOperator{
        IIP, T,
        MType,
        GType,
        JType,
        F,
        C,
        JV,
    } <: AbstractWOperator{T}
    mass_matrix::MType
    gamma::GType
    J::JType
    _func_cache::F           # cache used in `mul!`
    _concrete_form::C        # non-lazy form (matrix/number) of the operator
    jacvec::JV
    # Set by `mark_jacobian_updated!` whenever `J`'s contents change, cleared by the
    # consumer via `mark_jacobian_current!`. A solver that caches a factorization of `J`
    # across steps needs to tell a new `gamma` from a new Jacobian; `gamma` it can compare
    # directly, but an in-place write to `J` is otherwise invisible. See `jacobian_stale`.
    jac_stale::Bool

    function WOperator{IIP}(mass_matrix, gamma, J, u, jacvec = nothing) where {IIP}
        if J isa Union{Number, ScalarOperator}
            _concrete_form = -mass_matrix / gamma + convert(Number, J)
            _func_cache = nothing
        else
            AJ = J isa MatrixOperator ? convert(AbstractMatrix, J) : J
            mm = mass_matrix isa MatrixOperator ?
                convert(AbstractMatrix, mass_matrix) : mass_matrix
            _concrete_form = -mm / gamma + AJ
            _func_cache = zero(u)
        end
        T = eltype(_concrete_form)
        MType = typeof(mass_matrix)
        GType = typeof(gamma)
        JType = typeof(J)
        F = typeof(_func_cache)
        C = typeof(_concrete_form)
        JV = typeof(jacvec)
        return new{IIP, T, MType, GType, JType, F, C, JV}(
            mass_matrix, gamma, J,
            # Starts stale: nobody has factorized this Jacobian yet.
            _func_cache, _concrete_form, jacvec, true
        )
    end

    function Base.copy(W::WOperator{IIP, T, MType, GType, JType, F, C, JV}) where {IIP, T, MType, GType, JType, F, C, JV}
        return new{IIP, T, MType, GType, JType, F, C, JV}(
            W.mass_matrix,
            W.gamma,
            W.J,
            copy(W._func_cache),
            copy(W._concrete_form),
            W.jacvec,
            W.jac_stale
        )
    end
end
Base.eltype(W::WOperator) = eltype(W.J)

"""
    jacobian_stale(W) -> Bool

Report whether the Jacobian-dependent state of `W` may have changed since a
consumer last declared its cached factorization or reduction current.

# Arguments

  - `W`: The object whose Jacobian-dependent state is being queried. For a
    `WOperator`, this is its `jac_stale` flag. Other objects use the
    conservative fallback.

# Returns

`true` means that a consumer must assume its Jacobian-dependent cache is
stale. `false` means that the consumer may reuse that cache with respect to
the Jacobian; it must still handle changes to `gamma` according to its own
algorithm. The fallback returns `true`.

# Interface Rules

A solver that owns a factorization or reduction of `W.J` should query this
predicate before reusing it and call [`mark_jacobian_current!`](@ref) only
after refreshing its cache. The flag is shared by consumers, so a single
`WOperator` must not be used as the cache source for independent consumers;
give each consumer its own operator or track its own generation.

# Examples

```julia
using LinearAlgebra, SciMLOperators

W = WOperator{true}(I, 0.5, [1.0 0.0; 0.0 2.0], zeros(2))
jacobian_stale(W) # true until the consumer has initialized its cache
mark_jacobian_current!(W)
!jacobian_stale(W)
```
"""
jacobian_stale(W::WOperator) = W.jac_stale
jacobian_stale(::Any) = true

"""
    mark_jacobian_updated!(W) -> W

Announce that the contents of `W`'s Jacobian have changed, invalidating any
Jacobian-dependent factorization or reduction a solver may be holding. A
no-op for anything that does not track a Jacobian, so it is safe to call
unconditionally after an in-place Jacobian update.

# Arguments

  - `W`: A `WOperator`, or any object for which the unconditional no-op
    fallback is desired.

# Returns

The same object `W`. For a `WOperator`, its `jac_stale` flag is set to `true`.

# Interface Rules

Call this after the contents of `W.J` have been changed in place. Changing
`gamma` is a different event and needs no announcement because it is visible
as a field. This function does not refactorize or otherwise update a cache.

# Examples

```julia
using LinearAlgebra, SciMLOperators

J = [1.0 0.0; 0.0 2.0]
W = WOperator{true}(I, 0.5, J, zeros(2))
mark_jacobian_current!(W)
J .= 3I
mark_jacobian_updated!(W)
jacobian_stale(W) # true
```
"""
mark_jacobian_updated!(A) = A
function mark_jacobian_updated!(W::WOperator)
    W.jac_stale = true
    return W
end

"""
    mark_jacobian_current!(W) -> W

Declare that the caller's Jacobian-dependent cache is up to date, clearing
the flag [`mark_jacobian_updated!`](@ref) set. A no-op off the type.

# Arguments

  - `W`: A `WOperator`, or any object for which the unconditional no-op
    fallback is desired.

# Returns

The same object `W`. For a `WOperator`, its `jac_stale` flag is set to `false`.

# Interface Rules

Call this only after the consumer has actually refreshed its factorization or
reduction. Calling it does not inspect `J` and does not refresh any cache.

# Examples

```julia
using LinearAlgebra, SciMLOperators

W = WOperator{true}(I, 0.5, [1.0 0.0; 0.0 2.0], zeros(2))
mark_jacobian_current!(W)
@assert !jacobian_stale(W)
```

!!! warning "One consumer per operator"
    This is a single shared flag, not a per-consumer generation count. Two solvers caching
    factorizations of the same `W` will race: whichever clears first hides the event from
    the other, which then reuses a stale factorization. Give each consumer its own
    `WOperator` if you need more than one.
"""
mark_jacobian_current!(A) = A
function mark_jacobian_current!(W::WOperator)
    W.jac_stale = false
    return W
end

# In WOperator update_coefficients!, accept both missing u/p/t and missing gamma and don't update them in that case.
# This helps support partial updating logic used with Newton solvers.
# Accept both `gamma` and deprecated `dtgamma` kwargs for backwards compatibility.
function update_coefficients!(
        W::WOperator,
        u = nothing,
        p = nothing,
        t = nothing;
        gamma = nothing,
        dtgamma = nothing
    )
    if dtgamma !== nothing
        Base.depwarn(
            "keyword argument `dtgamma` is deprecated, use `gamma` instead",
            :update_coefficients!
        )
        if gamma === nothing
            gamma = dtgamma
        end
    end
    if (u !== nothing) && (p !== nothing) && (t !== nothing)
        update_coefficients!(W.J, u, p, t)
        update_coefficients!(W.mass_matrix, u, p, t)
        !isnothing(W.jacvec) && update_coefficients!(W.jacvec, u, p, t)
    end
    gamma !== nothing && (W.gamma = gamma)
    return W
end

function Base.convert(::Type{AbstractMatrix}, W::WOperator{IIP}) where {IIP}
    if !IIP || W.J isa AbstractSciMLOperator
        # Mirror the constructor: materialize a MatrixOperator mass matrix so the
        # result type matches `_concrete_form` (otherwise this becomes an
        # AddedOperator and the assignment back into the Matrix-typed slot fails).
        # The IIP case with a plain-matrix `J` is maintained externally via
        # `jacobian2W!` writing into `_concrete_form`; when `J` is an operator no
        # caller maintains it, so it must be rebuilt here with the current gamma.
        mm = W.mass_matrix isa MatrixOperator ?
            convert(AbstractMatrix, W.mass_matrix) : W.mass_matrix
        concrete_form = -mm / W.gamma + convert(AbstractMatrix, W.J)
        # Operator arithmetic can make `_concrete_form` lazy even though its
        # concretization is a matrix, so the rebuilt value may not fit its field type.
        W.J isa AbstractSciMLOperator && return concrete_form
        W._concrete_form = concrete_form
    end
    return W._concrete_form
end
function Base.convert(::Type{Number}, W::WOperator)
    W._concrete_form = -W.mass_matrix / W.gamma + convert(Number, W.J)
    return W._concrete_form
end
# `convert(AbstractMatrix, W)` fuses `-M/γ` with `convert(AbstractMatrix, W.J)`, so `W` is
# convertible exactly when both its mass matrix `M` and its Jacobian `W.J` are. A matrix-free
# `W.J` (e.g. a Jacobian-vector-product operator) — or a matrix-free mass matrix — makes `W`
# matrix-free too; without this the default `all(isconvertible, getops(W)) ==
# all(isconvertible, ()) == true` would wrongly claim a matrix-free `W` is convertible.
isconvertible(W::WOperator) = isconvertible(W.mass_matrix) && isconvertible(W.J)
Base.size(W::WOperator) = size(W.J)
Base.size(W::WOperator, d::Integer) = d <= 2 ? size(W)[d] : 1
function Base.getindex(W::WOperator, i::Int)
    return -W.mass_matrix[i] / W.gamma + W.J[i]
end
function Base.getindex(W::WOperator, I::Vararg{Int, N}) where {N}
    return -W.mass_matrix[I...] / W.gamma + W.J[I...]
end
function Base.:*(W::WOperator, x::AbstractVecOrMat)
    return (W.mass_matrix * x) / -W.gamma + W.J * x
end
function Base.:*(W::WOperator, x::Number)
    return (W.mass_matrix * x) / -W.gamma + W.J * x
end
function Base.:\(W::WOperator, x::AbstractVecOrMat)
    return if size(W) == () # scalar operator
        convert(Number, W) \ x
    else
        convert(AbstractMatrix, W) \ x
    end
end
function Base.:\(W::WOperator, x::Number)
    return if size(W) == () # scalar operator
        convert(Number, W) \ x
    else
        convert(AbstractMatrix, W) \ x
    end
end

function LinearAlgebra.mul!(Y::AbstractVector, W::WOperator, B::AbstractVector)
    # Compute mass_matrix * B
    if isa(W.mass_matrix, UniformScaling)
        a = -W.mass_matrix.λ / W.gamma
        @. Y = a * B
    else
        mul!(Y, W.mass_matrix, B)
        lmul!(-inv(W.gamma), Y)
    end
    # Compute J * B and add
    if isnothing(W.jacvec)
        mul!(W._func_cache, W.J, B)
    else
        mul!(W._func_cache, W.jacvec, B)
    end
    return Y .+= W._func_cache
end

function LinearAlgebra.mul!(Y::AbstractArray, W::WOperator, B::AbstractArray)
    # Compute mass_matrix * B
    if isa(W.mass_matrix, UniformScaling)
        a = -W.mass_matrix.λ / W.gamma
        @. Y = a * B
    else
        mul!(vec(Y), W.mass_matrix, vec(B))
        lmul!(-inv(W.gamma), Y)
    end
    # Compute J * B and add
    if isnothing(W.jacvec)
        mul!(vec(W._func_cache), W.J, vec(B))
    else
        mul!(vec(W._func_cache), W.jacvec, vec(B))
    end
    return vec(Y) .+= vec(W._func_cache)
end

has_concretization(::AbstractWOperator) = true
