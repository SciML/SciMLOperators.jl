#

###
# operator update interface
###

"""
$SIGNATURES

The default update function for `AbstractSciMLOperator`s, a no-op that
leaves the operator state unchanged.

# Arguments

  - `A`: Current operator state.
  - `u`: State or update vector supplied by the caller.
  - `p`: Parameter object supplied by the caller.
  - `t`: Time or other scalar update value.

# Returns

The unchanged value `A`. The positional update arguments are accepted so this
function can be used wherever an operator update function is expected.
"""
DEFAULT_UPDATE_FUNC(A, u, p, t) = A

const UPDATE_COEFFS_WARNING = """
!!! warning
    The user-provided `update_func[!]` must not use `u` in
    its computation. Positional argument `(u, p, t)` to `update_func[!]` are
    passed down by `update_coefficients[!](L, u, p, t)`, where `u` is the
    input-vector to the composite `AbstractSciMLOperator`. For that reason,
    the values of `u`, or even shape, may not correspond to the input
    expected by `update_func[!]`. If an operator's state depends on its
    input vector, then it is, by definition, a nonlinear operator.
    We recommend sticking such nonlinearities in `FunctionOperator.`
    This topic is further discussed in
    [this issue](https://github.com/SciML/SciMLOperators.jl/issues/159).
"""

"""
$SIGNATURES

Update the state of `L` based on `u`, input vector, `p` parameter object,
`t`, and keyword arguments. Internally, `update_coefficients` calls the
user-provided `update_func` method for every component operator in `L`
with the positional arguments `(u, p, t)` and keyword arguments
corresponding to the symbols provided to the operator via kwarg
`accepted_kwargs`.

This method is out-of-place, i.e. fully non-mutating and `Zygote`-compatible.

$(UPDATE_COEFFS_WARNING)

# Arguments

  - `L`: Operator whose state should be updated.
  - `u`: State or update vector supplied to the operator.
  - `p`: Parameter object supplied to the operator.
  - `t`: Time or other scalar update value.

# Keyword Arguments

  - `kwargs...`: Keywords accepted by the operator's update function and
    recorded by its `accepted_kwargs` constructor option.

# Returns

A new operator representing the updated state. `L` is not mutated.

# Examples

```
using SciMLOperators

mat_update_func = (A, u, p, t; scale = 1.0) -> p * p' * scale * t

M = MatrixOperator(zeros(4,4); update_func = mat_update_func,
                   accepted_kwargs = Val((:scale,)))

L = M + IdentityOperator(4)

u = rand(4)
p = rand(4)
t = 1.0
v = rand(4)

# Update the operator to `(u,p,t)` and apply it to `v`
L = update_coefficients(L, u, p, t; scale = 2.0)
result = L * v

# Or use the interface which separates the update from the application
result = L(v, u, p, t; scale = 2.0)
```
"""
update_coefficients(L, u, p, t; kwargs...) = L

"""
$SIGNATURES

Update in-place the state of `L` based on `u`, input vector, `p` parameter
object, `t`, and keyword arguments. Internally, `update_coefficients!` calls
the user-provided mutating `update_func!` method for every component operator
in `L` with the positional arguments `(u, p, t)` and keyword arguments
corresponding to the symbols provided to the operator via kwarg
`accepted_kwargs`.

$(UPDATE_COEFFS_WARNING)

# Arguments

  - `L`: Operator whose state should be updated.
  - `u`: State or update vector supplied to the operator.
  - `p`: Parameter object supplied to the operator.
  - `t`: Time or other scalar update value.

# Keyword Arguments

  - `kwargs...`: Keywords accepted by the operator's mutating update function
    and recorded by its `accepted_kwargs` constructor option.

# Returns

`nothing`. The operator `L` is mutated in place.

# Examples

```
using SciMLOperators

_A = rand(4, 4)
mat_update_func! = (A, u, p, t; scale = 1.0) -> copy!(A, _A)

M = MatrixOperator(zeros(4,4); update_func! = mat_update_func!)

L = M + IdentityOperator(4)

u = rand(4)
p = rand(4)
t = 1.0
v = rand(4)

update_coefficients!(L, u, p, t)
L * v
```
"""
update_coefficients!(L, u, p, t; kwargs...) = nothing

# We cannot use @generate because we don't know the type structure of L in advance
function update_coefficients!(L::AbstractSciMLOperator, u, p, t; kwargs...)
    foreach(op -> update_coefficients!(op, u, p, t; kwargs...), getops(L))

    return nothing
end

###
# operator evaluation interface
###

# Out-of-place: v is action vector, u is update vector
function (L::AbstractSciMLOperator)(v, u, p, t; kwargs...)
    return update_coefficients(L, u, p, t; kwargs...) * v
end
# In-place: w is destination, v is action vector, u is update vector
function (L::AbstractSciMLOperator)(w, v, u, p, t; kwargs...)
    (update_coefficients!(L, u, p, t; kwargs...); mul!(w, L, v))
end
# In-place with scaling: w = α*(L*v) + β*w
function (L::AbstractSciMLOperator)(w, v, u, p, t, α, β; kwargs...)
    (update_coefficients!(L, u, p, t; kwargs...); mul!(w, L, v, α, β))
end

function (L::AbstractSciMLOperator)(w::Number, v::Number, u, p, t, args...; kwargs...)
    msg = """Nonallocating L(w, v, u, p, t) type methods are not available for
    subtypes of `Number`."""
    throw(ArgumentError(msg))
end

###
# operator caching interface
###

getops(L) = ()

"""
$SIGNATURES

Checks whether `L` has preallocated caches for inplace evaluations.

# Arguments

  - `L`: Operator to inspect. For a composite, all child caches are checked.

# Returns

`true` when the operator and every child needed for in-place evaluation has a
usable cache, otherwise `false`.
"""
function iscached(L::AbstractSciMLOperator)
    has_cache = hasfield(typeof(L), :cache) # TODO - confirm this is static
    isset = has_cache ? !isnothing(L.cache) : true

    return isset & all(iscached, getops(L))
end

"""
Check if `SciMLOperator` `L` has preallocated cache-arrays for in-place
computation.
"""
iscached(L) = true

iscached(
    ::Union{# LinearAlgebra
        AbstractMatrix,
        UniformScaling,
        Factorization, # Base
        Number,
    }
) = true

"""
$SIGNATURES

Allocate caches for `L` for in-place evaluation with `u`-like input vectors.

# Arguments

  - `L`: Operator to prepare.
  - `u`: Prototype vector or matrix whose shape determines compatible scratch.

# Returns

`L` or a cached replacement. The returned operator must have the same action,
size, and trait values as the input operator.

# Interface Rules

Call this before repeated `mul!` or callable in-place evaluations when the
operator needs scratch storage. A custom type that needs storage should
implement `cache_self` for its own buffers and `cache_internals` for child
operators.
"""
cache_operator(L, u) = L

function cache_operator(L::AbstractSciMLOperator, u::AbstractVecOrMat)
    L = cache_self(L, u)
    L = cache_internals(L, u)
    return L
end

cache_self(L::AbstractSciMLOperator, ::AbstractVecOrMat) = L
cache_internals(L::AbstractSciMLOperator, ::AbstractVecOrMat) = L

"""
$SIGNATURES

Return the cache held by `op`, or `nothing` when it holds none.

Defining this method opts `op`'s type into the scratch sharing `AddedOperator` does
between its summands, which is sound because `mul!` applies them strictly serially and no
two of their caches are ever live at once. A summand may then be given the cache of an
earlier one whose cache has the identical type and identically sized slots, so only define
this for operators where two such caches really are interchangeable — including which
slots deliberately alias each other. The `nothing` default keeps a type out of it entirely.

Two further conditions come with defining it. The first is on the operator's own
`cache_self`: it must treat the buffers it is handed as a template to `similar`, `zero` or
`reshape` and never retain a reference to them, since the whole point is that those buffers
may afterwards be replaced by an earlier summand's. Every `cache_self` in this package does.
The second is on the caller: applying the summands of an `AddedOperator` concurrently, by
hand, is no longer safe once they share scratch — a cached operator was never safe to apply
concurrently, but now its summands are not independently safe either.

A wrapper that holds no scratch of its own — `ScaledOperator`, `AdjointOperator`,
`TransposedOperator` — is transparent to all of this and forwards `getcache`,
[`update_cache`](@ref) and [`adopt_cache`](@ref) to the operator it wraps. A new wrapper of
that kind should do the same; without it the operator underneath drops out of sharing even
though its cache would have qualified.

See also [`adopt_cache`](@ref), which additionally spares the operator from allocating a
cache it would only discard.
"""
getcache(::AbstractSciMLOperator) = nothing

"""
$SIGNATURES

Replace `op`'s cache with `new_cache`. Only ever called with a `new_cache` of exactly
the same type as the one it replaces, so `op`'s type is unchanged.

# Arguments

  - `op`: Cached operator whose scratch should be replaced.
  - `new_cache`: Cache with the same type, shape, and aliasing layout as the
    existing cache.

# Returns

An operator with `new_cache` installed. Implementations must preserve the
operator action and type parameters.
"""
update_cache(op::AbstractSciMLOperator, new_cache) = @reset op.cache = new_cache

# Identical cache types already pin element type, layout and device, so only the extents
# are left to check. Anything else is conservatively treated as not interchangeable.
_slots_match(a::AbstractArray, b::AbstractArray) = size(a) == size(b)
_slots_match(::Nothing, ::Nothing) = true
function _slots_match(a::NTuple{N, Any}, b::NTuple{N, Any}) where {N}
    return all(map(_slots_match, a, b)) && _aliasing_matches(a, b)
end
_slots_match(a, b) = false

# Which slots deliberately point at the same buffer is part of the layout, not an accident
# of it: `TensorProductOperator` folds two of its slots into one exactly when both factors
# are square. Comparing the pattern keeps an operator that needs distinct buffers from ever
# being handed a cache that shares one, instead of relying on the sizes to give it away.
function _aliasing_matches(a::NTuple{N, Any}, b::NTuple{N, Any}) where {N}
    return all(
        ntuple(Val(N)) do i
            all(ntuple(j -> (a[i] === a[j]) === (b[i] === b[j]), Val(N)))
        end
    )
end

# Having the same cache type is settled at compile time; only the sizes cost anything at
# run time. Caches of differing type take the second method and are never interchanged.
function _swap_cache(op, mine::C, theirs::C) where {C}
    return _slots_match(mine, theirs) ? update_cache(op, theirs) : nothing
end
_swap_cache(op, mine, theirs) = nothing

# Point an *already cached* `op` at an interchangeable cache held by one of the `earlier`
# summands, so its own buffers can be collected. Leaves `op` alone when nothing matches.
#
# This does not make `op` work — `cache_operator` already did that — it makes it smaller.
# Nothing here describes what a cache ought to look like: the buffers being compared have
# already been allocated, so there is no second description of the layout that could drift
# out of sync with `cache_self`.
_deduplicate_cache(op, ::Tuple{}) = op

function _deduplicate_cache(op, earlier::Tuple)
    mine = getcache(op)
    mine === nothing && return op

    swapped = _swap_cache(op, mine, getcache(first(earlier)))
    swapped === nothing || return swapped

    return _deduplicate_cache(op, Base.tail(earlier))
end

"""
$SIGNATURES

Return `op` set up for in-place use with `v` using `cache` — the cache of an operator
already established as interchangeable with it — instead of allocating buffers of its own.
Return `nothing` to decline, in which case `op` is cached the ordinary way and then offered
the same cache after the fact, which costs an allocation that is immediately discarded.
What comes back must report `cache` as its own — [`getcache`](@ref) on it is checked against
what was handed over, and anything else is treated as having declined, since an
implementation that quietly allocated instead would otherwise skip the ordinary path too.

Declining is the default, because there is no implementation that is correct for every
operator: one that follows the `cache_self`/`cache_internals` split opts in with

    adopt_cache(op::MyOperator, cache, v) = cache_internals(update_cache(op, cache), v)

whereas one that defines its own `cache_operator` must instead do whatever that does, minus
the allocation — the line above would silently skip its internal caching entirely.

Defining this asks more of an operator than [`getcache`](@ref) alone. `getcache` only
claims that two caches of the same type with the same slot sizes are interchangeable, and
that is checked against buffers that already exist. This additionally claims that what
`cache_self` builds is a function of the operator's type, its operands' sizes and `v`, since
that is what the caller compares before any buffer exists. Leave it undefined when the cache
depends on anything else — `FunctionOperator` does, because it sizes buffers from
`traits.sizes`, which `size` does not expose.

# Arguments

  - `op`: Operator that may adopt the supplied cache.
  - `cache`: An interchangeable cache from another operator.
  - `v`: Prototype action vector or matrix used to build child caches.

# Returns

An operator using `cache`, or `nothing` to decline. Returning an operator that
silently allocates a different cache is a contract violation.
"""
adopt_cache(::AbstractSciMLOperator, cache, v) = nothing

# Everything `cache_self` may read: the operator's own size and, recursively, its operands'.
# `adopt_cache`'s contract is that a cache layout follows from the operator's type together
# with these, so equal types and equal signatures mean interchangeable caches.
_shape_signature(x) = ()
_shape_signature(x::AbstractArray) = size(x)
_shape_signature(L::AbstractSciMLOperator) = (size(L), map(_shape_signature, getops(L)))

# Cache `op`, taking the buffers of one of the `donors` outright when they are
# interchangeable, so that nothing is allocated only to be thrown away.
#
# Every donor already shares `op`'s concrete type — the caller settles that at compile time,
# which pins every element type — so only their shape signatures are compared here. The two
# checks are not redundant: `M(8, 8) * M(8, 8)` and `M(8, 16) * M(16, 8)` have identical
# types and identical overall size while needing different cache layouts.
#
# Falls back to caching `op` normally and deduplicating afterwards. That happens when no
# donor fits, and for any type that has not opted into `adopt_cache`.
function _cache_summand(op, donors::Tuple, earlier::Tuple, v)
    adopted = _try_adopt(op, donors, v)
    adopted === nothing || return adopted

    return _deduplicate_cache(cache_operator(op, v), earlier)
end

# Walk the same-typed candidates in order, taking the first whose sizes also line up.
_try_adopt(op, ::Tuple{}, v) = nothing

function _try_adopt(op, donors::Tuple, v)
    donor = first(donors)
    cache = getcache(donor)

    if cache !== nothing && _shape_signature(op) == _shape_signature(donor)
        adopted = adopt_cache(op, cache, v)
        # An `adopt_cache` that comes back reporting something other than the cache it was
        # handed did not use it — it allocated buffers of its own, and taking it at its word
        # would skip the post-hoc path as well and leave the summand sharing nothing. Take
        # the ordinary route instead, where `_deduplicate_cache` still gets its turn.
        if adopted !== nothing && getcache(adopted) === cache
            return adopted
        end
    end

    return _try_adopt(op, Base.tail(donors), v)
end

###
# operator traits
###

Base.size(A::AbstractSciMLOperator, d::Integer) = d <= 2 ? size(A)[d] : 1
Base.eltype(::Type{<:AbstractSciMLOperator{T}}) where {T} = T
Base.eltype(::AbstractSciMLOperator{T}) where {T} = T

_promote_operator_eltype(A, B) = promote_type(eltype(A), eltype(B))

Base.oneunit(L::AbstractSciMLOperator) = one(L)
Base.oneunit(LType::Type{<:AbstractSciMLOperator}) = one(LType)

Base.iszero(::AbstractSciMLOperator) = false # TODO

"""
$SIGNATURES

Check if `adjoint(L)` is lazily defined.

# Arguments

  - `L`: Operator to inspect.

# Returns

`true` only when `adjoint(L)` is supported without relying on an
unadvertised conversion fallback.
"""
has_adjoint(L::AbstractSciMLOperator) = false # L', adjoint(L)
"""
$SIGNATURES

Check if `expmv!(w, L, v, t)`, equivalent to `mul!(w, exp(t * A), v)`, is
defined for `Number` `t`, and `AbstractArray`s `w, v` of appropriate sizes.

# Arguments

  - `L`: Operator to inspect.

# Returns

`true` only when the in-place exponential action is part of the operator's
supported contract.
"""
has_expmv!(L::AbstractSciMLOperator) = false # expmv!(v, L, t, u)
"""
$SIGNATURES

Check if `expmv(L, v, t)`, equivalent to `exp(t * A) * v`, is defined for
`Number` `t`, and `AbstractArray` `u` of appropriate size.

# Arguments

  - `L`: Operator to inspect.

# Returns

`true` only when the out-of-place exponential action is supported.
"""
has_expmv(L::AbstractSciMLOperator) = false # v = exp(L, t, u)
"""
$SIGNATURES

Check if `exp(L)` is defined lazily.

# Arguments

  - `L`: Operator to inspect.

# Returns

`true` only when `exp(L)` is a supported lazy operation.
"""
has_exp(L::AbstractSciMLOperator) = islinear(L)
"""
$SIGNATURES

Check if `L * v` is defined for `AbstractArray` `u` of appropriate size.

# Arguments

  - `L`: Operator to inspect.

# Returns

`true` only when `L * v` is supported for compatible action arrays.
"""
has_mul(L::AbstractSciMLOperator) = true # du = L*u
"""
$SIGNATURES

Check if `mul!(w, L, v)` is defined for `AbstractArray`s `w, v` of
appropriate sizes.

# Arguments

  - `L`: Operator to inspect.

# Returns

`true` only when the in-place multiplication and its return-value contract
are supported for compatible arrays.
"""
has_mul!(L::AbstractSciMLOperator) = true # mul!(du, L, u)
"""
$SIGNATURES

Check if `L \\ v` is defined for `AbstractArray` `v` of appropriate size.

# Arguments

  - `L`: Operator to inspect.

# Returns

`true` only when the out-of-place solve is supported for compatible right-hand
sides.
"""
has_ldiv(L::AbstractSciMLOperator) = false # du = L\u
"""
$SIGNATURES

Check if `ldiv!(w, L, v)` is defined for `AbstractArray`s `w, v` of
appropriate sizes.

# Arguments

  - `L`: Operator to inspect.

# Returns

`true` only when the in-place solve is supported for compatible arrays.
"""
has_ldiv!(L::AbstractSciMLOperator) = false # ldiv!(du, L, u)

### Extra standard assumptions

"""
$SIGNATURES

Checks if an `L`'s state is constant or needs to be updated by calling
`update_coefficients`.

# Arguments

  - `L`: Operator or operator-like object to inspect.

# Returns

`true` when updating with supported `(u, p, t; kwargs...)` values cannot change
the operator action. A stateful leaf must override this trait explicitly.
"""
isconstant(
    ::Union{# LinearAlgebra
        AbstractMatrix,
        UniformScaling,
        Factorization, # Base
        Number,
    }
) = true
isconstant(L::AbstractSciMLOperator) = all(isconstant, getops(L))
isconstant(L) = false

"""
    isconvertible(L) -> Bool

Checks if `L` can be cheaply converted to an `AbstractMatrix` via eager fusion.

# Arguments

  - `L`: Operator to inspect.

# Returns

`true` only when `convert(AbstractMatrix, L)` is a supported operation for the
current operator state. This trait does not require eager conversion to be the
preferred execution path.
"""
isconvertible(L::AbstractSciMLOperator) = all(isconvertible, getops(L))

function isconvertible(
        ::Union{
            # LinearAlgebra
            AbstractMatrix,
            UniformScaling,
            Factorization,

            # Base
            Number,

            # SciMLOperators
            AbstractSciMLScalarOperator,
        }
    )
    return true
end

"""
    concretize(L) -> AbstractMatrix

    concretize(L) -> Number

Convert `SciMLOperator` to a concrete type via eager fusion. This method is a
no-op for types that are already concrete.

# Arguments

  - `L`: Matrix-like or scalar operator to materialize.

# Returns

An `AbstractMatrix` for matrix-like operators or a `Number` for scalar
operators, with the same current action as `L`.

# Errors

Throws the conversion error from `L` when the operator does not support the
requested concrete representation. Check `isconvertible(L)` before relying
on eager matrix materialization.
"""
function concretize(
        L::Union{# LinearAlgebra
            AbstractMatrix,
            Factorization, # SciMLOperators
            AbstractSciMLOperator,
        }
    )
    return convert(AbstractMatrix, L)
end

function concretize(
        L::Union{
            # LinearAlgebra
            UniformScaling,

            # Base
            Number,

            # SciMLOperators
            AbstractSciMLScalarOperator,
        }
    )
    return convert(Number, L)
end

"""
$SIGNATURES

Checks if `L` is a linear operator.

# Arguments

  - `L`: Operator to inspect.

# Returns

`true` when the action is linear in the action vector, even if the operator
state depends on `(u, p, t)`.
"""
islinear(::AbstractSciMLOperator) = false
islinear(L) = false

islinear(
    ::Union{# LinearAlgebra
        AbstractMatrix,
        UniformScaling,
        Factorization, # Base
        Number,
    }
) = true

has_mul(L) = false
has_mul(
    ::Union{# LinearAlgebra
        AbstractVecOrMat,
        AbstractMatrix,
        UniformScaling, # Base
        Number,
    }
) = true

has_mul!(L) = false
has_mul!(
    ::Union{# LinearAlgebra
        AbstractVecOrMat,
        AbstractMatrix,
        UniformScaling, # Base
        Number,
    }
) = true

has_ldiv(L) = false
has_ldiv(
    ::Union{
        AbstractMatrix,
        Factorization,
        Number,
    }
) = true

has_ldiv!(L) = false
has_ldiv!(
    ::Union{
        Diagonal,
        Bidiagonal,
        Factorization,
    }
) = true

has_adjoint(L) = islinear(L)
has_adjoint(
    ::Union{# LinearAlgebra
        AbstractMatrix,
        UniformScaling,
        Factorization, # Base
        Number,
    }
) = true

"""
Checks if `size(L, 1) == size(L, 2)`.

# Arguments

  - `L`: Matrix-like object or operator to inspect.

# Returns

`true` for a square operator and `false` for a rectangular operator or vector.
For multiple arguments, the result is the elementwise conjunction of their
individual square predicates.
"""
issquare(L) = ndims(L) >= 2 && size(L, 1) == size(L, 2)
issquare(::AbstractVector) = false
issquare(
    ::Union{# LinearAlgebra
        UniformScaling, # SciMLOperators
        AbstractSciMLScalarOperator, # Base
        Number,
    }
) = true
issquare(A...) = @. (&)(issquare(A)...)

Base.length(L::AbstractSciMLOperator) = prod(size(L))
Base.ndims(L::AbstractSciMLOperator) = length(size(L))
Base.isreal(L::AbstractSciMLOperator{T}) where {T} = T <: Real
Base.Matrix(L::AbstractSciMLOperator) = Matrix(convert(AbstractMatrix, L))

Base.exp(L::AbstractSciMLOperator, t) = exp(t * L)
expmv(L::AbstractSciMLOperator, u, p, t) = exp(L, t) * u
expmv!(v, L::AbstractSciMLOperator, u, p, t) = mul!(v, exp(L, t), u)

###
# fallback implementations
###

function Base.conj(L::AbstractSciMLOperator)
    isreal(L) && return L
    if !isconvertible(L)
        @warn """using convert-based fallback for Base.conj"""
    end
    return concretize(L) |> conj
end

function Base.:(==)(L1::AbstractSciMLOperator, L2::AbstractSciMLOperator)
    if !isconvertible(L1) || !isconvertible(L2)
        @warn """using convert-based fallback for Base.=="""
    end
    size(L1) != size(L2) && return false
    return concretize(L1) == concretize(L2)
end

function Base.getindex(L::AbstractSciMLOperator, I::Vararg{Int, N}) where {N}
    if !isconvertible(L)
        @warn """using convert-based fallback for Base.getindex"""
    end
    return concretize(L)[I...]
end

function Base.resize!(L::AbstractSciMLOperator, n::Integer)
    throw(MethodError(resize!, typeof.((L, n))))
end

Base.exp(L::AbstractSciMLOperator) = exp(Matrix(L))

function LinearAlgebra.opnorm(L::AbstractSciMLOperator, p::Real = 2)
    if !isconvertible(L)
        @warn """using convert-based fallback in LinearAlgebra.opnorm."""
    end
    return opnorm(concretize(L), p)
end

for op in (:sum, :prod)
    @eval function Base.$op(L::AbstractSciMLOperator; kwargs...)
        if !isconvertible(L)
            @warn """using convert-based fallback in $($op)."""
        end
        return $op(concretize(L); kwargs...)
    end
end

for pred in (
        :issymmetric,
        :ishermitian,
        :isposdef,
    )
    @eval function LinearAlgebra.$pred(L::AbstractSciMLOperator)
        has_concretization(L) || return false
        if !isconvertible(L)
            @warn """using convert-based fallback in $($pred)."""
        end
        return $pred(concretize(L))
    end
end

function LinearAlgebra.mul!(v::AbstractArray, L::AbstractSciMLOperator, u::AbstractArray)
    if !isconvertible(L)
        @warn """using convert-based fallback in mul!."""
    end
    return mul!(v, concretize(L), u)
end

function LinearAlgebra.mul!(
        v::AbstractArray,
        L::AbstractSciMLOperator,
        u::AbstractArray,
        α,
        β
    )
    if !isconvertible(L)
        @warn """using convert-based fallback in mul!."""
    end
    return mul!(v, concretize(L), u, α, β)
end

#
