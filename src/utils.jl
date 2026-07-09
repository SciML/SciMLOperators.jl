#

###
# kwarg filter
###

"""
$SIGNATURES

This type indicates to `preprocess_update_func` to not to filter keyword
arguments. Required in implementation of lazy `Base.adjoint`,
`Base.conj`, `Base.transpose`.
"""
struct NoKwargFilter end

function preprocess_update_func(update_func, accepted_kwargs)
    # Convert accepted_kwargs to Val for compile-time kwarg filtering to avoid allocations
    _accepted_kwargs = if accepted_kwargs === nothing
        Val(())
    elseif accepted_kwargs isa Tuple
        # Deprecation: Encourage users to use Val((...)) directly for better performance
        @warn """Passing accepted_kwargs as a plain Tuple is deprecated and will be removed in a future version.
        Please use Val((...)) instead for zero-allocation kwarg filtering.
        Example: accepted_kwargs = Val((:dtgamma,)) instead of accepted_kwargs = (:dtgamma,)
        This message will only be shown once per session.""" maxlog = 1
        Val(accepted_kwargs)
    else
        accepted_kwargs  # Already a Val or NoKwargFilter
    end
    # accepted_kwargs can be passed as nothing to indicate that we should not filter
    # (e.g. if the function already accepts all kwargs...).
    return (_accepted_kwargs isa NoKwargFilter) ? update_func :
        FilterKwargs(update_func, _accepted_kwargs)
end

update_func_isconstant(::Nothing) = true
function update_func_isconstant(update_func)
    if update_func isa FilterKwargs
        return update_func.f === DEFAULT_UPDATE_FUNC || update_func.f === nothing
    else
        return update_func === DEFAULT_UPDATE_FUNC
    end
end

# Keyword argument filtering
struct FilterKwargs{F, K}
    f::F
    accepted_kwargs::K
end

# Filter keyword arguments to those accepted by function.
# Avoid throwing errors here if a keyword argument is not provided: defer
# this to the function call for a more readable error.
function get_filtered_kwargs(
        kwargs::AbstractDict,
        accepted_kwargs::NTuple{N, Symbol}
    ) where {N}
    return (kw => kwargs[kw] for kw in accepted_kwargs if haskey(kwargs, kw))
end
function get_filtered_kwargs(
        kwargs::Union{AbstractDict, NamedTuple},
        ::Val{accepted_kwargs}
    ) where {accepted_kwargs}
    kwargs_nt = NamedTuple(kwargs)
    # Only extract keys that exist in kwargs_nt to avoid errors
    filtered_keys = filter(k -> haskey(kwargs_nt, k), accepted_kwargs)
    return NamedTuple{filtered_keys}(kwargs_nt)
end

function (f::FilterKwargs)(args...; kwargs...)
    filtered_kwargs = get_filtered_kwargs(kwargs, f.accepted_kwargs)
    return f.f(args...; filtered_kwargs...)
end

isnothingfunc(f::FilterKwargs) = isnothingfunc(f.f)
isnothingfunc(f::Nothing) = true
isnothingfunc(f) = false
#

_unwrap_val(x) = x
_unwrap_val(::Val{X}) where {X} = X

"""
$SIGNATURES

Return whether `L` can be materialized into a concrete scalar or matrix
representation for fallback operations.

# Arguments

  - `L`: An operator-like object.

# Returns

`true` when `concretize(L)` is expected to succeed by calling
`convert(AbstractMatrix, L)` or `convert(Number, L)`, and `false` otherwise.

# Interface Rules

Subtypes of `AbstractSciMLOperator` should define `has_concretization(L)` as
`true` only when their current state can be materialized without changing the
operator action. Composite operators should return `true` only when every
component needed for the materialization also has concretization.

This trait is intentionally separate from `isconvertible(L)`: an operator can
have a correct concrete representation but avoid cheap eager fusion in generic
algebra paths.

# Examples

```julia
using SciMLOperators

A = MatrixOperator([1.0 2.0; 3.0 4.0])
has_concretization(A) # true

F = FunctionOperator((y, x, u, p, t) -> copyto!(y, x), zeros(2), zeros(2);
    isinplace = true, T = Float64, islinear = true)
has_concretization(F) # false
```
"""
has_concretization(::AbstractSciMLOperator) = false
has_concretization(::Union{AbstractMatrix, UniformScaling, Factorization, Number}) = true
