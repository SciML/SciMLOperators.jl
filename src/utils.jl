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
    _accepted_kwargs = (accepted_kwargs === nothing) ? () : accepted_kwargs
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
function get_filtered_kwargs(kwargs::AbstractDict,
        accepted_kwargs::NTuple{N, Symbol}) where {N}
    (kw => kwargs[kw] for kw in accepted_kwargs if haskey(kwargs, kw))
end
function get_filtered_kwargs(kwargs::Union{AbstractDict, NamedTuple},
        ::Val{accepted_kwargs}) where {accepted_kwargs}
    kwargs_nt = NamedTuple(kwargs)
    return NamedTuple{accepted_kwargs}(kwargs_nt)  # This creates a new NamedTuple with keys specified by `accepted_kwargs`
end

function (f::FilterKwargs)(args...; kwargs...)
    filtered_kwargs = get_filtered_kwargs(kwargs, f.accepted_kwargs)
    f.f(args...; filtered_kwargs...)
end

isnothingfunc(f::FilterKwargs) = isnothingfunc(f.f)
isnothingfunc(f::Nothing) = true
isnothingfunc(f) = false
#

_unwrap_val(x) = x
_unwrap_val(::Val{X}) where {X} = X
