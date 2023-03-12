#
function _mat_sizes(L::AbstractSciMLOperator, u::AbstractArray)
    m, n = size(L)
    nk = length(u)

    size_in  = u isa AbstractVecOrMat ? size(u) : (n, nk ÷ n)
    size_out = issquare(L) ? size_in : (m, size(u)[2:end]...)

    size_in, size_out
end

dims(A) = length(size(A))
dims(::AbstractArray{<:Any,N}) where{N} = N
dims(::AbstractSciMLOperator) = 2

# Keyword argument filtering
struct FilterKwargs{F,K}
    f::F
    accepted_kwargs::K
end
function (f_filter::FilterKwargs)(args...; kwargs...)
    # Filter keyword arguments to those accepted by function.
    # Avoid throwing errors here if a keyword argument is not provided: defer this to the function call for a more readable error.
    filtered_kwargs = (kwarg => kwargs[kwarg] for kwarg in f_filter.accepted_kwargs if haskey(kwargs, kwarg))
    f_filter.f(args...; filtered_kwargs...)
end
# automatically convert NamedTuple's, etc. to a normalized kwargs representation (i.e. Base.Pairs) 
normalize_kwargs(; kwargs...) = kwargs
normalize_kwargs(kwargs) = normalize_kwargs(; kwargs...)
#
