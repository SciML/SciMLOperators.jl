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

# Filter keyword arguments to those accepted by function.
# Avoid throwing errors here if a keyword argument is not provided: defer
# this to the function call for a more readable error.
function get_filtered_kwargs(kwargs::Base.Pairs, accepted_kwargs::NTuple{N,Symbol}) where{N}
    (kw => kwargs[kw] for kw in accepted_kwargs if haskey(kwargs, kw))
end

function (f::FilterKwargs)(args...; kwargs...)
    filtered_kwargs = get_filtered_kwargs(kwargs, f.accepted_kwargs)
    f.f(args...; filtered_kwargs...)
end
#
