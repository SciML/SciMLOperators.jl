#
""" use Base.ReshapedArray """
_reshape(a, dims::NTuple{D,Int}) where{D} = reshape(a,dims)
function _reshape(a::AbstractArray, dims::NTuple{D,Int}) where{D}
    @assert prod(dims) == length(a) "cannot reshape array of size $(size(a)) to size $dims"
    dims == size(a) && return a
    ReshapedArray(a, dims, ())
end

_vec(a) = vec(a)
_vec(a::AbstractVector) = a
_vec(a::AbstractArray) = _reshape(a,(length(a),))
_vec(a::ReshapedArray) = _vec(a.parent)

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
#
