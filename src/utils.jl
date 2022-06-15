#
""" use Base.ReshapedArray """
_reshape(a, dims::NTuple{D,Int}) where{D} = reshape(a,dims)
function _reshape(a::AbstractArray, dims::NTuple{D,Int}) where{D}
    @assert prod(dims) == length(a) "cannot reshape $a to size $dims"
    dims == size(a) && return a
    ReshapedArray(a, dims, ())
end

_vec(a) = vec(a)
_vec(a::AbstractVector) = a
_vec(a::AbstractArray) = _reshape(a,(length(a),))
_vec(a::ReshapedArray) = _vec(a.parent)

function _view(a, dims::NTuple{D,Int}) where{D}
    # just one Colon -> _vec
    all(dim -> isa(dim, Colon), dims) && return a
    dims == size(a) && return a
    length(a) == prod(dims) && return a

    view(a, dims...)
end

function _mat_sizes(L::AbstractSciMLOperator, u::AbstractArray)

    size_in = u isa AbstractVecOrMat ? size(u) : begin
        m, n = size(L)
        nk = length(u)

        (n, nk ÷ n)
    end

    size_out = issquare(L) ? size_in : (m, size(u)[2:end]...)

    size_in, size_out
end
#
