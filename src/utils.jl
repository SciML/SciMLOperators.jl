#
""" use Base.ReshapedArray """
_reshape(a, dims::NTuple{D,Int}) where{D} = reshape(a,dims)
function _reshape(a::AbstractArray, dims::NTuple{D,Int}) where{D}
    ReshapedArray(a, dims, ())
end

_vec(a) = vec(a)
_vec(a::AbstractVector) = a
_vec(a::AbstractArray) = _reshape(a,(length(a),))
_vec(a::ReshapedArray) = _vec(a.parent)
#
