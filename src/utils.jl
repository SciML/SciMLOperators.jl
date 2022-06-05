#
""" use Base.ReshapedArray """
_reshape(a,dims::NTuple{D,Int}) where{D} = reshape(a,dims)
_reshape(a::Array, dims::NTuple{D,Int}) where{D} = ReshapedArray(a, dims, ())

_vec(a) = vec(a)
_vec(a::AbstractVector) = a
_vec(a::AbstractArray) = _reshape(a,(length(a),))
_vec(a::ReshapedArray) = _vec(a.parent)
#
