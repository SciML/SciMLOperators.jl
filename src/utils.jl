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
#
