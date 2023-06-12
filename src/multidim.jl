#
###
# operations on multidimensional arrays
###

for op in (
           :*, :\,
          )
    @eval function Base.$op(L::AbstractSciMLOperator, u::AbstractArray)
        u isa AbstractVecOrMat && error("Operation $(Base.$op) not defined for $(typeof(L)), $(typeof(u)).")

        sz_in, sz_out = _mat_sizes(L, u)

        U = reshape(u, sz_in)
        V = $op(L, U)

        reshape(V, sz_out)
    end
end

function LinearAlgebra.mul!(v::AbstractArray, L::AbstractSciMLOperator, u::AbstractArray, args...)
    u isa AbstractVecOrMat && @error "LinearAlgebra.mul! not defined for $(typeof(L)), $(typeof(u))."

    sz_in, sz_out = _mat_sizes(L, u)

    U = reshape(u, sz_in)
    V = reshape(v, sz_out)

    mul!(V, L, U, args...)

    v
end

function LinearAlgebra.ldiv!(v::AbstractArray, L::AbstractSciMLOperator, u::AbstractArray)
    u isa AbstractVecOrMat && @error "LinearAlgebra.ldiv! not defined for $(typeof(L)), $(typeof(u))."

    sz_in, sz_out = _mat_sizes(L, u)

    U = reshape(u, sz_in)
    V = reshape(v, sz_out)

    ldiv!(V, L, U)

    v
end

function LinearAlgebra.ldiv!(L::AbstractSciMLOperator, u::AbstractArray)
    u isa AbstractVecOrMat && @error "LinearAlgebra.ldiv! not defined for $(typeof(L)), $(typeof(u))."

    sz_in, _ = _mat_sizes(L, u)

    U = reshape(u, sz_in)

    ldiv!(L, U)

    u
end
#
