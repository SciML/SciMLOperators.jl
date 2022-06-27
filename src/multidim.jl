#
###
# operations on multidimensional arrays
###

for op in (
           :*, :\,
          )
    @eval function Base.$op(L::AbstractSciMLOperator, u::AbstractArray)
        u isa AbstractVecOrMat && @error "Base.$op not defined for $(typeof(L)), $(typeof(u))."

        sizes = _mat_sizes(L, u)
        sizev = issquare(L) ? size(u) : begin
            (size(L, 1), size(u)[2:end]...,)
        end

        uu = _reshape(u, sizes[1])
        vv = $op(L, uu)

        _reshape(vv, sizev)
    end
end

function LinearAlgebra.mul!(v::AbstractArray, L::AbstractSciMLLinearOperator, u::AbstractArray)
    u isa AbstractVecOrMat && @error "LinearAlgebra.mul! not defined for $(typeof(L)), $(typeof(u))."

    sizes = _mat_sizes(L, u)

    uu = _reshape(u, sizes[1])
    vv = _reshape(v, sizes[2])

    mul!(vv, L, uu)

    v
end

function LinearAlgebra.mul!(v::AbstractArray, L::AbstractSciMLLinearOperator, u::AbstractArray, α, β)
    u isa AbstractVecOrMat && @error "LinearAlgebra.mul! not defined for $(typeof(L)), $(typeof(u))."

    sizes = _mat_sizes(L, u)

    uu = _reshape(u, sizes[1])
    vv = _reshape(v, sizes[2])

    mul!(vv, L, uu, α, β)

    v
end

function LinearAlgebra.ldiv!(v::AbstractArray, L::AbstractSciMLLinearOperator, u::AbstractArray)
    u isa AbstractVecOrMat && @error "LinearAlgebra.ldiv! not defined for $(typeof(L)), $(typeof(u))."

    sizes = _mat_sizes(L, u)

    uu = _reshape(u, sizes[1])
    vv = _reshape(v, sizes[2])

    ldiv!(vv, L, uu)

    v
end

function LinearAlgebra.ldiv!(L::AbstractSciMLLinearOperator, u::AbstractArray)
    u isa AbstractVecOrMat && @error "LinearAlgebra.ldiv! not defined for $(typeof(L)), $(typeof(u))."

    sizes = _mat_sizes(L, u)

    uu = _reshape(u, sizes[1])

    ldiv!(L, uu)

    u
end
#
