"""
$(TYPEDEF)

Lazy block diagonal operator built from `AbstractSciMLOperator` blocks.

# Arguments

  - `ops`: Operators or matrices to place on the block diagonal. Matrix
    arguments are wrapped in `MatrixOperator`.

# Fields

$(FIELDS)

# Interface Rules

`BlockDiagonalOperator` applies each block to the corresponding slice of the
input and concatenates the results. Its size is the sum of block row and
column sizes. `update_coefficients[!]`, caching, and trait queries are
forwarded to each block, so a block diagonal operator has concretization,
in-place multiplication, or adjoint support only when the required component
operators do.

# Examples

```julia
using LinearAlgebra, SciMLOperators

A = MatrixOperator([1.0 2.0; 3.0 4.0])
B = MatrixOperator(Diagonal([5.0, 6.0, 7.0]))
L = BlockDiagonalOperator(A, B)

v = ones(5)
L * v == Matrix(L) * v
```
"""
struct BlockDiagonalOperator{
        T,
        O <: Tuple{Vararg{AbstractSciMLOperator}},
    } <: AbstractSciMLOperator{T}
    ops::O

    function BlockDiagonalOperator(ops::Tuple{Vararg{AbstractSciMLOperator}})
        @assert !isempty(ops)
        T = mapreduce(eltype, promote_type, ops)
        return new{T, typeof(ops)}(ops)
    end
end

function BlockDiagonalOperator(ops::Union{AbstractMatrix, AbstractSciMLOperator}...)
    return BlockDiagonalOperator(map(op -> op isa AbstractMatrix ? MatrixOperator(op) : op, ops))
end

getops(L::BlockDiagonalOperator) = L.ops

function Base.show(io::IO, L::BlockDiagonalOperator)
    print(io, "BlockDiagonalOperator(")
    show(io, L.ops[1])
    for i in 2:length(L.ops)
        print(io, ", ")
        show(io, L.ops[i])
    end
    return print(io, ")")
end

function Base.size(L::BlockDiagonalOperator)
    return (sum(op -> size(op, 1), L.ops), sum(op -> size(op, 2), L.ops))
end

function Base.convert(::Type{AbstractMatrix}, L::BlockDiagonalOperator)
    A = zeros(eltype(L), size(L))
    row_start = 1
    col_start = 1
    for op in L.ops
        m, n = size(op)
        rows = row_start:(row_start + m - 1)
        cols = col_start:(col_start + n - 1)
        A[rows, cols] .= convert(AbstractMatrix, op)
        row_start += m
        col_start += n
    end
    return A
end
has_concretization(L::BlockDiagonalOperator) = all(has_concretization, L.ops)

for op in (:adjoint, :transpose)
    @eval Base.$op(L::BlockDiagonalOperator) = BlockDiagonalOperator($op.(L.ops)...)
end
Base.conj(L::BlockDiagonalOperator) = BlockDiagonalOperator(conj.(L.ops)...)

function update_coefficients(L::BlockDiagonalOperator, u, p, t; kwargs...)
    ops = map(op -> update_coefficients(op, u, p, t; kwargs...), L.ops)
    return BlockDiagonalOperator(ops)
end

function update_coefficients!(L::BlockDiagonalOperator, u, p, t; kwargs...)
    for op in L.ops
        update_coefficients!(op, u, p, t; kwargs...)
    end
    return nothing
end

function Base.copy(L::BlockDiagonalOperator)
    return BlockDiagonalOperator(map(copy, L.ops))
end

function cache_internals(L::BlockDiagonalOperator, v::AbstractVecOrMat)
    ops = ()
    col_start = 1
    for op in L.ops
        n = size(op, 2)
        cols, col_start = _block_range(col_start, n)
        ops = (ops..., cache_operator(op, _block_view(v, cols)))
    end
    return BlockDiagonalOperator(ops)
end

isconstant(L::BlockDiagonalOperator) = all(isconstant, L.ops)
islinear(L::BlockDiagonalOperator) = all(islinear, L.ops)
Base.iszero(L::BlockDiagonalOperator) = all(iszero, L.ops)
has_adjoint(L::BlockDiagonalOperator) = all(has_adjoint, L.ops)
has_mul(L::BlockDiagonalOperator) = all(has_mul, L.ops)
has_mul!(L::BlockDiagonalOperator) = all(has_mul!, L.ops)

function _block_range(start::Int, len::Int)
    stop = start + len - 1
    return start:stop, stop + 1
end

function _block_view(v::AbstractMatrix, rows)
    return view(v, rows, :)
end
function _block_view(v::AbstractVector, rows)
    return view(v, rows)
end

function Base.:*(L::BlockDiagonalOperator, v::AbstractVecOrMat)
    @assert size(v, 1) == size(L, 2)
    T = promote_type(eltype(L), eltype(v))
    w = v isa AbstractMatrix ? similar(v, T, (size(L, 1), size(v, 2))) :
        similar(v, T, size(L, 1))
    return mul!(w, L, v)
end

function LinearAlgebra.mul!(
        w::AbstractVecOrMat, L::BlockDiagonalOperator, v::AbstractVecOrMat
    )
    @assert size(v, 1) == size(L, 2)
    @assert size(w, 1) == size(L, 1)
    row_start = 1
    col_start = 1
    for op in L.ops
        m, n = size(op)
        rows, row_start = _block_range(row_start, m)
        cols, col_start = _block_range(col_start, n)
        mul!(_block_view(w, rows), op, _block_view(v, cols))
    end
    return w
end

function LinearAlgebra.mul!(
        w::AbstractVecOrMat,
        L::BlockDiagonalOperator,
        v::AbstractVecOrMat,
        α,
        β
    )
    @assert size(v, 1) == size(L, 2)
    @assert size(w, 1) == size(L, 1)
    row_start = 1
    col_start = 1
    for op in L.ops
        m, n = size(op)
        rows, row_start = _block_range(row_start, m)
        cols, col_start = _block_range(col_start, n)
        mul!(_block_view(w, rows), op, _block_view(v, cols), α, β)
    end
    return w
end

function (L::BlockDiagonalOperator)(v::AbstractVecOrMat, u, p, t; kwargs...)
    L = update_coefficients(L, u, p, t; kwargs...)
    return L * v
end

function (L::BlockDiagonalOperator)(
        w::AbstractVecOrMat, v::AbstractVecOrMat, u, p, t; kwargs...
    )
    update_coefficients!(L, u, p, t; kwargs...)
    return mul!(w, L, v)
end

function (L::BlockDiagonalOperator)(
        w::AbstractVecOrMat, v::AbstractVecOrMat, u, p, t, α, β; kwargs...
    )
    update_coefficients!(L, u, p, t; kwargs...)
    return mul!(w, L, v, α, β)
end
