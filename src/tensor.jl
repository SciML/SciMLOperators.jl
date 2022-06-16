
"""
    Lazy Tensor Product Operator

    TensorProductOperator(A, B) = A ⊗ B

    (A ⊗ B)(u) = vec(B * U * transpose(A))

    where U is a lazy representation of the vector u as
    a matrix with the appropriate size.
"""
struct TensorProductOperator{T,O,I,C} <: AbstractSciMLOperator{T}
    outer::O
    inner::I

    cache::C
    isset::Bool

    function TensorProductOperator(out, in, cache, isset)
        T = promote_type(eltype.((out, in))...)
        isset = cache !== nothing
        new{T,
            typeof(out),
            typeof(in),
            typeof(cache)
           }(
             out, in, cache, isset
            )
    end
end

function TensorProductOperator(out, in; cache = nothing)
    isset = cache !== nothing
    TensorProductOperator(out, in, cache, isset)
end

# constructors
TensorProductOperator(op::AbstractSciMLOperator) = op
TensorProductOperator(op::AbstractMatrix) = MatrixOperator(op)
TensorProductOperator(ops...) = reduce(TensorProductOperator, ops)

# overload ⊗ (\otimes)
⊗(ops::Union{AbstractMatrix,AbstractSciMLOperator}...) = TensorProductOperator(ops...)

# TODO - overload Base.kron
#Base.kron(ops::Union{AbstractMatrix,AbstractSciMLOperator}...) = TensorProductOperator(ops...)

# convert to matrix
Base.kron(ops::AbstractSciMLOperator...) = kron(convert.(AbstractMatrix, ops)...)

function Base.convert(::Type{AbstractMatrix}, L::TensorProductOperator)
    kron(convert(AbstractMatrix, L.outer), convert(AbstractMatrix, L.inner))
end

function SparseArrays.sparse(L::TensorProductOperator)
    kron(sparse(L.outer), sparse(L.inner))
end

#LinearAlgebra.opnorm(L::TensorProductOperator) = prod(opnorm, L.ops)

Base.size(L::TensorProductOperator) = size(L.inner) .* size(L.outer)

for op in (
           :adjoint,
           :transpose,
          )
    @eval function Base.$op(L::TensorProductOperator)
        TensorProductOperator(
                              $op(L.outer),
                              $op(L.inner);
                              cache = issquare(L.inner) ? L.cache : nothing
                             )
    end
end

getops(L::TensorProductOperator) = (L.outer, L.inner)
islinear(L::TensorProductOperator) = islinear(L.outer) & islinear(L.inner)
Base.iszero(L::TensorProductOperator) = iszero(L.outer) | iszero(L.inner)
has_adjoint(L::TensorProductOperator) = has_adjoint(L.outer) & has_adjoint(L.inner)
has_mul!(L::TensorProductOperator) = has_mul!(L.outer) & has_mul!(L.inner)
has_ldiv(L::TensorProductOperator) = has_ldiv(L.outer) & has_ldiv(L.inner)
has_ldiv!(L::TensorProductOperator) = has_ldiv!(L.outer) & has_ldiv!(L.inner)

# operator application
for op in (
           :*, :\,
          )
    @eval function Base.$op(L::TensorProductOperator, u::AbstractVecOrMat)
        mi, ni = size(L.inner)
        mo, no = size(L.outer)
        m , n  = size(L)
        k = size(u, 2)

        U = _reshape(u, (ni, no*k))
        C = $op(L.inner, U)

        V = if k > 1
            C = _reshape(C, (mi, no, k))
            V = similar( u, (mi, mo, k))

            @views for i=1:k
                V[:,:,i] = transpose($op(L.outer, transpose(C[:,:,i])))
            end

            V
        else
            transpose($op(L.outer, transpose(C)))
        end

        u isa AbstractMatrix ? _reshape(V, (m, k)) : _reshape(V, (m,))
    end
end

function cache_self(L::TensorProductOperator, u::AbstractVecOrMat)
    mi, _  = size(L.inner)
    _ , no = size(L.outer)
    k = size(u, 2)

    @set! L.cache = similar(u, (mi, no*k))
    L
end

function cache_internals(L::TensorProductOperator, u::AbstractVecOrMat) where{D}
    if !(L.isset)
        L = cache_self(L, u)
    end

    mi, ni = size(L.inner)
    _ , no = size(L.outer)
    k = size(u, 2)

    uinner = _reshape(u, (ni, no*k))
    uouter = _reshape(L.cache, (no, mi*k))

    @set! L.inner = cache_operator(L.inner, uinner)
    @set! L.outer = cache_operator(L.outer, uouter)
    L
end

# TODO - use permutedims!(dst,src,perm) for tensorproduct
function LinearAlgebra.mul!(v::AbstractVecOrMat, L::TensorProductOperator, u::AbstractVecOrMat)
    @assert L.isset "cache needs to be set up for operator of type $(typeof(L)).
    set up cache by calling cache_operator(L::AbstractSciMLOperator, u::AbstractArray)"

    mi, ni = size(L.inner)
    mo, no = size(L.outer)
    k = size(u, 2)

    C = L.cache
    U = _reshape(u, (ni, no*k))

    """
        v .= kron(B, A) * u
        V .= A * U * B'
    """

    println("inner")
    @show size.((C, L.inner, U))
    # C .= A * U
    mul!(C, L.inner, U)

    # V .= U * B' <===> V' .= B * C'
    if k>1
        V = _reshape(v, (mi, mo, k))
        C = _reshape(C, (mi, no, k))

        @views for i=1:k
            println("outer")
            @show size.((transpose(V[:,:,i]), L.outer, transpose(C[:,:,i])))
            mul!(transpose(V[:,:,i]), L.outer, transpose(C[:,:,i]))
        end
    else
        V = _reshape(v, (mi, mo))
        C = _reshape(C, (mi, no))
        mul!(transpose(V), L.outer, transpose(C))
    end

    v
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::TensorProductOperator, u::AbstractVecOrMat, α, β)
    @assert L.isset "cache needs to be set up for operator of type $(typeof(L)).
    set up cache by calling cache_operator(L::AbstractSciMLOperator, u::AbstractArray)"

    mi, ni = size(L.inner)
    mo, no = size(L.outer)
    k = size(u, 2)

    C = L.cache
    U = _reshape(u, (ni, no*k))

    """
        v .= α * kron(B, A) * u + β * v
        V .= α * (A * U * B') + β * v
    """

    # C .= A * U
    mul!(C, L.inner, U)

    # V = α(C * B') + β(V)
    if k>1
        V = _reshape(v, (mi, mo, k))
        C = _reshape(C, (mi, no, k))

        @views for i=1:k
            mul!(transpose(V[:,:,i]), L.outer, transpose(C[:,:,i]), α, β)
        end
    else
        V = _reshape(v, (mi, mo))
        C = _reshape(C, (mi, no))
        mul!(transpose(V), L.outer, transpose(C), α, β)
    end

    v
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::TensorProductOperator, u::AbstractVecOrMat)
    @assert L.isset "cache needs to be set up for operator of type $(typeof(L)).
    set up cache by calling cache_operator(L::AbstractSciMLOperator, u::AbstractArray)"

    mi, ni = size(L.inner)
    mo, no = size(L.outer)
    k = size(u, 2)

    C = L.cache
    U = _reshape(u, (ni, no*k))

    """
        v .= kron(B, A) ldiv u
        V .= (A ldiv U) / B'
    """

    # C .= A \ U
    ldiv!(C, L.inner, U)

    # V .= C / B' <===> V' .= B \ C'
    if k>1
        C = _reshape(C, (mi, no, k))
        V = _reshape(v, (mi, mo, k))

        @views for i=1:k
            ldiv!(transpose(V[:,:,i]), L.outer, transpose(C[:,:,i]))
        end
    else
        V = _reshape(v, (mi, mo))
        C = _reshape(C, (mi, no))
        ldiv!(transpose(V), L.outer, transpose(C))
    end

    v
end

function LinearAlgebra.ldiv!(L::TensorProductOperator, u::AbstractVecOrMat)
    @assert L.isset "cache needs to be set up for operator of type $(typeof(L)).
    set up cache by calling cache_operator(L::AbstractSciMLOperator, u::AbstractArray)"

    mi, ni = size(L.inner)
    _ , no = size(L.outer)
    k = size(u, 2)

    U = _reshape(u, (ni, no*k))

    """
        u .= kron(B, A) ldiv u
        U .= (A ldiv U) / B'
    """

    # U .= A \ U
    ldiv!(L.inner, U)

    # U .= U / B' <===> U' .= B \ U'
    if k>1
        U = _reshape(U, (mi, no, k))

        @views for i=1:k
            ldiv!(L.outer, transpose(U[:,:,i]))
        end
    else
        ldiv!(L.outer, transpose(U))
    end

    u
end
#
