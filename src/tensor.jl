#
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
TensorProductOperator(Io::IdentityOperator{No}, Ii::IdentityOperator{Ni}) where{No,Ni} = IdentityOperator{No*Ni}()

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
function Base.:*(L::TensorProductOperator, u::AbstractVecOrMat)
    mi, ni = size(L.inner)
    mo, no = size(L.outer)
    m , n  = size(L)
    k = size(u, 2)

    U = _reshape(u, (ni, no*k))
    C = L.inner * U

    V = outer_mul(L, u, C)

    u isa AbstractMatrix ? _reshape(V, (m, k)) : _reshape(V, (m,))
end

function Base.:\(L::TensorProductOperator, u::AbstractVecOrMat)
    mi, ni = size(L.inner)
    mo, no = size(L.outer)
    m , n  = size(L)
    k = size(u, 2)

    U = _reshape(u, (ni, no*k))
    C = L.inner \ U

    V = outer_div(L, u, C)

    u isa AbstractMatrix ? _reshape(V, (m, k)) : _reshape(V, (m,))
end

function cache_self(L::TensorProductOperator, u::AbstractVecOrMat)
    mi, _  = size(L.inner)
    mo, no = size(L.outer)
    k = size(u, 2)

    c1 = similar(u, (mi, no*k))  # c1 = L.inner * u
    c2 = similar(u, (no, mi, k)) # permut (2, 1, 3)
    c3 = similar(u, (mo, mi*k))  # c3 = L.outer * c2
    c4 = similar(u, (mo*mi, k))  # cache v in 5 arg mul!

    @set! L.cache = (c1, c2, c3, c4,)
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
    uouter = L.cache[2]

    @set! L.inner = cache_operator(L.inner, uinner)
    @set! L.outer = cache_operator(L.outer, uouter)
    L
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::TensorProductOperator, u::AbstractVecOrMat)
    @assert L.isset "cache needs to be set up for operator of type $(typeof(L)).
    set up cache by calling cache_operator(L::AbstractSciMLOperator, u::AbstractArray)"

    mi, ni = size(L.inner)
    mo, no = size(L.outer)
    k = size(u, 2)

    C1, C2, C3, _ = L.cache
    U = _reshape(u, (ni, no*k))

    """
        v .= kron(B, A) * u
        V .= A * U * B'
    """

    # C .= A * U
    mul!(C1, L.inner, U)

    # V .= U * B' <===> V' .= B * C'
    outer_mul!(v, L, u)

    v
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::TensorProductOperator, u::AbstractVecOrMat, α, β)
    @assert L.isset "cache needs to be set up for operator of type $(typeof(L)).
    set up cache by calling cache_operator(L::AbstractSciMLOperator, u::AbstractArray)"

    mi, ni = size(L.inner)
    mo, no = size(L.outer)
    k = size(u, 2)

    C1, C2, C3, c4 = L.cache
    U = _reshape(u, (ni, no*k))

    """
        v .= α * kron(B, A) * u + β * v
        V .= α * (A * U * B') + β * v
    """

    # C .= A * U
    mul!(C1, L.inner, U)

    # V = α(C * B') + β(V)
    outer_mul!(v, L, u, α, β)

    v
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::TensorProductOperator, u::AbstractVecOrMat)
    @assert L.isset "cache needs to be set up for operator of type $(typeof(L)).
    set up cache by calling cache_operator(L::AbstractSciMLOperator, u::AbstractArray)"

    mi, ni = size(L.inner)
    mo, no = size(L.outer)
    k = size(u, 2)

    C1, C2, C3, _ = L.cache
    U = _reshape(u, (ni, no*k))

    """
        v .= kron(B, A) ldiv u
        V .= (A ldiv U) / B'
    """

    # C .= A \ U
    ldiv!(C1, L.inner, U)

    # V .= C / B' <===> V' .= B \ C'
    outer_div!(v, L, u)

    v
end

function LinearAlgebra.ldiv!(L::TensorProductOperator, u::AbstractVecOrMat)
    @assert L.isset "cache needs to be set up for operator of type $(typeof(L)).
    set up cache by calling cache_operator(L::AbstractSciMLOperator, u::AbstractArray)"

    ni = size(L.inner, 1)
    no = size(L.outer, 1)
    k  = size(u, 2)

    U = _reshape(u, (ni, no*k))

    """
        u .= kron(B, A) ldiv u
        U .= (A ldiv U) / B'
    """

    # U .= A \ U
    ldiv!(L.inner, U)

    # U .= U / B' <===> U' .= B \ U'
    outer_div!(L, u)

    u
end


# helper functions
const PERM = (2, 1, 3)

function outer_mul(L::TensorProductOperator, u::AbstractVecOrMat, C::AbstractVecOrMat)
    if L.outer isa IdentityOperator
        return C
    elseif L.outer isa ScalarOperator
        return L.outer.λ * outer_mul(L.outer.L, C)
    end

    k = size(u, 2)
    if k == 1
        return transpose(L.outer * transpose(C))
    end

    mi, ni = size(L.inner)
    mo, no = size(L.outer)
    m , n  = size(L)

    C = _reshape(C, (mi, no, k))
    C = permutedims(C, PERM)
    C = _reshape(C, (no, mi*k))

    V = L.outer * C
    V = _reshape(V, (mo, mi, k))
    V = permutedims(V, PERM)

    V
end

function outer_mul!(v::AbstractVecOrMat, L::TensorProductOperator, u::AbstractVecOrMat)
    C1 = first(L.cache)

    if L.outer isa IdentityOperator
        copyto!(v, C1)
        return v
    elseif L.outer isa ScalarOperator
        outer_mul!(v, L.outer.L, u)
        lmul!(L.outer.λ, v)
        return v
    end

    mi, ni = size(L.inner)
    mo, no = size(L.outer)
    m , n  = size(L)
    k = size(u, 2)

    if k == 1
        V  = _reshape(v, (mi, mo))
        C1 = _reshape(C1, (mi, no))
        mul!(transpose(V), L.outer, transpose(C1))
        return v
    end

    _, C2, C3, _ = L.cache

    C1 = _reshape(C1, (mi, no, k))
    permutedims!(C2, C1, PERM)
    C2 = _reshape(C2, (no, mi*k))
    mul!(C3, L.outer, C2)
    C3 = _reshape(C3, (mo, mi, k))
    V  = _reshape(v , (mi, mo, k))
    permutedims!(V, C3, PERM)

    v
end

function outer_mul!(v::AbstractVecOrMat, L::TensorProductOperator, u::AbstractVecOrMat, α, β)
    C1 = first(L.cache)

    if L.outer isa IdentityOperator
        c1 = _reshape(C1, (m, k))
        axpby!(α, c1, β, v)
        return v
#   elseif L.outer isa ScalarOperator
#       outer_mul!(v, L.outer.L, u, α, β) # <- figure out which cache is still unused
#       lmul!(L.outer.λ, v)
#       return v
    end

    mi, ni = size(L.inner)
    mo, no = size(L.outer)
    m , n  = size(L)
    k = size(u, 2)

    if k == 1
        V  = _reshape(v, (mi, mo))
        C1 = _reshape(C1, (mi, no))
        mul!(transpose(V), L.outer, transpose(C1), α, β)
        return v
    end

    _, C2, C3, c4 = L.cache

    C1 = _reshape(C1, (mi, no, k))
    permutedims!(C2, C1, PERM)
    C2 = _reshape(C2, (no, mi*k))
    mul!(C3, L.outer, C2)
    C3 = _reshape(C3, (mo, mi, k))
    V  = _reshape(v , (mi, mo, k))
    copy!(c4, v)
    permutedims!(V, C3, PERM)
    axpby!(β, c4, α, v)

    v
end

function outer_div(L::TensorProductOperator, u::AbstractVecOrMat, C::AbstractVecOrMat)
    if L.outer isa IdentityOperator
        return c
    elseif L.outer isa ScalarOperator
        return L.outer.λ \ outer_div(L.outer.L, c)
    end

    k = size(u, 2)
    if k == 1
        return transpose(L.outer \ transpose(C))
    end

    mi, ni = size(L.inner)
    mo, no = size(L.outer)
    m , n  = size(L)

    C = _reshape(C, (mi, no, k))
    C = permutedims(C, PERM)
    C = _reshape(C, (no, mi*k))

    V = L.outer \ C
    V = _reshape(V, (mo, mi, k))
    V = permutedims(V, PERM)

    V
end

function outer_div!(v::AbstractVecOrMat, L::TensorProductOperator, u::AbstractVecOrMat)
    C1 = first(L.cache)

    if L.outer isa IdentityOperator
        copyto!(v, C1)
        return v
    elseif L.outer isa ScalarOperator
        outer_div!(v, L.outer.L, u)
        ldiv!(L.outer.λ, v)
        return v
    end

    mi, ni = size(L.inner)
    mo, no = size(L.outer)
    m , n  = size(L)
    k = size(u, 2)

    if k == 1
        V  = _reshape(v, (mi, mo))
        C1 = _reshape(C1, (mi, no))
        ldiv!(transpose(V), L.outer, transpose(C1))
        return v
    end

    _, C2, C3, _ = L.cache

    C1 = _reshape(C1, (mi, no, k))
    permutedims!(C2, C1, PERM)
    C2 = _reshape(C2, (no, mi*k))
    ldiv!(C3, L.outer, C2)
    C3 = _reshape(C3, (mo, mi, k))
    V  = _reshape(v , (mi, mo, k))
    permutedims!(V, C3, PERM)

    v
end

function outer_div!(L::TensorProductOperator, u::AbstractVecOrMat)
    if L.outer isa IdentityOperator
        return u
    elseif L.outer isa ScalarOperator
        outer_div!(L.outer.L, u)
        ldiv!(L.outer.λ, u)
        return u
    end

    mi, ni = size(L.inner)
    mo, no = size(L.outer)
    m , n  = size(L)
    k = size(u, 2)

    U = _reshape(u, (ni, no*k))

    if k == 1
        ldiv!(L.outer, transpose(U))
        return u
    end

    C = first(L.cache)

    U = _reshape(U, (ni, no, k))
    C = _reshape(C, (no, ni, k))
    permutedims!(C, U, PERM)
    ldiv!(L.outer, C)
    permutedims!(U, C, PERM)

    u
end
#
