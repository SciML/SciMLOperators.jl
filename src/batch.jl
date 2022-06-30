#
struct BatchedDiagonalOperator{T,D,F} <: AbstractSciMLOperator
    diag::D
    update_func::F

    function BatchedDiagonalOperator(
                                     diag::AbstractVecOrMat;
                                     update_func=DEFAULT_UPDATE_FUNC
                                    )
        new{
            eltype(diag),
            typeof(diag),
            typeof(update_func)
           }(
             D, update_func,
            )
    end
end

function DiagonalOperator(u::AbstractVecOrMat; update_func=DEFAULT_UPDATE_FUNC)
    BatchedDiagonalOperator(u; update_func=diag_update_func)
end

Base.size(L::BatchedDiagonalOperator) = (N = size(L.diag, 1); (N, N))
Base.transpose(L::BatchedDiagonalOperator) = L
Base.transpose(L::BatchedDiagonalOperator) = conj(L)
function Base.conj(L::BatchedDiagonalOperator) # TODO - test this thoroughly
    diag = conj(L.diag)
    update_func = (L,u,p,t) -> conj(L.update_func(conj(L.diag),u,p,t))
    BatchedDiagonalOperator(diag; update_func=update_func)
end

update_coefficients!(L::BatchedDiagonalOperator,u,p,t) = (L.update_func(L.diag,u,p,t); L)
#
# traits
LinearAlgebra.issymmetric(L::BatchedDiagonalOperator) = true
LinearAlgebra.ishermitian(L::BatchedDiagonalOperator) = true
LinearAlgebra.isposdef(isposdef(Diagonal(_vec(L.diag))))

issquare(L::BatchedDiagonalOperator) = true
has_ldiv(L::BatchedDiagonalOperator) = all(x -> !iszero(x), L.diag)
has_ldiv!(L::BatchedDiagonalOperator) = has_ldiv(L)

# operator application
Base.:*(L::BatchedDiagonalOperator, u::AbstractVecOrMat) = L.diag .* u
Base.:\(L::BatchedDiagonalOperator, u::AbstractVecOrMat) = L.diag .\ u

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::BatchedDiagonalOperator, u::AbstractVecOrMat)
    V = _vec(v)
    U = _vec(u)
    d = _vec(L.diag)
    D = Diagonal(d)
    mul!(V, D, U)

    v
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::BatchedDiagonalOperator, u::AbstractVecOrMat, α, β)
    V = _vec(v)
    U = _vec(u)
    d = _vec(L.diag)
    D = Diagonal(d)
    mul!(V, D, U, α, β)

    v
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::BatchedDiagonalOperator, u::AbstractVecOrMat)
    V = _vec(v)
    U = _vec(u)
    d = _vec(L.diag)
    D = Diagonal(d)
    ldiv!(V, D, U)

    v
end

function LinearAlgebra.ldiv!(L::BatchedDiagonalOperator, u::AbstractVecOrMat)
    U = _vec(u)
    d = _vec(L.diag)
    D = Diagonal(d)
    ldiv!(D, U)

    u
end
#
