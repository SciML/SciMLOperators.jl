#
"""
    BatchedDiagonalOperator(diag; update_func=nothing, accepted_kwargs=())

Represents a time-dependent elementwise scaling (diagonal-scaling) operation.
Acts on `AbstractArray`s of the same size as `diag`. The update function is called
by `update_coefficients!` and is assumed to have the following signature:

    update_func(diag::AbstractVector,u,p,t; <accepted kwarg fields>) -> [modifies diag]
"""
struct BatchedDiagonalOperator{T,D,F} <: AbstractSciMLOperator{T}
    diag::D
    update_func::F

    function BatchedDiagonalOperator(
                                     diag::AbstractArray;
                                     update_func=nothing,
                                     accepted_kwargs=()
                                    )
        _update_func = preprocess_update_func(update_func, accepted_kwargs)
        new{
            eltype(diag),
            typeof(diag),
            typeof(_update_func)
           }(
             diag, _update_func,
            )
    end
end

function DiagonalOperator(u::AbstractArray; update_func=nothing, accepted_kwargs=())
    BatchedDiagonalOperator(u; update_func, accepted_kwargs)
end

# traits
Base.size(L::BatchedDiagonalOperator) = (N = size(L.diag, 1); (N, N))
Base.iszero(L::BatchedDiagonalOperator) = iszero(L.diag)
Base.transpose(L::BatchedDiagonalOperator) = L
Base.adjoint(L::BatchedDiagonalOperator) = conj(L)
function Base.conj(L::BatchedDiagonalOperator) # TODO - test this thoroughly
    diag = conj(L.diag)
    update_func = if isreal(L)
        L.update_func
    else
        (L,u,p,t; kwargs...) -> conj(L.update_func(conj(L.diag),u,p,t; kwargs...))
    end
    BatchedDiagonalOperator(diag; update_func=update_func)
end

LinearAlgebra.issymmetric(L::BatchedDiagonalOperator) = true
function LinearAlgebra.ishermitian(L::BatchedDiagonalOperator)
    if isreal(L)
        true
    else
        d = vec(L.diag)
        D = Diagonal(d)
        ishermitian(d)
    end
end
LinearAlgebra.isposdef(L::BatchedDiagonalOperator) = isposdef(Diagonal(vec(L.diag)))

isconstant(L::BatchedDiagonalOperator) = update_func_isconstant(L.update_func)
islinear(::BatchedDiagonalOperator) = true
has_adjoint(L::BatchedDiagonalOperator) = true
has_ldiv(L::BatchedDiagonalOperator) = all(x -> !iszero(x), L.diag)
has_ldiv!(L::BatchedDiagonalOperator) = has_ldiv(L)

getops(L::BatchedDiagonalOperator) = (L.diag,)

update_coefficients!(L::BatchedDiagonalOperator,u,p,t; kwargs...) = (L.update_func(L.diag,u,p,t; kwargs...); nothing)

# operator application
Base.:*(L::BatchedDiagonalOperator, u::AbstractVecOrMat) = L.diag .* u
Base.:\(L::BatchedDiagonalOperator, u::AbstractVecOrMat) = L.diag .\ u

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::BatchedDiagonalOperator, u::AbstractVecOrMat)
    V = vec(v)
    U = vec(u)
    d = vec(L.diag)
    D = Diagonal(d)
    mul!(V, D, U)

    v
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::BatchedDiagonalOperator, u::AbstractVecOrMat, α, β)
    V = vec(v)
    U = vec(u)
    d = vec(L.diag)
    D = Diagonal(d)
    mul!(V, D, U, α, β)

    v
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::BatchedDiagonalOperator, u::AbstractVecOrMat)
    V = vec(v)
    U = vec(u)
    d = vec(L.diag)
    D = Diagonal(d)
    ldiv!(V, D, U)

    v
end

function LinearAlgebra.ldiv!(L::BatchedDiagonalOperator, u::AbstractVecOrMat)
    U = vec(u)
    d = vec(L.diag)
    D = Diagonal(d)
    ldiv!(D, U)

    u
end
#
