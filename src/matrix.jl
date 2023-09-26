#
"""
$SIGNATURES

Represents a linear operator given by an `AbstractMatrix` that may be
applied to an `AbstractVecOrMat`. Its state is updated by the user-provided
`update_func` during operator evaluation (`L([v,], u, p, t)`), or by calls
to `update_coefficients[!](L, u, p, t)`. Both recursively call the
`update_function`, `update_func` which is assumed to have the signature

    update_func(A::AbstractMatrix, u, p, t; <accepted kwargs>) -> newA

or

    update_func!(A::AbstractMatrix, u ,p , t; <accepted kwargs>) -> [modifies A]

The set of keyword-arguments accepted by `update_func[!]` must be provided
to `MatrixOperator` via the kwarg `accepted_kwargs` as a tuple of `Symbol`s.
`kwargs` cannot be passed down to `update_func[!]` if `accepted_kwargs`
are not provided.

$(UPDATE_COEFFS_WARNING)

# Interface

Lazy matrix algebra is defined for `AbstractSciMLOperator`s. The Interface
supports lazy addition, subtraction, multiplication, inversion,
adjoints, transposes.

# Example

Out-of-place update and usage
```
u = rand(4)
p = rand(4, 4)
t = rand()

mat_update = (A, u, p, t; scale = 0.0) -> t * p
M = MatrixOperator(0.0; update_func = mat_update; accepted_kwargs = (:scale,))

L = M * M + 3I
L = cache_operator(M, u)

# update L and evaluate
v = L(u, p, t; scale = 1.0)
```

In-place update and usage
```
v = zero(4)
u = rand(4)
p = nothing
t = rand()

mat_update! = (A, u, p, t; scale = 0.0) -> (copy!(A, p); lmul!(t, A))
M = MatrixOperator(zeros(4, 4); update_func! = val_update!; accepted_kwargs = (:scale,))
L = M * M + 3I

# update L in-place and evaluate
L(v, u, p, t; scale = 1.0)
```
"""
struct MatrixOperator{T,AT<:AbstractMatrix{T},F,F!} <: AbstractSciMLOperator{T}
    A::AT
    update_func::F
    update_func!::F!

    function MatrixOperator(A, update_func, update_func!)

        new{
            eltype(A),
            typeof(A),
            typeof(update_func),
            typeof(update_func!),
           }(
             A, update_func, update_func!,
            )
    end
end

function MatrixOperator(A;
                        update_func = DEFAULT_UPDATE_FUNC,
                        update_func! = DEFAULT_UPDATE_FUNC,
                        accepted_kwargs = nothing,)

    update_func  = preprocess_update_func(update_func , accepted_kwargs)
    update_func! = preprocess_update_func(update_func!, accepted_kwargs)

    MatrixOperator(A, update_func, update_func!)
end

# constructors
function Base.similar(L::MatrixOperator, ::Type{T}, dims::Dims) where{T}
    MatrixOperator(similar(L.A, T, dims))
end

# traits
@forward MatrixOperator.A (
                           LinearAlgebra.issymmetric,
                           LinearAlgebra.ishermitian,
                           LinearAlgebra.isposdef,

                           issquare,
                           has_ldiv,
                           has_ldiv!,
                          )

isconvertible(::MatrixOperator) = true
islinear(::MatrixOperator) = true

function Base.show(io::IO, L::MatrixOperator)
    a, b = size(L)
    print(io, "MatrixOperator($a × $b)")
end
Base.size(L::MatrixOperator) = size(L.A)
Base.iszero(L::MatrixOperator) = iszero(L.A)
for op in (
           :adjoint,
           :transpose,
          )
    @eval function Base.$op(L::MatrixOperator)
        isconstant(L) && return MatrixOperator($op(L.A))

        update_func  = (A, u, p, t; kwargs...) -> $op(L.update_func( $op(L.A), u, p, t; kwargs...))
        update_func! = (A, u, p, t; kwargs...) -> $op(L.update_func!($op(L.A), u, p, t; kwargs...))

        MatrixOperator($op(L.A);
                       update_func = update_func,
                       update_func! = update_func!,
                       accepted_kwargs = NoKwargFilter(),
                      )
    end
end

function Base.conj(L::MatrixOperator)
    isconstant(L) && return MatrixOperator(conj(L.A))

    update_func  = (A, u, p, t; kwargs...) -> conj(L.update_func(conj(L.A), u, p, t; kwargs...))
    update_func! = (A, u, p, t; kwargs...) -> begin
        L.update_func!(conj!(L.A), u, p, t; kwargs...)
        conj!(L.A)
    end

    MatrixOperator(conj(L.A);
                   update_func = update_func,
                   update_func! = update_func!,
                   accepted_kwargs = NoKwargFilter(),
                  )
end

has_adjoint(A::MatrixOperator) = has_adjoint(A.A)
getops(L::MatrixOperator) = (L.A,)
isconstant(L::MatrixOperator) = update_func_isconstant(L.update_func) & update_func_isconstant(L.update_func!)

function update_coefficients(L::MatrixOperator, u, p, t; kwargs...)
    @set! L.A = L.update_func(L.A, u, p, t; kwargs...)
end

function update_coefficients!(L::MatrixOperator, u, p, t; kwargs...)
    L.update_func!(L.A, u, p, t; kwargs...)
end

SparseArrays.sparse(L::MatrixOperator) = sparse(L.A)
SparseArrays.issparse(L::MatrixOperator) = issparse(L.A)

# TODO - add tests for MatrixOperator indexing
# propagate_inbounds here for the getindex fallback
Base.@propagate_inbounds Base.convert(::Type{AbstractMatrix}, L::MatrixOperator) = convert(AbstractMatrix, L.A)
Base.@propagate_inbounds Base.setindex!(L::MatrixOperator, v, i::Int) = (L.A[i] = v)
Base.@propagate_inbounds Base.setindex!(L::MatrixOperator, v, I::Vararg{Int, N}) where{N} = (L.A[I...] = v)

Base.eachcol(L::MatrixOperator) = eachcol(L.A)
Base.eachrow(L::MatrixOperator) = eachrow(L.A)
Base.length(L::MatrixOperator) = length(L.A)
Base.iterate(L::MatrixOperator,args...) = iterate(L.A,args...)
Base.axes(L::MatrixOperator) = axes(L.A)
Base.eachindex(L::MatrixOperator) = eachindex(L.A)
Base.IndexStyle(::Type{<:MatrixOperator{T,AType}}) where{T,AType} = Base.IndexStyle(AType)
Base.copyto!(L::MatrixOperator, rhs) = (copyto!(L.A, rhs); L)
Base.copyto!(L::MatrixOperator, rhs::Base.Broadcast.Broadcasted{<:StaticArraysCore.StaticArrayStyle}) = (copyto!(L.A, rhs); L)
Base.Broadcast.broadcastable(L::MatrixOperator) = L
Base.ndims(::Type{<:MatrixOperator{T,AType}}) where{T,AType} = ndims(AType)
ArrayInterface.issingular(L::MatrixOperator) = ArrayInterface.issingular(L.A)
Base.copy(L::MatrixOperator) = MatrixOperator(copy(L.A);update_func=L.update_func, accepted_kwargs=NoKwargFilter())

# operator application
Base.:*(L::MatrixOperator, u::AbstractVecOrMat) = L.A * u
Base.:\(L::MatrixOperator, u::AbstractVecOrMat) = L.A \ u
LinearAlgebra.mul!(v::AbstractVecOrMat, L::MatrixOperator, u::AbstractVecOrMat) = mul!(v, L.A, u)
LinearAlgebra.mul!(v::AbstractVecOrMat, L::MatrixOperator, u::AbstractVecOrMat, α, β) = mul!(v, L.A, u, α, β)
LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::MatrixOperator, u::AbstractVecOrMat) = ldiv!(v, L.A, u)
LinearAlgebra.ldiv!(L::MatrixOperator, u::AbstractVecOrMat) = ldiv!(L.A, u)

"""
$SIGNATURES

Represents an elementwise scaling (diagonal-scaling) operation that may
be applied to an `AbstractVecOrMat`. When `diag` is an `AbstractVector`
of length N, `L = DiagonalOpeator(diag, ...)` can be applied to
`AbstractArray`s with `size(u, 1) == N`. Each column of the `u` will be
scaled by `diag`, as in `LinearAlgebra.Diagonal(diag) * u`.

When `diag` is a multidimensional array, `L = DiagonalOperator(diag, ...)` forms
an operator of size `(N, N)` where `N = size(diag, 1)` is the leading length of `diag`.
`L` then is the elementwise-scaling operation on arrays of `length(u) = length(diag)`
with leading length `size(u, 1) = N`.

Its state is updated by the user-provided `update_func` during operator
evaluation (`L([v,], u, p, t)`), or by calls to
`update_coefficients[!](L, u, p, t)`. Both recursively call the
`update_function`, `update_func` which is assumed to have the signature

    update_func(diag::AbstractVecOrMat, u, p, t; <accepted kwargs>) -> new_diag
or

    update_func!(diag::AbstractVecOrMat, u, p, t; <accepted kwargs>) -> [modifies diag]

The set of keyword-arguments accepted by `update_func[!]` must be provided
to `MatrixOperator` via the kwarg `accepted_kwargs` as a tuple of `Symbol`s.
`kwargs` cannot be passed down to `update_func[!]` if `accepted_kwargs`
are not provided.

$(UPDATE_COEFFS_WARNING)

# Example

"""
function DiagonalOperator(diag::AbstractVector;
                          update_func = DEFAULT_UPDATE_FUNC,
                          update_func! = DEFAULT_UPDATE_FUNC,
                          accepted_kwargs = nothing,
                         )

    diag_update_func = update_func_isconstant(update_func) ? update_func :
        (A, u, p, t; kwargs...) -> update_func(A.diag, u, p, t; kwargs...) |> Diagonal

    diag_update_func! = update_func_isconstant(update_func!) ? update_func! :
        (A, u, p, t; kwargs...) -> update_func!(A.diag, u, p, t; kwargs...)

    MatrixOperator(Diagonal(diag);
                   update_func = diag_update_func,
                   update_func! = diag_update_func!,
                   accepted_kwargs = accepted_kwargs,
                  )
end
LinearAlgebra.Diagonal(L::MatrixOperator) = MatrixOperator(Diagonal(L.A))

function Base.show(io::IO, L::MatrixOperator{T,<:Diagonal}) where{T}
    n = length(L.A.diag)
    print(io, "DiagonalOperator($n × $n)")
end

const AdjointFact = isdefined(LinearAlgebra, :AdjointFactorization) ? LinearAlgebra.AdjointFactorization : Adjoint
const TransposeFact = isdefined(LinearAlgebra, :TransposeFactorization) ? LinearAlgebra.TransposeFactorization : Transpose

"""
$SIGNATURES

Stores an operator and its factorization (or inverse operator).
Supports left division and `ldiv!` via `F`, and operator evaluation
via `L`.
"""
struct InvertibleOperator{T,LT,FT} <: AbstractSciMLOperator{T}
    L::LT
    F::FT

    function InvertibleOperator(L, F)
        @assert has_ldiv(F) | has_ldiv!(F) "$F is not invertible"
        T = promote_type(eltype(L), eltype(F))

        new{T,typeof(L),typeof(F)}(L, F)
    end
end

# constructor
function LinearAlgebra.factorize(L::AbstractSciMLOperator)
    fact = factorize(convert(AbstractMatrix, L))
    InvertibleOperator(L, fact)
end

for fact in (
             :lu, :lu!,
             :qr, :qr!,
             :cholesky, :cholesky!,
             :ldlt, :ldlt!,
             :bunchkaufman, :bunchkaufman!,
             :lq, :lq!,
             :svd, :svd!,
            )

    @eval LinearAlgebra.$fact(L::AbstractSciMLOperator, args...) =
        InvertibleOperator(L, $fact(convert(AbstractMatrix, L), args...))
    @eval LinearAlgebra.$fact(L::AbstractSciMLOperator; kwargs...) =
        InvertibleOperator(L, $fact(convert(AbstractMatrix, L); kwargs...))
end

function Base.convert(::Type{<:Factorization}, L::InvertibleOperator{T,LT,<:Factorization}) where{T,LT}
    L.F
end

Base.convert(::Type{AbstractMatrix}, L::InvertibleOperator) = convert(AbstractMatrix, L.L)

# traits
function Base.show(io::IO, L::InvertibleOperator)
    print(io, "InvertibleOperator(")
    show(io, L.L)
    print(io, ", ")
    show(io, L.F)
    print(io, ")")
end

Base.size(L::InvertibleOperator) = size(L.L)
Base.transpose(L::InvertibleOperator) = InvertibleOperator(transpose(L.L), transpose(L.F))
Base.adjoint(L::InvertibleOperator) = InvertibleOperator(L.L', L.F')
Base.conj(L::InvertibleOperator) = InvertibleOperator(conj(L.L), conj(L.F))
Base.resize!(L::InvertibleOperator, n::Integer) = (resize!(L.L, n); resize!(L.F, n); L)
LinearAlgebra.opnorm(L::InvertibleOperator{T}, p=2) where{T} = one(T) / opnorm(L.F)
LinearAlgebra.issuccess(L::InvertibleOperator) = issuccess(L.F)

function update_coefficients(L::InvertibleOperator, u, p, t)
    @set! L.L = update_coefficients(L.L, u, p, t)
    @set! L.F = update_coefficients(L.F, u, p, t)
    L
end

getops(L::InvertibleOperator) = (L.L, L.F,)
islinear(L::InvertibleOperator) = islinear(L.L)
isconvertible(L::InvertibleOperator) = isconvertible(L.L)

@forward InvertibleOperator.L (
                               # LinearAlgebra
                               LinearAlgebra.issymmetric,
                               LinearAlgebra.ishermitian,
                               LinearAlgebra.isposdef,

                               # SciML
                               isconstant,
                               has_adjoint,
                               has_mul,
                               has_mul!,
                              )

has_ldiv(L::InvertibleOperator) = has_mul(L.F)
has_ldiv!(L::InvertibleOperator) = has_ldiv!(L.F)

function cache_internals(L::InvertibleOperator, u::AbstractVecOrMat)

    @set! L.L = cache_operator(L.L, u)
    @set! L.F = cache_operator(L.F, u)

    L
end

# operator application
Base.:*(L::InvertibleOperator, x::AbstractVecOrMat) = L.L * x
Base.:\(L::InvertibleOperator, x::AbstractVecOrMat) = L.F \ x
LinearAlgebra.mul!(v::AbstractVecOrMat, L::InvertibleOperator, u::AbstractVecOrMat) = mul!(v, L.L, u)
LinearAlgebra.mul!(v::AbstractVecOrMat, L::InvertibleOperator, u::AbstractVecOrMat,α, β) = mul!(v, L.L, u, α, β)
LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::InvertibleOperator, u::AbstractVecOrMat) = ldiv!(v, L.F, u)
LinearAlgebra.ldiv!(L::InvertibleOperator, u::AbstractVecOrMat) = ldiv!(L.F, u)

"""
$SIGNATURES

Represents a generalized affine operation (`v = A * u + B * b`) that may
be applied to an `AbstractVecOrMat`. The user-provided update functions,
`update_func[!]` update the `AbstractVecOrMat` `b`, and are called
during operator evaluation (`L([v,], u, p, t)`), or by calls
to `update_coefficients[!](L, u, p, t)`. The update functions are
assumed to have the syntax

    update_func(b::AbstractVecOrMat, u, p, t; <accepted kwargs>) -> new_b
or

    update_func!(b::AbstractVecOrMat, u ,p , t; <accepted kwargs>) -> [modifies b]

and `B`, `b` are expected to have an appropriate size so that
`A * u + B * b` makes sense. Specifically, `size(A, 1) == size(B, 1)`, and
`size(u, 2) == size(b, 2)`.

The set of keyword-arguments accepted by `update_func[!]` must be provided
to `AffineOperator` via the kwarg `accepted_kwargs` as a tuple of `Symbol`s.
`kwargs` cannot be passed down to `update_func[!]` if `accepted_kwargs`
are not provided.

# Example

```
u = rand(4)
p = rand(4)
t = rand()

A = MatrixOperator(rand(4, 4))
B = MatrixOperator(rand(4, 4))

vec_update_func = (b, u, p, t) -> p * t
L = AffineOperator(A, B, zero(4); update_func = vec_update_func)
L = cache_operator(M, u)

# update L and evaluate
v = L(u, p, t) # == A * u + B * (p * t)
```

"""
struct AffineOperator{T,AT,BT,bT,F,F!} <: AbstractSciMLOperator{T}
    A::AT
    B::BT
    b::bT

    update_func::F # updates b
    update_func!::F! # updates b

    function AffineOperator(A, B, b, update_func, update_func!)
        T = promote_type(eltype.((A,B,b))...)

        new{T,
            typeof(A),
            typeof(B),
            typeof(b),
            typeof(update_func),
            typeof(update_func!),
           }(
             A, B, b, update_func, update_func!,
            )
    end
end

function AffineOperator(A::Union{AbstractMatrix,AbstractSciMLOperator},
                        B::Union{AbstractMatrix,AbstractSciMLOperator},
                        b::AbstractArray;
                        update_func = DEFAULT_UPDATE_FUNC,
                        update_func! = DEFAULT_UPDATE_FUNC,
                        accepted_kwargs = nothing,
                       )

    @assert size(A, 1) == size(B, 1) "Dimension mismatch: A, B don't output vectors
    of same size"

    update_func  = preprocess_update_func(update_func , accepted_kwargs)
    update_func! = preprocess_update_func(update_func!, accepted_kwargs)

    A = A isa AbstractMatrix ? MatrixOperator(A) : A
    B = B isa AbstractMatrix ? MatrixOperator(B) : B

    AffineOperator(A, B, b, update_func, update_func!)
end

"""
$SIGNATURES

Represents the affine operation `v = I * u + I * b`. The update functions,
`update_func[!]` update the state of `AbstractVecOrMat ` `b`. See
documentation of `AffineOperator` for more details.
"""
function AddVector(b::AbstractVecOrMat;
                   update_func = DEFAULT_UPDATE_FUNC,
                   update_func! = DEFAULT_UPDATE_FUNC,
                   accepted_kwargs = nothing
                  )

    N  = size(b, 1)
    Id = IdentityOperator(N)

    AffineOperator(Id, Id, b;
                   update_func = update_func,
                   update_func! = update_func!,
                   accepted_kwargs = accepted_kwargs,
                  )
end

"""
$SIGNATURES

Represents the affine operation `v = I * u + B * b`. The update functions,
`update_func[!]` update the state of `AbstractVecOrMat ` `b`. See
documentation of `AffineOperator` for more details.
"""
function AddVector(B, b::AbstractVecOrMat;
                   update_func = DEFAULT_UPDATE_FUNC,
                   update_func! = DEFAULT_UPDATE_FUNC,
                   accepted_kwargs=nothing
                  )

    N = size(B, 1)
    Id = IdentityOperator(N)

    AffineOperator(Id, B, b;
                   update_func = update_func,
                   update_func! = update_func!,
                   accepted_kwargs = accepted_kwargs,
                  )
end

function update_coefficients(L::AffineOperator, u, p, t; kwargs...)
    @set! L.A = update_coefficients(L.A, u, p, t; kwargs...)
    @set! L.B = update_coefficients(L.B, u, p, t; kwargs...)
    @set! L.b = L.update_func(L.b, u, p, t; kwargs...)
end

function update_coefficients!(L::AffineOperator, u, p, t; kwargs...)
    L.update_func!(L.b, u, p, t; kwargs...)
    for op in getops(L)
        update_coefficients!(op, u, p, t; kwargs...)
    end
    nothing
end

function isconstant(L::AffineOperator)
    update_func_isconstant(L.update_func) &
    update_func_isconstant(L.update_func!) &
    all(isconstant, (L.A, L.B))
end

getops(L::AffineOperator) = (L.A, L.B, L.b)

islinear(::AffineOperator) = false
isconvertible(::AffineOperator) = false

function Base.show(io::IO, L::AffineOperator)
    show(io, L.A)
    print(io, " + ")
    show(io, L.B)
    print(io, " * ")

    if L.b isa AbstractVector
        n = size(L.b, 1)
        print(io, "AbstractVector($n)")
    elseif L.b isa AbstractMatrix
        n, k = size(L.b)
        print(io, "AbstractMatrix($n × $k)")
    end
end

Base.size(L::AffineOperator) = size(L.A)
Base.iszero(L::AffineOperator) = all(iszero, getops(L))
function Base.resize!(L::AffineOperator, n::Integer)

    resize!(L.A, n)
    resize!(L.B, n)
    resize!(L.b, n)

    L
end

function Base.convert(::Type{AbstractMatrix}, L::AffineOperator)
    m, n = size(L)
    msg = """$L cannot be represented by an $m × $n AbstractMatrix"""
    throw(ArgumentError(msg))
end

has_adjoint(L::AffineOperator) = false
has_mul(L::AffineOperator) = has_mul(L.A)
has_mul!(L::AffineOperator) = has_mul!(L.A)
has_ldiv(L::AffineOperator) = has_ldiv(L.A)
has_ldiv!(L::AffineOperator) = has_ldiv!(L.A)

function cache_internals(L::AffineOperator, u::AbstractVecOrMat)
    @set! L.A = cache_operator(L.A, u)
    @set! L.B = cache_operator(L.B, L.b)
    L
end

# operator application
function Base.:*(L::AffineOperator, u::AbstractVecOrMat)
    @assert size(L.b, 2) == size(u, 2)
    (L.A * u) + (L.B * L.b)
end

function Base.:\(L::AffineOperator, u::AbstractVecOrMat)
    @assert size(L.b, 2) == size(u, 2)
    L.A \ (u - (L.B * L.b))
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::AffineOperator, u::AbstractVecOrMat)
    mul!(v, L.B, L.b)
    mul!(v, L.A, u, true, true)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::AffineOperator, u::AbstractVecOrMat, α, β)
    mul!(v, L.B, L.b, α, β)
    mul!(v, L.A, u, α, true)
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::AffineOperator, u::AbstractVecOrMat)
    copy!(v, u)
    ldiv!(L, v)
end

function LinearAlgebra.ldiv!(L::AffineOperator, u::AbstractVecOrMat)
    mul!(u, L.B, L.b, -1, 1)
    ldiv!(L.A, u)
end
#
