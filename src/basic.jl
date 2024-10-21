"""
$(TYPEDEF)

Operator representing the identity function `id(u) = u`
"""
struct IdentityOperator <: AbstractSciMLOperator{Bool}
    len::Int
end

# constructors
function Base.one(L::AbstractSciMLOperator)
    @assert issquare(L)
    N = size(L, 1)
    IdentityOperator(N)
end

Base.convert(::Type{AbstractMatrix}, ii::IdentityOperator) = Diagonal(ones(Bool, ii.len))

# traits
Base.show(io::IO, ii::IdentityOperator) = print(io, "IdentityOperator($(ii.len))")
Base.size(ii::IdentityOperator) = (ii.len, ii.len)
Base.adjoint(A::IdentityOperator) = A
Base.transpose(A::IdentityOperator) = A
Base.conj(A::IdentityOperator) = A

LinearAlgebra.opnorm(::IdentityOperator, p::Real = 2) = true
for pred in (:issymmetric, :ishermitian, :isposdef)
    @eval LinearAlgebra.$pred(::IdentityOperator) = true
end

getops(::IdentityOperator) = ()
isconstant(::IdentityOperator) = true
islinear(::IdentityOperator) = true
has_adjoint(::IdentityOperator) = true
has_mul!(::IdentityOperator) = true
has_ldiv(::IdentityOperator) = true
has_ldiv!(::IdentityOperator) = true

# operator application
for op in (:*, :\)
    @eval function Base.$op(ii::IdentityOperator, u::AbstractVecOrMat)
        @assert size(u, 1) == ii.len
        copy(u)
    end
end

@inline function LinearAlgebra.mul!(
        v::AbstractVecOrMat, ii::IdentityOperator, u::AbstractVecOrMat)
    @assert size(u, 1) == ii.len
    copy!(v, u)
end

@inline function LinearAlgebra.mul!(v::AbstractVecOrMat,
        ii::IdentityOperator,
        u::AbstractVecOrMat,
        α,
        β)
    @assert size(u, 1) == ii.len
    mul!(v, I, u, α, β)
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, ii::IdentityOperator, u::AbstractVecOrMat)
    @assert size(u, 1) == ii.len
    copy!(v, u)
end

function LinearAlgebra.ldiv!(ii::IdentityOperator, u::AbstractVecOrMat)
    @assert size(u, 1) == ii.len
    u
end

# operator fusion with identity returns operator itself
for op in (:*, :∘)
    @eval function Base.$op(ii::IdentityOperator, A::AbstractSciMLOperator)
        @assert size(A, 1) == ii.len
        A
    end

    @eval function Base.$op(A::AbstractSciMLOperator, ii::IdentityOperator)
        @assert size(A, 2) == ii.len
        A
    end

    @eval function Base.$op(ii1::IdentityOperator, ii2::IdentityOperator)
        @assert ii1.len == ii2.len
        IdentityOperator(ii1.len)
    end
end

function Base.:\(ii::IdentityOperator, A::AbstractSciMLOperator)
    @assert size(A, 1) == ii.len
    A
end

function Base.:/(A::AbstractSciMLOperator, ii::IdentityOperator)
    @assert size(A, 2) == ii.len
    A
end

"""
$(TYPEDEF)

Operator representing the null function `n(u) = 0 * u`
"""
struct NullOperator <: AbstractSciMLOperator{Bool}
    len::Int
end

# constructors
function Base.zero(L::AbstractSciMLOperator)
    @assert issquare(L)
    N = size(L, 1)
    NullOperator(N)
end

Base.convert(::Type{AbstractMatrix}, nn::NullOperator) = Diagonal(zeros(Bool, nn.len))

# traits
Base.show(io::IO, nn::NullOperator) = print(io, "NullOperator($(nn.len))")
Base.size(nn::NullOperator) = (nn.len, nn.len)
Base.adjoint(A::NullOperator) = A
Base.transpose(A::NullOperator) = A
Base.conj(A::NullOperator) = A
LinearAlgebra.opnorm(::NullOperator, p::Real = 2) = false
for pred in (:issymmetric, :ishermitian)
    @eval LinearAlgebra.$pred(::NullOperator) = true
end
LinearAlgebra.isposdef(::NullOperator) = false

getops(::NullOperator) = ()
isconstant(::NullOperator) = true
islinear(::NullOperator) = true
Base.iszero(::NullOperator) = true
has_adjoint(::NullOperator) = true
has_mul!(::NullOperator) = true

# operator application
Base.:*(nn::NullOperator, u::AbstractVecOrMat) = (@assert size(u, 1) == nn.len; zero(u))

function LinearAlgebra.mul!(v::AbstractVecOrMat, nn::NullOperator, u::AbstractVecOrMat)
    @assert size(u, 1) == size(v, 1) == nn.len
    lmul!(false, v)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat,
        nn::NullOperator,
        u::AbstractVecOrMat,
        α,
        β)
    @assert size(u, 1) == size(v, 1) == nn.len
    lmul!(β, v)
end

# operator fusion, composition
for op in (:*, :∘)
    @eval function Base.$op(nn::NullOperator, A::AbstractSciMLOperator)
        @assert size(A, 1) == nn.len
        NullOperator(nn.len)
    end

    @eval function Base.$op(A::AbstractSciMLOperator, nn::NullOperator)
        @assert size(A, 2) == nn.len
        NullOperator(nn.len)
    end

    @eval function Base.$op(nn1::NullOperator, nn2::NullOperator)
        @assert nn1.len == nn2.len
        NullOperator(nn1.len)
    end

    @eval function Base.$op(nn::NullOperator, ii::IdentityOperator)
        @assert nn.len == ii.len
        NullOperator(nn.len)
    end

    @eval function Base.$op(ii::IdentityOperator, nn::NullOperator)
        @assert nn.len == ii.len
        NullOperator(nn.len)
    end

    @eval function Base.$op(nn::NullOperator, ii::IdentityOperator)
        @assert nn.len == ii.len
        NullOperator(nn.len)
    end

    @eval function Base.$op(λ::AbstractSciMLScalarOperator, nn::NullOperator)
        NullOperator(nn.len)
    end

    @eval function Base.$op(nn::NullOperator, λ::AbstractSciMLScalarOperator)
        NullOperator(nn.len)
    end

end

# operator addition, subtraction with NullOperator returns operator itself
for op in (:+, :-)
    @eval function Base.$op(nn::NullOperator, A::AbstractSciMLOperator)
        @assert size(A) == (nn.len, nn.len)
        A
    end

    @eval function Base.$op(A::AbstractSciMLOperator, nn::NullOperator)
        @assert size(A) == (nn.len, nn.len)
        A
    end

    @eval function Base.$op(nn1::NullOperator, nn2::NullOperator)
        @assert nn1.len == nn2.len
        nn1
    end

end

"""
$TYPEDEF

    ScaledOperator

    (λ L)*(u) = λ * L(u)
"""
struct ScaledOperator{T,
    λType,
    LType
} <: AbstractSciMLOperator{T}
    λ::λType
    L::LType

    function ScaledOperator(λ::AbstractSciMLScalarOperator{Tλ},
            L::AbstractSciMLOperator{TL}) where {Tλ, TL}
        T = promote_type(Tλ, TL)
        new{T, typeof(λ), typeof(L)}(λ, L)
    end
end

# constructors
for T in SCALINGNUMBERTYPES[2:end]
    @eval ScaledOperator(λ::$T, L::AbstractSciMLOperator) = ScaledOperator(
        ScalarOperator(λ),
        L)
end

for T in SCALINGNUMBERTYPES
    @eval function ScaledOperator(λ::$T, L::ScaledOperator)
        λ = λ * L.λ
        ScaledOperator(λ, L.L)
    end

    for LT in SCALINGCOMBINETYPES
        @eval Base.:*(λ::$T, L::$LT) = ScaledOperator(λ, L)
        @eval Base.:*(L::$LT, λ::$T) = ScaledOperator(λ, L)

        @eval Base.:\(λ::$T, L::$LT) = ScaledOperator(inv(λ), L)
        @eval Base.:\(L::$LT, λ::$T) = ScaledOperator(λ, inv(L))

        @eval Base.:/(L::$LT, λ::$T) = ScaledOperator(inv(λ), L)
        @eval Base.:/(λ::$T, L::$LT) = ScaledOperator(λ, inv(L))
    end
end

Base.:-(L::AbstractSciMLOperator) = ScaledOperator(-true, L)
Base.:+(L::AbstractSciMLOperator) = L

function Base.convert(::Type{AbstractMatrix}, L::ScaledOperator)
    convert(Number, L.λ) * convert(AbstractMatrix, L.L)
end

# traits
function Base.show(io::IO, L::ScaledOperator{T}) where {T}
    show(io, L.λ)
    print(io, " * ")
    show(io, L.L)
end
Base.size(L::ScaledOperator) = size(L.L)
for op in (:adjoint,
    :transpose)
    @eval Base.$op(L::ScaledOperator) = ScaledOperator($op(L.λ), $op(L.L))
end
Base.conj(L::ScaledOperator) = conj(L.λ) * conj(L.L)
Base.resize!(L::ScaledOperator, n::Integer) = (resize!(L.L, n); L)
LinearAlgebra.opnorm(L::ScaledOperator, p::Real = 2) = abs(L.λ) * opnorm(L.L, p)

function update_coefficients(L::ScaledOperator, u, p, t)
    @reset L.L = update_coefficients(L.L, u, p, t)
    @reset L.λ = update_coefficients(L.λ, u, p, t)

    L
end

function update_coefficients!(L::ScaledOperator, u, p, t)
    update_coefficients!(L.L, u, p, t)
    update_coefficients!(L.λ, u, p, t)

    nothing
end

getops(L::ScaledOperator) = (L.λ, L.L)
isconstant(L::ScaledOperator) = isconstant(L.L) & isconstant(L.λ)
islinear(L::ScaledOperator) = islinear(L.L)
Base.iszero(L::ScaledOperator) = iszero(L.L) | iszero(L.λ)
has_adjoint(L::ScaledOperator) = has_adjoint(L.L)
has_mul(L::ScaledOperator) = has_mul(L.L)
has_mul!(L::ScaledOperator) = has_mul!(L.L)
has_ldiv(L::ScaledOperator) = has_ldiv(L.L) & !iszero(L.λ)
has_ldiv!(L::ScaledOperator) = has_ldiv!(L.L) & !iszero(L.λ)

function cache_internals(L::ScaledOperator, u::AbstractVecOrMat)
    @reset L.L = cache_operator(L.L, u)
    @reset L.λ = cache_operator(L.λ, u)
    L
end

# getindex
Base.getindex(L::ScaledOperator, i::Int) = L.coeff * L.L[i]
Base.getindex(L::ScaledOperator, I::Vararg{Int, N}) where {N} = L.λ * L.L[I...]

factorize(L::ScaledOperator) = L.λ * factorize(L.L)
for fact in (:lu, :lu!,
    :qr, :qr!,
    :cholesky, :cholesky!,
    :ldlt, :ldlt!,
    :bunchkaufman, :bunchkaufman!,
    :lq, :lq!,
    :svd, :svd!)
    @eval LinearAlgebra.$fact(L::ScaledOperator, args...) = L.λ * fact(L.L, args...)
end

# operator application, inversion
Base.:*(L::ScaledOperator, u::AbstractVecOrMat) = L.λ * (L.L * u)
Base.:\(L::ScaledOperator, u::AbstractVecOrMat) = L.λ \ (L.L \ u)

@inline function LinearAlgebra.mul!(
        v::AbstractVecOrMat, L::ScaledOperator, u::AbstractVecOrMat)
    iszero(L.λ) && return lmul!(false, v)
    a = convert(Number, L.λ)
    mul!(v, L.L, u, a, false)
end

@inline function LinearAlgebra.mul!(v::AbstractVecOrMat,
        L::ScaledOperator,
        u::AbstractVecOrMat,
        α,
        β)
    iszero(L.λ) && return lmul!(β, v)
    a = convert(Number, L.λ * α)
    mul!(v, L.L, u, a, β)
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::ScaledOperator, u::AbstractVecOrMat)
    ldiv!(v, L.L, u)
    ldiv!(L.λ, v)
end

function LinearAlgebra.ldiv!(L::ScaledOperator, u::AbstractVecOrMat)
    ldiv!(L.λ, u)
    ldiv!(L.L, u)
end

"""
Lazy operator addition

    (A1 + A2 + A3...)u = A1*u + A2*u + A3*u ....
"""
struct AddedOperator{T,
    O <: Tuple{Vararg{AbstractSciMLOperator}}
} <: AbstractSciMLOperator{T}
    ops::O

    function AddedOperator(ops)
        @assert !isempty(ops)
        _check_AddedOperator_sizes(ops)
        T = promote_type(eltype.(ops)...)
        new{T, typeof(ops)}(ops)
    end
end

function AddedOperator(ops::AbstractSciMLOperator...)
    AddedOperator(ops)
end

AddedOperator(L::AbstractSciMLOperator) = L

@generated function _check_AddedOperator_sizes(ops::Tuple)
    ops_types = ops.parameters
    N = length(ops_types)
    sz_expr_list = ()
    sz_expr = :(sz = size(first(ops)))
    for i in 2:N
        sz_expr_list = (sz_expr_list..., :(size(ops[$i]) == sz))
    end

    quote
        $sz_expr
        @assert all(tuple($(sz_expr_list...))) "Dimension mismatch: cannot add operators of different sizes."
        nothing
    end
end

# constructors
Base.:+(A::AbstractSciMLOperator, B::AbstractMatrix) = A + MatrixOperator(B)
Base.:+(A::AbstractMatrix, B::AbstractSciMLOperator) = MatrixOperator(A) + B

Base.:+(ops::AbstractSciMLOperator...) = reduce(+, ops)
Base.:+(A::AbstractSciMLOperator, B::AbstractSciMLOperator) = AddedOperator(A, B)
Base.:+(A::AbstractSciMLOperator, B::AddedOperator) = AddedOperator(A, B.ops...)
Base.:+(A::AddedOperator, B::AbstractSciMLOperator) = AddedOperator(A.ops..., B)
Base.:+(A::AddedOperator, B::AddedOperator) = AddedOperator(A.ops..., B.ops...)

function Base.:+(A::AddedOperator, Z::NullOperator)
    @assert size(A) == size(Z)
    A
end

function Base.:+(Z::NullOperator, A::AddedOperator)
    @assert size(A) == size(Z)
    A
end

Base.:-(A::AbstractSciMLOperator, B::AbstractSciMLOperator) = AddedOperator(A, -B)
Base.:-(A::AbstractSciMLOperator, B::AbstractMatrix) = A - MatrixOperator(B)
Base.:-(A::AbstractMatrix, B::AbstractSciMLOperator) = MatrixOperator(A) - B

for op in (:+, :-)
    for T in SCALINGNUMBERTYPES
        for LT in SCALINGCOMBINETYPES
            @eval function Base.$op(L::$LT, λ::$T)
                @assert issquare(L)
                iszero(λ) && return L
                N = size(L, 1)
                Id = IdentityOperator(N)
                AddedOperator(L, $op(λ) * Id)
            end

            @eval function Base.$op(λ::$T, L::$LT)
                @assert issquare(L)
                iszero(λ) && return $op(L)
                N = size(L, 1)
                Id = IdentityOperator(N)
                AddedOperator(λ * Id, $op(L))
            end
        end
    end
end

for T in SCALINGNUMBERTYPES[2:end]
    @eval function Base.:*(λ::$T, L::AddedOperator)
        ops = map(op -> λ * op, L.ops)
        AddedOperator(ops)
    end

    @eval function Base.:*(L::AddedOperator, λ::$T)
        ops = map(op -> λ * op, L.ops)
        AddedOperator(ops)
    end

    @eval function Base.:/(L::AddedOperator, λ::$T)
        ops = map(op -> op / λ, L.ops)
        AddedOperator(ops)
    end
end

function Base.convert(::Type{AbstractMatrix}, L::AddedOperator)
    sum(op -> convert(AbstractMatrix, op), L.ops)
end

# traits
function Base.show(io::IO, L::AddedOperator)
    print(io, "(")
    show(io, L.ops[1])
    for i in 2:length(L.ops)
        print(io, " + ")
        show(io, L.ops[i])
    end
    print(io, ")")
end
Base.size(L::AddedOperator) = size(first(L.ops))
for op in (:adjoint,
    :transpose)
    @eval Base.$op(L::AddedOperator) = AddedOperator($op.(L.ops)...)
end
Base.conj(L::AddedOperator) = AddedOperator(conj.(L.ops))
function Base.resize!(L::AddedOperator, n::Integer)
    for op in L.ops
        resize!(op, n)
    end
    L
end

function update_coefficients(L::AddedOperator, u, p, t)
    ops = ()
    for op in L.ops
        ops = (ops..., update_coefficients(op, u, p, t))
    end

    @reset L.ops = ops
end

@generated function update_coefficients!(L::AddedOperator, u, p, t)
    ops_types = L.parameters[2].parameters
    N = length(ops_types)
    quote
        Base.@nexprs $N i->begin
            update_coefficients!(L.ops[i], u, p, t)
        end

        nothing
    end
end

getops(L::AddedOperator) = L.ops
islinear(L::AddedOperator) = all(islinear, getops(L))
Base.iszero(L::AddedOperator) = all(iszero, getops(L))
has_adjoint(L::AddedOperator) = all(has_adjoint, L.ops)

@generated function cache_internals(L::AddedOperator, u::AbstractVecOrMat)
    ops_types = L.parameters[2].parameters
    N = length(ops_types)
    quote
        Base.@nexprs $N i->begin
            @reset L.ops[i] = cache_operator(L.ops[i], u)
        end
        L
    end
end

getindex(L::AddedOperator, i::Int) = sum(op -> op[i], L.ops)
getindex(L::AddedOperator, I::Vararg{Int, N}) where {N} = sum(op -> op[I...], L.ops)

function Base.:*(L::AddedOperator, u::AbstractVecOrMat)
    sum(op -> iszero(op) ? zero(u) : op * u, L.ops)
end

@generated function LinearAlgebra.mul!(
        v::AbstractVecOrMat, L::AddedOperator, u::AbstractVecOrMat)
    ops_types = L.parameters[2].parameters
    N = length(ops_types)
    quote
        mul!(v, L.ops[1], u)
        Base.@nexprs $(N - 1) i->begin
            mul!(v, L.ops[i + 1], u, true, true)
        end
        v
    end
end

@generated function LinearAlgebra.mul!(v::AbstractVecOrMat,
        L::AddedOperator,
        u::AbstractVecOrMat,
        α,
        β)
    ops_types = L.parameters[2].parameters
    N = length(ops_types)
    quote
        lmul!(β, v)
        Base.@nexprs $(N) i->begin
            mul!(v, L.ops[i], u, α, true)
        end
        v
    end
end

"""
    Lazy operator composition

    ∘(A, B, C)(u) = A(B(C(u)))

    ops = (A, B, C)
    cache = (B*C*u , C*u)
"""
struct ComposedOperator{T, O, C} <: AbstractSciMLOperator{T}
    """ Tuple of N operators to be applied in reverse"""
    ops::O
    """ cache for 3 and 5 argument mul! """
    cache::C

    function ComposedOperator(ops, cache)
        @assert !isempty(ops)
        for i in reverse(2:length(ops))
            opcurr = ops[i]
            opnext = ops[i - 1]
            @assert size(opcurr, 1)==size(opnext, 2) "Dimension mismatch: cannot compose
          operators of sizes $(size(opnext)), and $(size(opcurr))."
        end

        T = promote_type(eltype.(ops)...)
        new{T, typeof(ops), typeof(cache)}(ops, cache)
    end
end

function ComposedOperator(ops::AbstractSciMLOperator...; cache = nothing)
    ComposedOperator(ops, cache)
end

# constructors
for op in (:*, :∘)
    @eval Base.$op(ops::AbstractSciMLOperator...) = reduce($op, ops)
    @eval Base.$op(A::AbstractSciMLOperator, B::AbstractSciMLOperator) = ComposedOperator(
        A,
        B)
    @eval Base.$op(A::ComposedOperator, B::AbstractSciMLOperator) = ComposedOperator(
        A.ops...,
        B)
    @eval Base.$op(A::AbstractSciMLOperator, B::ComposedOperator) = ComposedOperator(A,
        B.ops...)
    @eval Base.$op(A::ComposedOperator, B::ComposedOperator) = ComposedOperator(A.ops...,
        B.ops...)
end

for op in (:*, :∘)
    # identity
    @eval function Base.$op(ii::IdentityOperator, A::ComposedOperator)
        @assert size(A, 1) == ii.len
        A
    end

    @eval function Base.$op(A::ComposedOperator, ii::IdentityOperator)
        @assert size(A, 2) == ii.len
        A
    end

    # null operator
    @eval function Base.$op(nn::NullOperator, A::ComposedOperator)
        @assert size(A, 1) == nn.len
        zero(A)
    end

    @eval function Base.$op(A::ComposedOperator, nn::NullOperator)
        @assert size(A, 2) == nn.len
        zero(A)
    end

    # scalar operator
    @eval function Base.$op(λ::AbstractSciMLScalarOperator, L::ComposedOperator)
        ScaledOperator(λ, L)
    end

    @eval function Base.$op(L::ComposedOperator, λ::AbstractSciMLScalarOperator)
        ScaledOperator(λ, L)
    end
end

function Base.convert(::Type{AbstractMatrix}, L::ComposedOperator)
    prod(op -> convert(AbstractMatrix, op), L.ops)
end

# traits
function Base.show(io::IO, L::ComposedOperator)
    print(io, "(")
    show(io, L.ops[1])
    for i in 2:length(L.ops)
        print(io, " * ")
        show(io, L.ops[i])
    end
    print(io, ")")
end
Base.size(L::ComposedOperator) = (size(first(L.ops), 1), size(last(L.ops), 2))
for op in (:adjoint,
    :transpose)
    @eval Base.$op(L::ComposedOperator) = ComposedOperator($op.(reverse(L.ops))...;
        cache = iscached(L) ? reverse(L.cache) : nothing)
end
Base.conj(L::ComposedOperator) = ComposedOperator(conj.(L.ops); cache = L.cache)
function Base.resize!(L::ComposedOperator, n::Integer)
    for op in L.ops
        resize!(op, n)
    end

    for v in L.cache
        resize!(v, n)
    end

    L
end

LinearAlgebra.opnorm(L::ComposedOperator) = prod(opnorm, L.ops)

function update_coefficients(L::ComposedOperator, u, p, t)
    ops = ()
    for op in L.ops
        ops = (ops..., update_coefficients(op, u, p, t))
    end

    @reset L.ops = ops
end

getops(L::ComposedOperator) = L.ops
islinear(L::ComposedOperator) = all(islinear, L.ops)
Base.iszero(L::ComposedOperator) = all(iszero, getops(L))
has_adjoint(L::ComposedOperator) = all(has_adjoint, L.ops)
has_mul(L::ComposedOperator) = all(has_mul, L.ops)
has_mul!(L::ComposedOperator) = all(has_mul!, L.ops)
has_ldiv(L::ComposedOperator) = all(has_ldiv, L.ops)
has_ldiv!(L::ComposedOperator) = all(has_ldiv!, L.ops)

factorize(L::ComposedOperator) = prod(factorize, L.ops)
for fact in (:lu, :lu!,
    :qr, :qr!,
    :cholesky, :cholesky!,
    :ldlt, :ldlt!,
    :bunchkaufman, :bunchkaufman!,
    :lq, :lq!,
    :svd, :svd!)
    @eval LinearAlgebra.$fact(L::ComposedOperator, args...) = prod(
        op -> $fact(op, args...),
        reverse(L.ops))
end

# operator application
# https://github.com/SciML/SciMLOperators.jl/pull/94
#Base.:*(L::ComposedOperator, u::AbstractVecOrMat) = foldl((acc, op) -> op * acc, reverse(L.ops); init=u)
#Base.:\(L::ComposedOperator, u::AbstractVecOrMat) = foldl((acc, op) -> op \ acc, L.ops; init=u)

function Base.:\(L::ComposedOperator, u::AbstractVecOrMat)
    v = u
    for op in L.ops
        v = op \ v
    end

    v
end

function Base.:*(L::ComposedOperator, u::AbstractVecOrMat)
    v = u
    for op in reverse(L.ops)
        v = op * v
    end

    v
end

function cache_self(L::ComposedOperator, u::AbstractVecOrMat)
    K = size(u, 2)
    cache = (zero(u),)

    for i in reverse(2:length(L.ops))
        op = L.ops[i]

        M = size(op, 1)
        sz = u isa AbstractMatrix ? (M, K) : (M,)

        T = if op isa FunctionOperator #
            # FunctionOperator isn't guaranteed to play by the rules of
            # `promote_type`. For example, an irFFT is a complex operation
            # that accepts complex vector and returns  ones.
            output_eltype(op)
        else
            promote_type(eltype.((op, cache[1]))...)
        end

        cache = (similar(u, T, sz), cache...)
    end

    @reset L.cache = cache
    L
end

function cache_internals(L::ComposedOperator, u::AbstractVecOrMat)
    if isnothing(L.cache)
        L = cache_self(L, u)
    end

    ops = ()
    for i in reverse(1:length(L.ops))
        ops = (cache_operator(L.ops[i], L.cache[i]), ops...)
    end

    @reset L.ops = ops
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::ComposedOperator, u::AbstractVecOrMat)
    @assert iscached(L) """cache needs to be set up for operator of type
    $L. Set up cache by calling `cache_operator(L, u)`"""

    vecs = (v, L.cache[1:(end - 1)]..., u)
    for i in reverse(1:length(L.ops))
        mul!(vecs[i], L.ops[i], vecs[i + 1])
    end
    v
end

function LinearAlgebra.mul!(v::AbstractVecOrMat,
        L::ComposedOperator,
        u::AbstractVecOrMat,
        α,
        β)
    @assert iscached(L) """cache needs to be set up for operator of type
    $L. Set up cache by calling `cache_operator(L, u)`."""

    cache = L.cache[end]
    copy!(cache, v)

    mul!(v, L, u)
    lmul!(α, v)
    axpy!(β, cache, v)
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::ComposedOperator, u::AbstractVecOrMat)
    @assert iscached(L) """cache needs to be set up for operator of type
    $L. Set up cache by calling `cache_operator(L, u)`."""

    vecs = (u, reverse(L.cache[1:(end - 1)])..., v)
    for i in 1:length(L.ops)
        ldiv!(vecs[i + 1], L.ops[i], vecs[i])
    end
    v
end

function LinearAlgebra.ldiv!(L::ComposedOperator, u::AbstractVecOrMat)
    for i in 1:length(L.ops)
        ldiv!(L.ops[i], u)
    end
    u
end

"""
    Lazy Operator Inverse
"""
struct InvertedOperator{T, LType, C} <: AbstractSciMLOperator{T}
    L::LType
    cache::C

    function InvertedOperator(L::AbstractSciMLOperator{T}, cache) where {T}
        new{T, typeof(L), typeof(cache)}(L, cache)
    end
end

function InvertedOperator(L::AbstractSciMLOperator{T}; cache = nothing) where {T}
    InvertedOperator(L, cache)
end

function InvertedOperator(A::AbstractMatrix{T}; cache = nothing) where {T}
    InvertedOperator(MatrixOperator(A), cache)
end

Base.inv(L::AbstractSciMLOperator) = InvertedOperator(L)

Base.:\(A::AbstractSciMLOperator, B::AbstractSciMLOperator) = inv(A) * B
Base.:/(A::AbstractSciMLOperator, B::AbstractSciMLOperator) = A * inv(B)

function Base.convert(::Type{AbstractMatrix}, L::InvertedOperator)
    inv(convert(AbstractMatrix, L.L))
end

function Base.show(io::IO, L::InvertedOperator)
    print(io, "1 / ")
    show(io, L.L)
end
Base.size(L::InvertedOperator) = size(L.L) |> reverse
function Base.transpose(L::InvertedOperator)
    InvertedOperator(transpose(L.L); cache = iscached(L) ? L.cache' : nothing)
end
function Base.adjoint(L::InvertedOperator)
    InvertedOperator(adjoint(L.L); cache = iscached(L) ? L.cache' : nothing)
end
Base.conj(L::InvertedOperator) = InvertedOperator(conj(L.L); cache = L.cache)
function Base.resize!(L::InvertedOperator, n::Integer)
    resize!(L.L, n)
    resize!(L.cache, n)

    L
end

function update_coefficients(L::InvertedOperator, u, p, t)
    @reset L.L = update_coefficients(L.L, u, p, t)

    L
end

getops(L::InvertedOperator) = (L.L,)
islinear(L::InvertedOperator) = islinear(L.L)
isconvertible(::InvertedOperator) = false

has_mul(L::InvertedOperator) = has_ldiv(L.L)
has_mul!(L::InvertedOperator) = has_ldiv!(L.L)
has_ldiv(L::InvertedOperator) = has_mul(L.L)
has_ldiv!(L::InvertedOperator) = has_mul!(L.L)

@forward InvertedOperator.L (
    # LinearAlgebra
    LinearAlgebra.issymmetric,
    LinearAlgebra.ishermitian,
    LinearAlgebra.isposdef,
    LinearAlgebra.opnorm,

    # SciML
    isconstant,
    has_adjoint)

Base.:*(L::InvertedOperator, u::AbstractVecOrMat) = L.L \ u
Base.:\(L::InvertedOperator, u::AbstractVecOrMat) = L.L * u

function cache_self(L::InvertedOperator, u::AbstractVecOrMat)
    cache = zero(u)
    @reset L.cache = cache
    L
end

function cache_internals(L::InvertedOperator, u::AbstractVecOrMat)
    @reset L.L = cache_operator(L.L, u)
    L
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::InvertedOperator, u::AbstractVecOrMat)
    ldiv!(v, L.L, u)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat,
        L::InvertedOperator,
        u::AbstractVecOrMat,
        α,
        β)
    @assert iscached(L) """cache needs to be set up for operator of type
    $L. Set up cache by calling `cache_operator(L, u)`."""

    copy!(L.cache, v)
    ldiv!(v, L.L, u)
    lmul!(α, v)
    axpy!(β, L.cache, v)
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::InvertedOperator, u::AbstractVecOrMat)
    mul!(v, L.L, u)
end

function LinearAlgebra.ldiv!(L::InvertedOperator, u::AbstractVecOrMat)
    @assert iscached(L) """cache needs to be set up for operator of type
    $L. Set up cache by calling `cache_operator(L, u)`."""

    copy!(L.cache, u)
    mul!(u, L.L, L.cache)
end
#
