#
"""
Matrix free operator given by a function

$(FIELDS)
"""
mutable struct FunctionOperator{iip,oop,mul5,T<:Number,F,Fa,Fi,Fai,Tr,P,Tt,C} <: AbstractSciMLOperator{T}
    """ Function with signature op(u, p, t) and (if isinplace) op(v, u, p, t) """
    op::F
    """ Adjoint operator"""
    op_adjoint::Fa
    """ Inverse operator"""
    op_inverse::Fi
    """ Adjoint inverse operator"""
    op_adjoint_inverse::Fai
    """ Traits """
    traits::Tr
    """ Parameters """
    p::P
    """ Time """
    t::Tt
    """ Cache """
    cache::C

    function FunctionOperator(
                              op,
                              op_adjoint,
                              op_inverse,
                              op_adjoint_inverse,
                              traits,
                              p,
                              t,
                              cache
                             )

        iip = traits.isinplace
        oop = traits.outofplace
        mul5 = traits.has_mul5
        T   = traits.T

        new{
            iip,
            oop,
            mul5,
            T,
            typeof(op),
            typeof(op_adjoint),
            typeof(op_inverse),
            typeof(op_adjoint_inverse),
            typeof(traits),
            typeof(p),
            typeof(t),
            typeof(cache),
           }(
             op,
             op_adjoint,
             op_inverse,
             op_adjoint_inverse,
             traits,
             p,
             t,
             cache,
            )
    end
end

"""
$(SIGNATURES)

Wrap callable object `op` within an `AbstractSciMLOperator`. `op`
is assumed to have signature

    op(u, p, t; <accepted_kwargs>) -> v

or

    op(v, u, p, t; <accepted_kwargs>) -> [modifies v]

and optionally

    op(v, u, p, t, α, β; <accepted_kwargs>) -> [modifies v]

where `u`, `v` are `AbstractArray`s, `p` is a parameter object, and
`t`, `α`, `β` are scalars. The first signautre corresponds to applying
the operator with `Base.*`, and the latter two correspond to the
three-argument, and the five-argument `mul!` respectively.

`input` and `output` prototype `AbstractArray`s are required for
determining operator traits such as `eltype`, `size`, and for
preallocating cache. If `output` array is not provided, the output
is assumed to be of the same type and share as the input.

# Keyword Arguments

Keyword arguments are used to pass in the adjoint evaluation function,
`op_adjoint`, the inverse function, `op_inverse`, and the adjoint-inverse
function `adjoint_inverse`. All are assumed to have the same calling signature and
below traits.

## Traits

Keyword arguments are used to set operator traits, which are assumed to be
uniform across `op`, `op_adjoint`, `op_inverse`, `op_adjoint_inverse`.

* `p` - Prototype of parameter struct passed to the operator during evaluation, i.e. `L(u, p, t)`. `p` is set to `nothing` if no value is provided.
* `t` - Protype of scalar time variable passed to the operator during evaluation. `t` is set to `zero(T)` if no value is provided.
* `accepted_kwargs` - `Tuple` of `Symbol`s corresponding to the keyword arguments accepted by `op*`, and `update_coefficients[!]`. For example, if `op` accepts kwarg `scale`, as in `op(u, p, t; scale)`, then `accepted_kwargs = (:scale,)`.
* `T` - `eltype` of the operator. If no value is provided, the constructor inferrs the value from types of `input`, and `output`
* `isinplace` - `true` if the operator can be used is a mutating way with in-place allocations. This trait is inferred if no value is provided.
* `outofplace` - `true` if the operator can be used is a non-mutating way with in-place allocations. This trait is inferred if no value is provided.
* `has_mul5` - `true` if the operator provides a five-argument `mul!` via the signature `op(v, u, p, t, α, β; <accepted_kwargs>)`. This trait is inferred if no value is provided.
* `isconstant` - `true` if the operator is constant, and doesn't need to be updated via `update_coefficients[!]` during operator evaluation.
* `islinear` - `true` if the operator is linear. Defaults to `false`.
* `isconvertible` - `true` a cheap `convert(AbstractMatrix, L.op)` method is available. Defaults to `false`.
* `batch` - Boolean indicating if the input/output arrays comprise of batched column-vectors stacked in a matrix. If `true`, the input/output arrays must be `AbstractVecOrMat`s, and the length of the  second dimension (the batch dimension) must be the same. The batch dimension is not involved in size computation. For example, with `batch = true`, and `size(output), size(input) = (M, K), (N, K)`, the `FunctionOperator` size is set to `(M, N)`. If `batch = false`, which is the default, the `input`/`output` arrays may of any size so long as `ndims(input) == ndims(output)`, and the `size` of `FunctionOperator` is set to `(length(input), length(output))`.
* `ifcache` - Allocate cache arrays in constructor. Defaults to `true`. Cache can be generated afterwards by calling `cache_operator(L, input, output)`
* `cache` - Pregenerated cache arrays for in-place evaluations. Expected to be of type and shape `(similar(input), similar(output),)`. The constructor generates cache if no values are provided. Cache generation by the constructor can be disabled by setting the kwarg `ifcache = false`.
* `opnorm` - The norm of `op`. Can be a `Number`, or function `opnorm(p::Integer)`. Defaults to `nothing`.
* `issymmetric` - `true` if the operator is linear and symmetric. Defaults to `false`.
* `ishermitian` - `true` if the operator is linear and hermitian. Defaults to `false`.
* `isposdef` - `true` if the operator is linear and positive-definite. Defaults to `false`.
"""
function FunctionOperator(op,
                          input::AbstractArray,
                          output::AbstractArray = input;

                          op_adjoint=nothing,
                          op_inverse=nothing,
                          op_adjoint_inverse=nothing,

                          p=nothing,
                          t::Union{Number,Nothing}=nothing,
                          accepted_kwargs::NTuple{N,Symbol} = (),

                          # traits
                          T::Union{Type{<:Number},Nothing}=nothing,
                          isinplace::Union{Nothing,Bool}=nothing,
                          outofplace::Union{Nothing,Bool}=nothing,
                          has_mul5::Union{Nothing,Bool}=nothing,
                          isconstant::Bool = false,
                          islinear::Bool = false,
                          isconvertible::Bool = false,

                          batch::Bool = false,
                          ifcache::Bool = true,
                          cache::Union{Nothing, NTuple{2}}=nothing,

                          # LinearAlgebra traits
                          opnorm = nothing,
                          issymmetric::Bool = false,
                          ishermitian::Bool = false,
                          isposdef::Bool = false,
                         ) where{N}

    # establish types

    # store eltype of input/output for caching with ComposedOperator.
    eltypes = eltype(input), eltype(output)
    T  = isnothing(T) ? promote_type(eltypes...) : T
    t  = isnothing(t) ? zero(real(T)) : t

    @assert T <: Number """The `eltype` of `FunctionOperator`, as well as
    the `input`/`output` arrays must be `<:Number`."""

    # establish sizes

    @assert ndims(output) == ndims(input) """`input`/`output` arrays,
    ($(typeof(input)), $(typeof(output))) provided to `FunctionOperator`
    do not have the same number of dimensions. Further, if `batch = true`,
    then both arrays must be `AbstractVector`s, or both must be
    `AbstractMatrix` types."""

    if batch
        if !isa(input, AbstractVecOrMat)
            msg = """`FunctionOperator` constructed with `batch = true` only
                accepts `AbstractVecOrMat` types with
                `size(L, 2) == size(u, 1)`."""
            ArgumentError(msg) |> throw
        end

        if input isa AbstractMatrix
            # assume batches are 2nd dimension of `AbstractVecOrMat`
            if size(input, 2) != size(output, 2)
                msg = """ Batch size (length of second dimension) in `input`/
                    `output` arrays to `FunctionOperator` is not equal. Input
                    array, $(typeof(input)), has size $(size(input)), whereas
                    output array, $(typeof(output)), has size
                    $(size(output))."""
                ArgumentError(msg) |> throw
            end
        end
    end

    sizes = size(input), size(output)

    _size = if batch
        # assume batches are 2nd dimension of `AbstractVecOrMat`
        (size(output, 1), size(input, 1))
    else
        (length(output), length(input))
    end

    # evaluation signatures

    isinplace = if isnothing(isinplace)
        static_hasmethod(op, typeof((output, input, p, t)))
    else
        isinplace
    end

    outofplace = if isnothing(outofplace)
        static_hasmethod(op, typeof((input, p, t)))
    else
        outofplace
    end

    if !isinplace & !outofplace
        @error """Please provide a funciton with signatures `op(u, p, t)` for
        applying the operator out-of-place, and/or the signature is
        `op(v, u, p, t)` for in-place application."""
    end

    has_mul5 = if isnothing(has_mul5)
        has_mul5 = true
        for f in (
                  op, op_adjoint, op_inverse, op_adjoint_inverse,
                 )
            if !isnothing(f)
                has_mul5 *= static_hasmethod(f, typeof((output, input, p, t, t, t)))
            end
        end

        has_mul5
    end

    # traits

    isreal = T <: Real
    selfadjoint = ishermitian | (isreal & issymmetric)
    adjointable = !(op_adjoint isa Nothing) | selfadjoint
    invertible  = !(op_inverse isa Nothing)

    if selfadjoint & (op_adjoint isa Nothing)
        op_adjoint = op
    end

    if selfadjoint & invertible & (op_adjoint_inverse isa Nothing)
        op_adjoint_inverse = op_inverse
    end

    traits = (;
              islinear = islinear,
              isconvertible = isconvertible,
              isconstant = isconstant,

              opnorm = opnorm,
              issymmetric = issymmetric,
              ishermitian = ishermitian,
              isposdef = isposdef,

              isinplace = isinplace,
              outofplace = outofplace,
              has_mul5 = has_mul5,
              ifcache = ifcache,
              T = T,
              batch = batch,
              size = _size,
              sizes = sizes,
              eltypes = eltypes,
              accepted_kwargs = accepted_kwargs,
              kwargs = Dict{Symbol, Any}(),
             )

    L = FunctionOperator(
                         op,
                         op_adjoint,
                         op_inverse,
                         op_adjoint_inverse,
                         traits,
                         p,
                         t,
                         cache
                        )

    # create cache

    if ifcache & isnothing(L.cache)
        L = cache_operator(L, input)
    end

    L
end

function update_coefficients(L::FunctionOperator, u, p, t; kwargs...)

    # update p, t
    @set! L.p = p
    @set! L.t = t

    # filter and update kwargs
    filtered_kwargs = get_filtered_kwargs(kwargs, L.traits.accepted_kwargs)
    @set! L.traits.kwargs = Dict{Symbol, Any}(filtered_kwargs)

    isconstant(L) && return L

    @set! L.op = update_coefficients(L.op, u, p, t; filtered_kwargs...)
    @set! L.op_adjoint = update_coefficients(L.op_adjoint, u, p, t; filtered_kwargs...)
    @set! L.op_inverse = update_coefficients(L.op_inverse, u, p, t; filtered_kwargs...)
    @set! L.op_adjoint_inverse = update_coefficients(L.op_adjoint_inverse, u, p, t; filtered_kwargs...)
end

function update_coefficients!(L::FunctionOperator, u, p, t; kwargs...)

    # update p, t
    L.p = p
    L.t = t

    # filter and update kwargs
    filtered_kwargs = get_filtered_kwargs(kwargs, L.traits.accepted_kwargs)
    L.traits = (; L.traits..., kwargs = Dict{Symbol, Any}(filtered_kwargs))

    isconstant(L) && return

    for op in getops(L)
        update_coefficients!(op, u, p, t; filtered_kwargs...)
    end

    L
end

function iscached(L::FunctionOperator)
    L.traits.ifcache ? !isnothing(L.cache) : !L.traits.ifcache
    !isnothing(L.cache)
end

# fix method amg bw AbstractArray, AbstractVecOrMat
cache_operator(L::FunctionOperator, u::AbstractArray) = _cache_operator(L, u)
cache_operator(L::FunctionOperator, u::AbstractVecOrMat) = _cache_operator(L, u)

function _cache_operator(L::FunctionOperator, u::AbstractArray)

    U = if L.traits.batch
        if !isa(u, AbstractVecOrMat)
            msg = """$L constructed with `batch = true` only accepts
                `AbstractVecOrMat` types with `size(L, 2) == size(u, 1)`."""
            ArgumentError(msg) |> throw
        end

        if size(L, 2) != size(u, 1)
            msg = """Second dimension of $L of size $(size(L))
                is not consistent with first dimension of input array `u`
                of size $(size(u))."""
            DimensionMismatch(msg) |> throw
        end

        M = size(L, 1)
        K = size(u, 2)
        size_out = u isa AbstractVector ? (M,) : (M, K)
        @set! L.traits.sizes = size(u), size_out

        u
    else
        if size(L, 2) != length(u)
            msg = """Length of input array, $(typeof(u)), of size $(size(u))
                not consistent with second dimension of $L of size
                $(size(L))."""
            throw(DimensionMismatch(msg))
        end

        reshape(u, L.traits.sizes[1])
    end

    L = cache_self(L, U)
    L = cache_internals(L, U)
    L
end

# fix method amg bw AbstractArray, AbstractVecOrMat
cache_self(L::FunctionOperator, u::AbstractArray) = _cache_self(L, u)
cache_self(L::FunctionOperator, u::AbstractVecOrMat) = _cache_self(L, u)

function _cache_self(L::FunctionOperator, u::AbstractArray)

    _u = similar(u, L.traits.eltypes[1], L.traits.sizes[1])
    _v = similar(u, L.traits.eltypes[2], L.traits.sizes[2])

    @set! L.cache = (_u, _v)
end

# fix method amg bw AbstractArray, AbstractVecOrMat
cache_internals(L::FunctionOperator, u::AbstractArray) = _cache_internals(L, u)
cache_internals(L::FunctionOperator, u::AbstractVecOrMat) = _cache_internals(L, u)

function _cache_internals(L::FunctionOperator, u::AbstractArray)

    @set! L.op = cache_operator(L.op, u)
    @set! L.op_adjoint = cache_operator(L.op_adjoint, u)
    @set! L.op_inverse = cache_operator(L.op_inverse, u)
    @set! L.op_adjoint_inverse = cache_operator(L.op_adjoint_inverse, u)
end

function Base.show(io::IO, L::FunctionOperator)
    M, N = size(L)
    print(io, "FunctionOperator($M × $N)")
end
Base.size(L::FunctionOperator) = L.traits.size
function Base.adjoint(L::FunctionOperator)

    if ishermitian(L) | (isreal(L) & issymmetric(L))
        return L
    end

    if !(has_adjoint(L))
        return AdjointOperator(L)
    end

    op = L.op_adjoint
    op_adjoint = L.op

    op_inverse = L.op_adjoint_inverse
    op_adjoint_inverse = L.op_inverse

    traits = L.traits
    @set! traits.size = reverse(size(L))
    @set! traits.sizes = reverse(traits.sizes)
    @set! traits.eltypes = reverse(traits.eltypes)

    cache = if iscached(L)
        cache = reverse(L.cache)
    else
        nothing
    end

    FunctionOperator(op,
                     op_adjoint,
                     op_inverse,
                     op_adjoint_inverse,
                     traits,
                     L.p,
                     L.t,
                     cache,
                    )
end

function Base.inv(L::FunctionOperator)
    if !(has_ldiv(L))
        return InvertedOperator(L)
    end

    op = L.op_inverse
    op_inverse = L.op

    op_adjoint = L.op_adjoint_inverse
    op_adjoint_inverse = L.op_adjoint

    traits = L.traits
    @set! traits.size = reverse(size(L))
    @set! traits.sizes = reverse(traits.sizes)
    @set! traits.eltypes = reverse(traits.eltypes)

    @set! traits.opnorm = if traits.opnorm isa Number
        1 / traits.opnorm
    elseif traits.opnorm isa Nothing
        nothing
    else
        (p::Real) -> 1 / traits.opnorm(p)
    end

    cache = if iscached(L)
        cache = reverse(L.cache)
    else
        nothing
    end

    FunctionOperator(op,
                     op_adjoint,
                     op_inverse,
                     op_adjoint_inverse,
                     traits,
                     L.p,
                     L.t,
                     cache,
                    )
end

Base.convert(::Type{AbstractMatrix}, L::FunctionOperator) = convert(AbstractMatrix, L.op)

function Base.resize!(L::FunctionOperator, n::Integer)

    # input/output to `L` must be `AbstractVector`s
    if length(L.traits.sizes[1]) != 1
        msg = """`Base.resize!` is only supported by $L whose input/output
            arrays are `AbstractVector`s."""
        MethodError(msg) |> throw
    end

    for op in getops(L)
        if static_hasmethod(resize!, typeof((op, n)))
            resize!(op, n)
        end
    end

    for v in L.cache
        resize!(v, n)
    end

    L.traits = (; L.traits..., size = (n, n), sizes = ((n,), (n,)),)

    L
end

function LinearAlgebra.opnorm(L::FunctionOperator, p)
    L.traits.opnorm === nothing && error("""
      M.opnorm is nothing, please define opnorm as a function that takes one
      argument. E.g., `(p::Real) -> p == Inf ? 100 : error("only Inf norm is
      defined")`""")
    opn = L.traits.opnorm
    return opn isa Number ? opn : L.traits.opnorm(p)
end
LinearAlgebra.issymmetric(L::FunctionOperator) = L.traits.issymmetric
LinearAlgebra.ishermitian(L::FunctionOperator) = L.traits.ishermitian
LinearAlgebra.isposdef(L::FunctionOperator) = L.traits.isposdef

function getops(L::FunctionOperator)
    (;
     op = L.op,
     op_adjoint = L.op_adjoint,
     op_inverse = L.op_inverse,
     op_adjoint_inverse = L.op_adjoint_inverse,
    )
end

islinear(L::FunctionOperator) = L.traits.islinear
isconvertible(L::FunctionOperator) = L.traits.isconvertible
isconstant(L::FunctionOperator) = L.traits.isconstant
has_adjoint(L::FunctionOperator) = !(L.op_adjoint isa Nothing)
has_mul(::FunctionOperator{iip}) where{iip} = true
has_mul!(::FunctionOperator{iip}) where{iip} = iip
has_ldiv(L::FunctionOperator{iip}) where{iip} = !(L.op_inverse isa Nothing)
has_ldiv!(L::FunctionOperator{iip}) where{iip} = iip & !(L.op_inverse isa Nothing)

function _sizecheck(L::FunctionOperator, u, v)

    if L.traits.batch
        if !isnothing(u)
            if !isa(u, AbstractVecOrMat)
                msg = """$L constructed with `batch = true` only
                    accept input arrays that are `AbstractVecOrMat`s with
                    `size(L, 2) == size(u, 1)`."""
                ArgumentError(msg) |> throw
            end

            if size(u) != L.traits.sizes[1]
                msg = """$L expects input arrays of size $(L.traits.sizes[1]).
                    Recievd array of size $(size(u))."""
                DimensionMismatch(msg) |> throw
            end
        end

        if !isnothing(v)
            if size(v) != L.traits.sizes[2]
                msg = """$L expects input arrays of size $(L.traits.sizes[1]).
                    Recievd array of size $(size(v))."""
                DimensionMismatch(msg) |> throw
            end
        end
    else # !batch
        if !isnothing(u)
            if size(u) != L.traits.sizes[1]
                msg = """$L expects input arrays of size $(L.traits.sizes[1]).
                    Recievd array of size $(size(u))."""
                DimensionMismatch(msg) |> throw
            end
        end

        if !isnothing(v)
            if size(v) != L.traits.sizes[2]
                msg = """$L expects input arrays of size $(L.traits.sizes[1]).
                    Recievd array of size $(size(v))."""
                DimensionMismatch(msg) |> throw
            end
        end
    end # batch

    return
end

# operator application
function Base.:*(L::FunctionOperator{iip,true}, u::AbstractArray) where{iip}
    _sizecheck(L, u, nothing)

    L.op(u, L.p, L.t; L.traits.kwargs...)
end

function Base.:\(L::FunctionOperator{iip,true}, u::AbstractArray) where{iip}
    _sizecheck(L, nothing, u)

    L.op_inverse(u, L.p, L.t; L.traits.kwargs...)
end

# fallback *, \ for FunctionOperator with no OOP method

function Base.:*(L::FunctionOperator{true,false}, u::AbstractArray)
    _, co = L.cache
    v = zero(co)

    _sizecheck(L, u, v)

    L.op(v, u, L.p, L.t; L.traits.kwargs...)
end

function Base.:\(L::FunctionOperator{true,false}, u::AbstractArray)
    ci, _ = L.cache
    v = zero(ci)

    _sizecheck(L, v, u)

    L.op_inverse(v, u, L.p, L.t; L.traits.kwargs...)
end

function LinearAlgebra.mul!(v::AbstractArray, L::FunctionOperator{true}, u::AbstractArray)

    _sizecheck(L, u, v)

    L.op(v, u, L.p, L.t; L.traits.kwargs...)
end

function LinearAlgebra.mul!(v::AbstractArray, L::FunctionOperator{false}, u::AbstractArray, args...)
    @error "LinearAlgebra.mul! not defined for out-of-place FunctionOperators"
end

function LinearAlgebra.mul!(v::AbstractArray, L::FunctionOperator{true, oop, false}, u::AbstractArray, α, β) where{oop}
    _, co = L.cache

    _sizecheck(L, u, v)

    copy!(co, v)
    mul!(v, L, u)
    axpby!(β, co, α, v)
end

function LinearAlgebra.mul!(v::AbstractArray, L::FunctionOperator{true, oop, true}, u::AbstractArray, α, β) where{oop}

    _sizecheck(L, u, v)

    L.op(v, u, L.p, L.t, α, β; L.traits.kwargs...)
end

function LinearAlgebra.ldiv!(v::AbstractArray, L::FunctionOperator{true}, u::AbstractArray)

    _sizecheck(L, v, u)

    L.op_inverse(v, u, L.p, L.t; L.traits.kwargs...)
end

function LinearAlgebra.ldiv!(L::FunctionOperator{true}, u::AbstractArray)
    ci, _ = L.cache

    copy!(ci, u)
    ldiv!(u, L, ci)
end

function LinearAlgebra.ldiv!(v::AbstractArray, L::FunctionOperator{false}, u::AbstractArray)
    @error "LinearAlgebra.ldiv! not defined for out-of-place FunctionOperators"
end

function LinearAlgebra.ldiv!(L::FunctionOperator{false}, u::AbstractArray)
    @error "LinearAlgebra.ldiv! not defined for out-of-place FunctionOperators"
end
#
