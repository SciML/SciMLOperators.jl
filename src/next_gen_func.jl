@concrete mutable struct FunctionOperatorV2{T <: Number} <: AbstractSciMLOperator{T}
    # Operators
    op
    # Traits: All of these are Vals
    traits
    # Prototype Arrays
    w_prototype
    v_prototype
    # Store things
    u
    p
    t
end

const NVType = Union{Nothing, Val}
const True = Val(true)
const False = Val(false)

const FOP_ELTYPE_MSG = """
The `eltype` of `FunctionOperator`, as well as the prototype arrays must be `<: Number`.
"""

FOP_NDIMS_MSG(w, v, u) = """
`w` / `v` / `u` arrays, ($(typeof(w)), $(typeof(v)), $(typeof(u))) provided to
`FunctionOperator` do not have the same number of dimensions. Further, if `batch = $(True)`,
then both arrays must be `AbstractVector`s, or both must be `AbstractMatrix` types.
"""

"""
$(SIGNATURES)

Wrap callable object `op` within an `AbstractSciMLOperator`. `op` is assumed to have
signature

    op(v, u, p, t) -> w
    op(w, v, u, p, t) -> [modifies w]
    op(w, v, u, p, t, α, β) -> [modifies w] (optional)

where `u`, `v`, `w` are `AbstractArray`s, `p` is a parameter object, and `t`, `α`, `β` are
scalars.

!!! note

    This is meant to be a future replacement for `FunctionOperator`, however, currently this
    is significantly limited in terms of functionality.

## Matrix Multiplication

1. `Base.:*(L::AbstractSciMLOperator, v) -> L.op(v, L.u, L.p, L.t)`
2. `mul!(w, L::AbstractSciMLOperator, v) -> L.op(w, v, L.u, L.p, L.t)`
3. `mul!(w, L::AbstractSciMLOperator, v, α, β) -> L.op(w, v, L.u, L.p, L.t, α, β)`

Unlike `FunctionOperator`, which automatically handles reshaping and sizechecks, this
operator simply forwards the arguments to `op`, with an expectation that `op` will handle
whatever checks are necessary.

## Keyword Arguments

- `T`: `eltype` of the operator. Defaults to `promote_type(eltype(v), eltype(u), eltype(w))`
- `p = nothing`: Parameter object
- `t = zero(T)`: Time Variable
- `isinplace = nothing`: Whether `op` has an in-place signature. Defaults to
  `Val(static_hasmethod(op, typeof((w, v, u, p, t))))`
- `outofplace = nothing`: Whether `op` has an out-of-place signature. Defaults to
  `Val(static_hasmethod(op, typeof((v, u, p, t))))`
- `has_mul5 = nothing`: Whether `op` has a signature with 5 arguments. Defaults to
  `Val(static_hasmethod(op, typeof((w, v, u, p, t, t, t))))`
"""
function FunctionOperatorV2(op::F, u::AbstractArray, v::AbstractArray = u,
        w::AbstractArray = u; p = nothing, t::Union{Number, Nothing} = nothing,
        T::Union{Type{<:Number}, Nothing} = nothing, isinplace::NVType = nothing,
        outofplace::NVType = nothing, has_mul5::NVType = nothing) where {F}
    T = ifelse(T === nothing, promote_type(eltype(v), eltype(u), eltype(w)), T)
    t = ifelse(t === nothing, zero(real(_T)), t)

    # Check Arguments
    T <: Number || throw(ArgumentError(FOP_ELTYPE_MSG))
    ndims(w) == ndims(v) == ndims(u) || throw(ArgumentError(FOP_NDIMS_MSG(w, v, u)))

    isinplace = ifelse(isinplace === nothing,
        Val(static_hasmethod(op, typeof((w, v, u, p, t)))), isinplace)
    outofplace = ifelse(outofplace === nothing,
        Val(static_hasmethod(op, typeof((v, u, p, t)))), outofplace)

    if isinplace === False && outofplace === False
        msg = "FunctionOperator requires either an in-place or out-of-place method!"
        throw(ArgumentError(msg))
    end

    has_mul5 = ifelse(has_mul5 === nothing,
        Val(static_hasmethod(op, typeof((w, v, u, p, t, t, t)))), has_mul5)

    # traits
    isreal = Val(T <: Real)
    traits = (; isreal, isinplace, outofplace, has_mul5)

    return FunctionOperatorV2{T}(op, traits, w, v, u, p, t)
end

# Matrix Multiplication
function Base.:*(L::FunctionOperatorV2, v::AbstractArray)
    @unpack isinplace, outofplace = L.traits
    if outofplace === True
        return L.op(v, L.u, L.p, L.t)
    else # Has Inplace Function
        w = similar(L.w_prototype)
        L.op(w, v, L.u, L.p, L.t)
        return w
    end
end

function LinearAlgebra.mul!(w::AbstractArray, L::FunctionOperatorV2, v::AbstractArray)
    @unpack isinplace, outofplace = L.traits
    if isinplace === True
        L.op(w, v, L.u, L.p, L.t)
    else # Has Outofplace Function
        copyto!(w, L.op(v, L.u, L.p, L.t))
    end
    return w
end

function LinearAlgebra.mul!(w::AbstractArray, L::FunctionOperatorV2, v::AbstractArray,
        α::Number, β::Number)
    @unpack isinplace, outofplace, has_mul5 = L.traits
    if isinplace === True
        if has_mul5 === True
            L.op(w, v, L.u, L.p, L.t, α, β)
        else
            if β == 0
                L.op(w, v, L.u, L.p, L.t)
                @. w *= α
            else
                w_ = similar(L.w_prototype)
                L.op(w_, v, L.u, L.p, L.t)
                @. w = α * w_ + β * w
            end
        end
    else # Has Outofplace Function
        if β == 0
            w_ = L.op(v, L.u, L.p, L.t)
            if α == 0
                copyto!(w, w_)
            else
                @. w = α * w_
            end
        else
            w_ = L.op(v, L.u, L.p, L.t)
            @. w = α * w_ + β * w
        end
    end
    return w
end

# Update Coefficients
function update_coefficients(L::FunctionOperatorV2{T}, u, p, t) where {T}
    op = update_coefficients(L.op, u, p, t)
    return FunctionOperatorV2{promote_type(T, eltype(u))}(op, L.traits, L.w_prototype,
        L.v_prototype, u, p, t)
end

function update_coefficients!(L::FunctionOperatorV2{T}, u, p, t) where {T}
    update_coefficients!(L.op, u, p, t)
    L.u = u
    L.p = p
    L.t = t
    return L
end