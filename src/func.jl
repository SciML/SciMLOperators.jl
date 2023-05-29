#
"""
    Matrix free operators (given by a function)
"""
mutable struct FunctionOperator{iip,oop,mul5,T<:Number,F,Fa,Fi,Fai,Tr,P,Tt,C} <: AbstractSciMLOperator{T}
    """ Function with signature op(u, p, t) and (if isinplace) op(du, u, p, t) """
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

function FunctionOperator(op,
                          input::AbstractArray{<:Any,D},
                          output::AbstractArray{<:Any,D};
                          kwargs...) where{D}
    D ≤ 2 && @error "FunctionOperator not defined for $(typeof(input)), $(typeof(output))."

    NK = length(input)
    MK = length(output)

    M = size(output, 1)
    N = size(input, 1)

    K  = NK ÷ N

    @assert MK == M * K

    input  = reshape(input,  (N, K))
    output = reshape(output, (M, K))

    FunctionOperator(op, input, output; kwargs...)
end

# TODO: document constructor and revisit design as needed (e.g. for "accepted_kwargs")
function FunctionOperator(op,
                          input::AbstractVecOrMat,
                          output::AbstractVecOrMat = input;

                          isinplace::Union{Nothing,Bool}=nothing,
                          outofplace::Union{Nothing,Bool}=nothing,
                          isconstant::Bool = false,
                          has_mul5::Union{Nothing,Bool}=nothing,
                          cache::Union{Nothing, NTuple{2}}=nothing,
                          T::Union{Type{<:Number},Nothing}=nothing,

                          op_adjoint=nothing,
                          op_inverse=nothing,
                          op_adjoint_inverse=nothing,

                          p=nothing,
                          t::Union{Number,Nothing}=nothing,
                          accepted_kwargs::NTuple{N,Symbol} = (),

                          ifcache::Bool = true,

                          # traits
                          islinear::Bool = false,

                          opnorm = nothing,
                          issymmetric::Bool = false,
                          ishermitian::Bool = false,
                          isposdef::Bool = false,
                         ) where{N}

    # store eltype of input/output for caching with ComposedOperator.
    eltypes = eltype.((input, output))
    sz = (size(output, 1), size(input, 1))
    T  = isnothing(T) ? promote_type(eltypes...) : T
    t  = isnothing(t) ? zero(real(T)) : t

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

    if !isinplace & !outofplace
        @error "Please provide a funciton with signatures `op(u, p, t)` for applying
        the operator out-of-place, and/or the signature is `op(du, u, p, t)` for
        in-place application."
    end

    T isa Nothing && @error "Please provide a Number type for the Operator"

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
              size = sz,
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

    if ifcache & isnothing(L.cache)
        L = cache_operator(L, input, output)
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

function cache_self(L::FunctionOperator, u::AbstractVecOrMat, v::AbstractVecOrMat)
    !L.traits.ifcache && @warn """Cache is being allocated for a FunctionOperator
        created with kwarg ifcache = false."""
    @set! L.cache = zero.((u, v))
    L
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

function Base.resize!(L::FunctionOperator, n::Integer)

    for op in getops(L)
        if static_hasmethod(resize!, typeof((op, n)))
            resize!(op, n)
        end
    end

    for v in L.cache
        resize!(v, n)
    end

    L.traits = (; L.traits..., size = (n, n),)

    L
end

function LinearAlgebra.opnorm(L::FunctionOperator, p)
    L.traits.opnorm === nothing && error("""
      M.opnorm is nothing, please define opnorm as a function that takes one
      argument. E.g., `(p::Real) -> p == Inf ? 100 : error("only Inf norm is
      defined")`
    """)
    opn = L.traits.opnorm
    return opn isa Number ? opn : L.traits.opnorm(p)
end
LinearAlgebra.issymmetric(L::FunctionOperator) = L.traits.issymmetric
LinearAlgebra.ishermitian(L::FunctionOperator) = L.traits.ishermitian
LinearAlgebra.isposdef(L::FunctionOperator) = L.traits.isposdef

function getops(L::FunctionOperator)
    ops = (L.op,)

    ops = isa(L.op_adjoint, Nothing) ? ops : (ops..., L.op_adjoint)
    ops = isa(L.op_inverse, Nothing) ? ops : (ops..., L.op_inverse)
    ops = isa(L.op_adjoint_inverse, Nothing) ? ops : (ops..., L.op_adjoint_inverse)

    ops
end

islinear(L::FunctionOperator) = L.traits.islinear
isconstant(L::FunctionOperator) = L.traits.isconstant
has_adjoint(L::FunctionOperator) = !(L.op_adjoint isa Nothing)
has_mul(::FunctionOperator{iip}) where{iip} = true
has_mul!(::FunctionOperator{iip}) where{iip} = iip
has_ldiv(L::FunctionOperator{iip}) where{iip} = !(L.op_inverse isa Nothing)
has_ldiv!(L::FunctionOperator{iip}) where{iip} = iip & !(L.op_inverse isa Nothing)

# TODO - FunctionOperator, Base.conj, transpose

# operator application
function Base.:*(L::FunctionOperator{iip,true}, u::AbstractVecOrMat) where{iip}
    L.op(u, L.p, L.t; L.traits.kwargs...)
end

function Base.:\(L::FunctionOperator{iip,true}, u::AbstractVecOrMat) where{iip}
    L.op_inverse(u, L.p, L.t; L.traits.kwargs...)
end

function Base.:*(L::FunctionOperator{true,false}, u::AbstractVecOrMat)
    _, co = L.cache
    du = zero(co)
    L.op(du, u, L.p, L.t; L.traits.kwargs...)
end

function Base.:\(L::FunctionOperator{true,false}, u::AbstractVecOrMat)
    ci, _ = L.cache
    du = zero(ci)
    L.op_inverse(du, u, L.p, L.t; L.traits.kwargs...)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::FunctionOperator{true}, u::AbstractVecOrMat)
    L.op(v, u, L.p, L.t; L.traits.kwargs...)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::FunctionOperator{false}, u::AbstractVecOrMat, args...)
    @error "LinearAlgebra.mul! not defined for out-of-place FunctionOperators"
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::FunctionOperator{true, oop, false}, u::AbstractVecOrMat, α, β) where{oop}
    _, co = L.cache

    copy!(co, v)
    mul!(v, L, u)
    lmul!(α, v)
    axpy!(β, co, v)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::FunctionOperator{true, oop, true}, u::AbstractVecOrMat, α, β) where{oop}
    L.op(v, u, L.p, L.t, α, β; L.traits.kwargs...)
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::FunctionOperator{true}, u::AbstractVecOrMat)
    L.op_inverse(v, u, L.p, L.t; L.traits.kwargs...)
end

function LinearAlgebra.ldiv!(L::FunctionOperator{true}, u::AbstractVecOrMat)
    ci, _ = L.cache
    copy!(ci, u)
    ldiv!(u, L, ci)
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::FunctionOperator{false}, u::AbstractVecOrMat)
    @error "LinearAlgebra.ldiv! not defined for out-of-place FunctionOperators"
end

function LinearAlgebra.ldiv!(L::FunctionOperator{false}, u::AbstractVecOrMat)
    @error "LinearAlgebra.ldiv! not defined for out-of-place FunctionOperators"
end
#
