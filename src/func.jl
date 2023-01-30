#
"""
    Matrix free operators (given by a function)
"""
mutable struct FunctionOperator{iip,oop,T<:Number,F,Fa,Fi,Fai,Tr,P,Tt,C} <: AbstractSciMLOperator{T}
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
    """ Is cache set? """
    isset::Bool
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
                              isset,
                              cache
                             )

        iip = traits.isinplace
        oop = traits.outofplace
        T   = traits.T

        isset = cache !== nothing

        new{
            iip,
            oop,
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
             isset,
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

function FunctionOperator(op,
                          input::AbstractVecOrMat,
                          output::AbstractVecOrMat =  input;

                          isinplace::Union{Nothing,Bool}=nothing,
                          outofplace::Union{Nothing,Bool}=nothing,
                          T::Union{Type{<:Number},Nothing}=nothing,

                          op_adjoint=nothing,
                          op_inverse=nothing,
                          op_adjoint_inverse=nothing,

                          p=nothing,
                          t::Union{Number,Nothing}=nothing,

                          # traits
                          islinear::Bool = false,

                          opnorm = nothing,
                          issymmetric::Bool = false,
                          ishermitian::Bool = false,
                          isposdef::Bool = false,
                         )

    sz = (size(output, 1), size(input, 1))
    T  = T isa Nothing ? promote_type(eltype.((input, output))...) : T
    t  = t isa Nothing ? zero(real(T)) : t

    isinplace = if isinplace isa Nothing
        static_hasmethod(op, typeof((output, input, p, t)))
    else
        isinplace
    end

    outofplace = if outofplace isa Nothing
        static_hasmethod(op, typeof((input, p, t)))
    else
        outofplace
    end

    if !isinplace & !outofplace
        @error "Please provide a funciton with signatures `op(u, p, t)` for applying
        the operator out-of-place, and/or the signature is `op(du, u, p, t)` for
        in-place application."
    end

    T isa Nothing && @error "Please provide a Number type for the Operator"

    issquare = size(input) == size(output)

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
              issquare = issquare,
              islinear = islinear,

              opnorm = opnorm,
              issymmetric = issymmetric,
              ishermitian = ishermitian,
              isposdef = isposdef,

              isinplace = isinplace,
              outofplace = outofplace,
              T = T,
              size = sz,
             )

    cache = zero.((input, output))
    isset = true

    FunctionOperator(
                     op,
                     op_adjoint,
                     op_inverse,
                     op_adjoint_inverse,
                     traits,
                     p,
                     t,
                     isset,
                     cache,
                    )
end

function update_coefficients!(L::FunctionOperator, u, p, t)
    L.p = p
    L.t = t

    nothing
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

    p = L.p
    t = L.t

    isset = L.isset
    cache = if isset
        cache = reverse(L.cache)
    else
        nothing
    end

    FunctionOperator(op,
                     op_adjoint,
                     op_inverse,
                     op_adjoint_inverse,
                     traits,
                     p,
                     t,
                     isset,
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

    @set! traits.opnorm = if traits.opnorm isa Number
        1 / traits.opnorm
    elseif traits.opnorm isa Nothing
        nothing
    else
        (p::Real) -> 1 / traits.opnorm(p)
    end

    p = L.p
    t = L.t

    isset = L.cache !== nothing
    cache = if isset
        cache = reverse(L.cache)
    else
        nothing
    end

    FunctionOperator(op,
                     op_adjoint,
                     op_inverse,
                     op_adjoint_inverse,
                     traits,
                     p,
                     t,
                     isset,
                     cache,
                    )
end

function LinearAlgebra.opnorm(L::FunctionOperator, p)
    L.traits.opnorm === nothing && error("""
      M.opnorm is nothing, please define opnorm as a function that takes one
      argument. E.g., `(p::Real) -> p == Inf ? 100 : error("only Inf norm is
      defined")`
    """)
    opn = L.opnorm
    return opn isa Number ? opn : L.opnorm(p)
end
LinearAlgebra.issymmetric(L::FunctionOperator) = L.traits.issymmetric
LinearAlgebra.ishermitian(L::FunctionOperator) = L.traits.ishermitian
LinearAlgebra.isposdef(L::FunctionOperator) = L.traits.isposdef

getops(::FunctionOperator) = ()
islinear(L::FunctionOperator) = L.traits.islinear
has_adjoint(L::FunctionOperator) = !(L.op_adjoint isa Nothing)
has_mul(L::FunctionOperator{iip}) where{iip} = true
has_mul!(L::FunctionOperator{iip}) where{iip} = iip
has_ldiv(L::FunctionOperator{iip}) where{iip} = !(L.op_inverse isa Nothing)
has_ldiv!(L::FunctionOperator{iip}) where{iip} = iip & !(L.op_inverse isa Nothing)

# TODO - FunctionOperator, Base.conj, transpose

# operator application
Base.:*(L::FunctionOperator{iip,true}, u::AbstractVecOrMat) where{iip} = L.op(u, L.p, L.t)
Base.:\(L::FunctionOperator{iip,true}, u::AbstractVecOrMat) where{iip} = L.op_inverse(u, L.p, L.t)

function Base.:*(L::FunctionOperator{true,false}, u::AbstractVecOrMat)
    _, co = L.cache
    du = zero(co)
    L.op(du, u, L.p, L.t)
end

function Base.:\(L::FunctionOperator{true,false}, u::AbstractVecOrMat)
    ci, _ = L.cache
    du = zero(ci)
    L.op_inverse(du, u, L.p, L.t)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::FunctionOperator{true}, u::AbstractVecOrMat)
    L.op(v, u, L.p, L.t)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::FunctionOperator{false}, u::AbstractVecOrMat, args...)
    @error "LinearAlgebra.mul! not defined for out-of-place FunctionOperators"
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::FunctionOperator{true}, u::AbstractVecOrMat, α, β)
    _, co = L.cache

    copy!(co, v)
    mul!(v, L, u)
    lmul!(α, v)
    axpy!(β, co, v)
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::FunctionOperator{true}, u::AbstractVecOrMat)
    L.op_inverse(v, u, L.p, L.t)
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
