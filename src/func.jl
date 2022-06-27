#
"""
    Matrix free operators (given by a function)
"""
struct FunctionOperator{isinplace,T,F,Fa,Fi,Fai,Tr,P,Tt,C} <: AbstractSciMLOperator{T}
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

    function FunctionOperator(op,
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
        T   = traits.T

        isset = cache !== nothing

        new{iip,
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

function FunctionOperator(op;

                          # necessary
                          isinplace=nothing,
                          T=nothing,
                          size=nothing,

                          input_prototype=nothing,
                          output_prototype=nothing,

                          # optional
                          op_adjoint=nothing,
                          op_inverse=nothing,
                          op_adjoint_inverse=nothing,

                          p=nothing,
                          t=nothing,

                          # traits
                          opnorm=nothing,
                          issymmetric=false,
                          ishermitian=false,
                          isposdef=false,
                         )

    isinplace isa Nothing  && @error "Please provide a funciton signature
    by specifying `isinplace` as either `true`, or `false`.
    If `isinplace = false`, the signature is `op(u, p, t)`,
    and if `isinplace = true`, the signature is `op(du, u, p, t)`.
    Further, it is assumed that the function call would be nonallocating
    when called in-place"
    T isa Nothing  && @error "Please provide a Number type for the Operator"
    size isa Nothing  && @error "Please provide a size (m, n)"
    if (input_prototype isa Nothing) | (output_prototype isa Nothing)
        @error "Please provide input/out prototypes vectors/arrays."
    end

    isreal = T <: Real
    adjointable = ishermitian | (isreal & issymmetric)
    invertible  = !(op_inverse isa Nothing)

    if adjointable & (op_adjoint isa Nothing) 
        op_adjoint = op
    end

    if invertible & (op_adjoint_inverse isa Nothing)
        op_adjoint_inverse = op_inverse
    end

    t = t isa Nothing ? zero(T) : t

    traits = (;
              opnorm = opnorm,
              issymmetric = issymmetric,
              ishermitian = ishermitian,
              isposdef = isposdef,

              isinplace = isinplace,
              T = T,
              size = size,
             )

    cache = (
             similar(input_prototype),
             similar(output_prototype),
            )
    isset = cache === nothing

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
    @set! L.p = p
    @set! L.t = t
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

    p = L.p
    t = L.t

    cache = reverse(L.cache)
    isset = cache !== nothing

    FunctionOperator(op,
                     op_adjoint,
                     op_inverse,
                     op_adjoint_inverse,
                     traits,
                     p,
                     t,
                     isset,
                     cache
                    )
end

function LinearAlgebra.opnorm(L::FunctionOperator, p)
    L.traits.opnorm === nothing && error("""
      M.opnorm is nothing, please define opnorm as a function that takes one
      argument. E.g., `(p::Real) -> p == Inf ? 100 : error("only Inf norm is
      defined")`
    """)
    opn = L.opnorm
    return opn isa Number ? opn : M.opnorm(p)
end
LinearAlgebra.issymmetric(L::FunctionOperator) = L.traits.issymmetric
LinearAlgebra.ishermitian(L::FunctionOperator) = L.traits.ishermitian
LinearAlgebra.isposdef(L::FunctionOperator) = L.traits.isposdef

getops(::FunctionOperator) = ()
has_adjoint(L::FunctionOperator) = !(L.op_adjoint isa Nothing)
has_mul(L::FunctionOperator{iip}) where{iip} = true
has_mul!(L::FunctionOperator{iip}) where{iip} = iip
has_ldiv(L::FunctionOperator{iip}) where{iip} = !(L.op_inverse isa Nothing)
has_ldiv!(L::FunctionOperator{iip}) where{iip} = iip & !(L.op_inverse isa Nothing)

# operator application
Base.:*(L::FunctionOperator{false}, u::AbstractVecOrMat) = L.op(u, L.p, L.t)
Base.:\(L::FunctionOperator{false}, u::AbstractVecOrMat) = L.op_inverse(u, L.p, L.t)

function Base.:*(L::FunctionOperator{true}, u::AbstractVecOrMat)
    _, co = L.cache
    du = copy(co)
    L.op(du, u, L.p, L.t)
end

function Base.:\(L::FunctionOperator{true}, u::AbstractVecOrMat)
    ci, _ = L.cache
    du = copy(ci)
    L.op_inverse(du, u, L.p, L.t)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::FunctionOperator{true}, u::AbstractVecOrMat)
    L.op(v, u, L.p, L.t)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::FunctionOperator{false}, u::AbstractVecOrMat)
    @error "3-argument is mul! not defined for out-of-place FunctionOperators"
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::FunctionOperator{true}, u::AbstractVecOrMat, α, β)
    _, co = L.cache

    copy!(co, v)
    mul!(v, L, u)
    lmul!(α, v)
    axpy!(β, co, v)
end

function LinearAlgebra.mul!(v::AbstractVecOrMat, L::FunctionOperator{false}, u::AbstractVecOrMat, α, β)
    @error "5-argument is mul! not defined for out-of-place FunctionOperators"
end

function LinearAlgebra.ldiv!(v::AbstractVecOrMat, L::FunctionOperator, u::AbstractVecOrMat)
    L.op_inverse(v, u, L.p, L.t)
end

function LinearAlgebra.ldiv!(L::FunctionOperator, u::AbstractVecOrMat)
    ci, _ = L.cache
    copy!(ci, u)
    ldiv!(u, L, ci)
end
#
