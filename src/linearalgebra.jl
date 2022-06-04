#
struct AdjointedOperator{T,LType} <: AbstractSciMLOperator{T}
    L::LType

    function AdjointedOperator(L::AbstractSciMLOperator{T}) where{T}
        new{T,typeof(L)}(L)
    end
end

Base.adjoint(L::AbstractSciMLOperator) = AdjointedOperator(L)
Base.adjoint(L::AdjointedOperator) = L.L

Base.size(L::AdjointedOperator) = size(L.L) |> reverse

has_adjoint(L::AdjointedOperator) = true

@forward AdjointedOperator.L (
                              # Base
                              convert,

                              # LinearAlgebra
                              LinearAlgebra.isreal,
                              LinearAlgebra.issymmetric,
                              LinearAlgebra.ishermitian,
                              LinearAlgebra.isposdef,
                              LinearAlgebra.opnorm,

                              # SciML
                              getops,

                              isconstant,
                              has_mul!,
                              has_ldiv,
                              has_ldiv!,
                             )


# oeprator application
AbstractAdjointedVector = Adjoint{<:Number, <:AbstractVector}

Base.:*(u::AbstractAdjointedVector, L::AdjointedOperator) = (L.L * u.parent)'
Base.:/(u::AbstractAdjointedVector, L::AdjointedOperator) = (L.L \ u.parent)'

function LinearAlgebra.mul!(v::AbstractAdjointedVector, L::AdjointedOperator, u::AbstractAdjointedVector)
    mul!(v.parent, L.L, u.parent)'
end

function LinearAlgebra.mul!(v::AbstractAdjointedVector, L::AdjointedOperator, u::AbstractAdjointedVector, α::Number, β::Number)
    mul!(v.parent, L.L, u.parent, α, β)'
end

function LinearAlgebra.ldiv!(v::AbstractAdjointedVector, L::AdjointedOperator, u::AbstractAdjointedVector)
    ldiv!(v.parent, L.L, u.parent)'
end

function LinearAlgebra.ldiv!(L::AdjointedOperator, u::AbstractAdjointedVector)
    ldiv!(L.L, u.parent)'
end
#
