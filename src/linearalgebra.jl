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
                              isconstant,
                              has_mul!,
                              has_ldiv,
                              has_ldiv!,
                             )


getops(L::AdjointedOperator) = (L.L,)
