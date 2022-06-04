#
struct AdjointedOperator{T,LType} <: AbstractSciMLOperator{T}
    L::LType

    function AdjointedOperator(L::AbstractSciMLOperator{T}) where{T}
        new{T,typeof(L)}(L)
    end
end

Base.adjoint(L::AbstractSciMLOperator) = AdjointedOperator(L)
Base.adjoint(L::AdjointedSciMLOperator) = L.L

has_adjoint(L::AdjointedSciMLOperator) = true
isconstant(L::AdjointedSciMLOperator) = isconstant(L.L)

