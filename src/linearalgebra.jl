#
struct AdjointedOperator{T,LType} <: AbstractSciMLOperator{T}
    L::LType

    function AdjointedOperator(L::AbstractSciMLOperator{T}) where{T}
        new{T,typeof(L)}(L)
    end
end

Base.adjoint(L::AbstractSciMLOperator) = AdjointedOperator(L)
Base.adjoint(L::AdjointedOperator) = L.L

has_adjoint(L::AdjointedOperator) = true
isconstant(L::AdjointedOperator) = isconstant(L.L)

