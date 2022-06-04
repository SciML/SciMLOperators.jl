#
AbstractAdjointedVector  = Adjoint{  <:Number, <:AbstractVector}
AbstractTransposedVector = Transpose{<:Number, <:AbstractVector}

struct AdjointedOperator{T,LType} <: AbstractSciMLOperator{T}
    L::LType

    function AdjointedOperator(L::AbstractSciMLOperator{T}) where{T}
        new{T,typeof(L)}(L)
    end
end

struct TransposedOperator{T,LType} <: AbstractSciMLOperator{T}
    L::LType

    function TransposedOperator(L::AbstractSciMLOperator{T}) where{T}
        new{T,typeof(L)}(L)
    end
end

for (op, LType, VType) in (
                           (:adjoint,   :AdjointedOperator,  :AbstractAdjointedVector ),
                           (:transpose, :TransposedOperator, :AbstractTransposedVector),
                          )
    # constructor
    @eval Base.$op(L::AbstractSciMLOperator) = $LType(L)

    @eval Base.convert(AbstractMatrix, L::$LType) = $op(convert(AbstractMatrix, L))

    # traits
    @eval Base.size(L::$LType) = size(L.L) |> reverse
    @eval Base.$op(L::$LType) = L.L

    @eval has_adjoint(L::$LType) = true
    @eval getops(L::$LType) = (L.L,)

    @eval @forward $LType.L (
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

    # oeprator application
    @eval Base.:*(u::$VType, L::$LType) = $op(L.L * u.parent)
    @eval Base.:/(u::$VType, L::$LType) = $op(L.L \ u.parent)

    @eval function LinearAlgebra.mul!(v::$VType, L::$LType, u::$VType)
        mul!(v.parent, L.L, u.parent)
        $op(v)
    end

    @eval function LinearAlgebra.mul!(v::$VType, L::$LType, u::$VType, α::Number, β::Number)
        mul!(v.parent, L.L, u.parent, α, β)
        v
    end

    @eval function LinearAlgebra.ldiv!(v::$VType, L::$LType, u::$VType)
        ldiv!(v.parent, L.L, u.parent)
    end
    
    @eval function LinearAlgebra.ldiv!(L::$LType, u::$VType)
        ldiv!(L.L, u.parent)
        u
    end
end
#
