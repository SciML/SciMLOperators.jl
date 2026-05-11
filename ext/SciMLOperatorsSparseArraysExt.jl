module SciMLOperatorsSparseArraysExt

import SciMLOperators
import SparseArrays: sparse, issparse
import SparseArrays
import LinearAlgebra

SparseArrays.sparse(L::SciMLOperators.MatrixOperator) = sparse(L.A)
SparseArrays.issparse(L::SciMLOperators.MatrixOperator) = issparse(L.A)
SparseArrays.sparse(L::SciMLOperators.ScaledOperator) = L.λ * sparse(L.L)
SparseArrays.sparse(L::SciMLOperators.AddedOperator) = sum(sparse, L.ops)
SparseArrays.sparse(L::SciMLOperators.ComposedOperator) = prod(sparse, L.ops)
SparseArrays.sparse(L::SciMLOperators.IdentityOperator) = sparse(LinearAlgebra.I, size(L))
function SparseArrays.sparse(L::SciMLOperators.TensorProductOperator)
    return LinearAlgebra.kron(sparse.(L.ops)...)
end
function SparseArrays.sparse(L::SciMLOperators.BlockDiagonalOperator)
    return SparseArrays.blockdiag(sparse.(L.ops)...)
end
function SparseArrays.sparse(L::SciMLOperators.NullOperator)
    return SparseArrays.spzeros(eltype(L), size(L))
end
SparseArrays.sparse(L::SciMLOperators.TensorSumOperator) = sum(sparse, L.products)

end
