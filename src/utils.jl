#
"""
We should define `sparse` for these types in `SciMLBase` instead,
but that package doesn't know anything about sparse arrays yet, so
we'll introduce a temporary work-around here.
"""
function _sparse end

_sparse(L) = sparse(L)
_sparse(L::SciMLMatrixOperator) = _sparse(L.A)
_sparse(L::SciMLScaledOperator) = L.λ * _sparse(L.L)

