module SciMLOperatorsLoopVectorizationExt

import LoopVectorization: @turbo
import SciMLOperators

const StridedMatrixOperator = SciMLOperators.MatrixOperator{<:Any, <:StridedMatrix}

SciMLOperators._has_tensor_outer_mul_fast(::StridedMatrixOperator) = true

function SciMLOperators._tensor_outer_mul_fast!(
        w, outer::StridedMatrixOperator, C, mi::Int, mo::Int, no::Int, k::Int
    )
    A = outer.A
    C = reshape(C, (mi, no, k))
    W = reshape(w, (mi, mo, k))

    @turbo for j in 1:k, m in 1:mo, i in 1:mi
        acc = zero(eltype(w))
        for o in 1:no
            acc += A[m, o] * C[i, o, j]
        end
        W[i, m, j] = acc
    end

    return w
end

function SciMLOperators._tensor_outer_mul_fast!(
        w, outer::StridedMatrixOperator, C, mi::Int, mo::Int, no::Int, k::Int, α, β
    )
    A = outer.A
    C = reshape(C, (mi, no, k))
    W = reshape(w, (mi, mo, k))

    @turbo for j in 1:k, m in 1:mo, i in 1:mi
        acc = zero(eltype(w))
        for o in 1:no
            acc += A[m, o] * C[i, o, j]
        end
        W[i, m, j] = α * acc + β * W[i, m, j]
    end

    return w
end

end
