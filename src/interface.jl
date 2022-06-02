#
###
# (u,p,t) and (du,u,p,t) interface
###

#=
1. Function call and multiplication: L(du, u, p, t) for inplace and du = L(u, p, t) for
   out-of-place, meaning L*u and mul!(du, L, u).
2. If the operator is not a constant, update it with (u,p,t). A mutating form, i.e.
   update_coefficients!(A,u,p,t) that changes the internal coefficients, and a
   out-of-place form B = update_coefficients(A,u,p,t).
3. isconstant(A) trait for whether the operator is constant or not.
4. islinear(A) trait for whether the operator is linear or not.
=#

DEFAULT_UPDATE_FUNC(A,u,p,t) = A
function update_coefficients!(L::AbstractSciMLOperator, u, p, t)
    for op in getops(L)
        update_coefficients!(op, u, p, t)
    end
    L
end

for T in (
          SciMLIdentity,
          SciMLNullOperator,
          SciMLScalar,
          SciMLScaledOperator,
          SciMLAddedOperator,
          SciMLComposedOperator,

          SciMLMatrixOperator,
          SciMLFactorizedOperator,
#         SciMLFunctionOperator,
         )

    (L::T)(u, p, t) = (update_coefficients!(L, u, p, t); L * u)
    (L::T)(du, u, p, t) = (update_coefficients!(L, u, p, t); mul!(du, L, u))
end

update_coefficients!(L,u,p,t) = nothing
update_coefficients(L,u,p,t) = L

function SciMLOperator(L;
                       update_func=DEFAULT_UPDATE_FUNC,
                       islinear=false,
                       issymmetric=false,
                       kwargs...,
                      )

    if update_func === DEFAULT_UPDATE_FUNC
        isconstant = true
    end

    if L isa AbstractSciMLOperator
        L
    elseif L isa Union{
                       Number, UniformScaling,
                      }
        SciMLScalar(L; update_func=update_func)
    elseif L isa AbstractMatrix
        SciMLMatrixOperator(L; update_func=update_func)
    elseif L isa Factorization
        SciMLFactorizedOperator(L)
    else
        SciMLFunctionOperator(L, kwargs...)
    end
end
