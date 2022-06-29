#
###
# automatic differentiation interface
###

function ChainRulesCore.rrule(
                              ::typeof(Base.:*),
                              L::FunctionOperator{true},
                              u::AbstractVecOrMat,
                             )
    v = L * u

    project_L = ProjectTo(L)
    project_u = ProjectTo(u)

    function times_pullback(dv)
        @assert has_adjoint(L) "Adjoint not defined for in-place operator of
        type $(typeof(L)). To do reverse pass, either define the adjoint to the
        operator via the `op_adjoint` kwarg, or use an out-of-place version."

        # how to accumulate gradient WRT `p`, `t` ??
        # overwrite ProjectTo for functionoperator ??

#       dL = @thunk(project_L(dv * u')) 
#       dL = Tangent{L}(...)
        du = @thunk(project_u(L' * unthunk(dv)))

        NoTangent(), NoTangent(), du
    end

    v, times_pullback
end

###
# frule
###

#=
function ChainRulesCore.ProjectTo(L::MatrixOperator)
    info = (;
            A = ProjectTo(L.A),
            update_func = ProjectTo(L.update_func),
           )
    ProjectTo{MatrixOperator}(info)
end

function (p::ChainRulesCore.ProjectTo{MatrixOperator})(dx::AbstractMatrix)
end

function ChainRulesCore.frule(
                              (_,ΔL,Δu),
                              ::typeof(Base.:*),
                              L::AbstractSciMLOperator,
                              u::AbstractVecOrMat,
                             )
    @show typeof(ΔL) # <-- 
    @show typeof(Δu)
    v = L * u
    Δ = muladd(ΔL, u, L * Δu)
end
=#

#
