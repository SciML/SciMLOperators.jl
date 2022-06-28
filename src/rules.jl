#
###
# automatic differentiation interface
###

function ChainRulesCore.rrule(
                              ::typeof(Base.:*),
                              L::FunctionOperator,
                              u::AbstractVecOrMat,
                             )
    v = L * u

    # overwrite projectto for functionoperator.
    # only gradient holding field in FunctionOperator is `p`, `t`
    # accumulate gradient WRT p
    project_L = ProjectTo(L)
    project_u = ProjectTo(u)

    function times_pullback(dv)
        dv = unthunk(dv)

#       dL = @thunk(project_L(dv * u')) 
#       dL = Tangent{L}(...)
        du = @thunk(project_u(L' * dv))

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

###
# rrule
###

#=
function ChainRulesCore.rrule(
                              ::typeof(Base.:*),
                              L::AbstractSciMLOperator,
                              u::AbstractVecOrMat,
                             )
    v = L * u

    project_L = ProjectTo(L)
    project_u = ProjectTo(u)

    function times_pullback(dv)
        dv = unthunk(dv)

        dL = @thunk(project_L(dv * u'))
        dL = Tangent{L}(...)
        du = @thunk(project_u(L' * dv))

        NoTangent(), dL, du
    end

    v, times_pullback
end

#overload Tangent{L::MatrixOperator}(dv)

function ChainRulesCore.rrule(
                              ::typeof(Base.:*),
                              L::AffineOperator,
                              u::AbstractVecOrMat
                             )
    v = L * u

    project_u = ProjectTo(u)

    function times_pullback(dv)
        dv = unthunk(dv)

        dA = @thunk(dv * u')
        dB = @thunk()
        db = copy(dv)

        dL = Tangent{L}(; A=dA, B=dB, b=db)

        du = @thunk(project_u(L' * dv))

        NoTangent(), dL, du
    end

    v, times_pullback
end

function ChainRulesCore.rrule(
                              ::typeof(Base.:*),
                              L::FunctionOperator{true},
                              u::AbstractVecOrMat
                             )
    v = L * u

    project_u = ProjectTo(u)

    function times_pullback(dv)

#       dL = Tangent{L}()
        dL = NoTangent()
        du = @thunk(project_u(L' * dv))

        NoTangent(), dL, du
    end

    v, times_pullback
end
=#
#
