#
###
# automatic differentiation interface
###

# rrule for updatecoefficients!
#function ChainRulesCore.rrule(
#                              L::AbstractSciMLOperator,
#                              u::AbstractVecOrMat,
#                              p,
#                              t,
#                             )
#    L(u, p, t), function(dv)
#
#        NoTangent(), du, dp, dt
#    end
#end

# FunctionOperator
function ChainRulesCore.rrule(
                              ::typeof(Base.:*),
                              L::FunctionOperator{true},
                              u::AbstractVecOrMat,
                             )
    v = zero(L.cache[2])

    rrule_ret = rrule(L.op, v, u, L.p, L.t)

    op_pullback = if rrule_ret isa Nothing
        L.op(v, u, L.p, L.t)

        nothing
    else
        rrule_ret[2]
    end

    project_u = ProjectTo(u)

    function times_pullback(dv)
        if op_pullback isa Nothing
            if !(has_adjoint(L))
                @error "cant really ad w/o adjoint or pullback"
            end

            @warn "not computing tangents for p, t"
            du = @thunk(project_u(L' * unthunk(dv)))
            dp = NoTangent()
            dt = NoTangent()
        else
            _, du, dp, dt = op_pullback(dv)
        end

        dL = Tangent{FunctionOperator}(;p=dp, t=dt)

        NoTangent(), dL, du
    end

    v, times_pullback
end
#
