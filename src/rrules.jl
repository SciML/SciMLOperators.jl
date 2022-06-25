#
###
# automatic differentiation interface
###

# TODO - specialize for each type ? Or leaf nodes?
# or do we just need it for special cases?
#   - tensor product, functionoperator, etc


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

#function get_adjoint(L::AbstractSciMLOperator, dv)
#end

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
#
