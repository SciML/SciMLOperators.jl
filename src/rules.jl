#
###
# automatic differentiation interface
###

# ChainRulesCore.frule

# - tensor product, functionoperator, etc

# TODO overwrite ProjectTo
#function ChainRulesCore.ProjectTo(L::MatrixOperator{<:Any,<:Diagonal})
#end

###
# frule
###

#function Base.:*(ΔL::Tangent{P<:AbstractSciMLOperator,T}, u::AbstractVecOrMat) where{P,T}
##   P(ΔL.backing...) * u
#end



# look at ProjectTo{Diagonal} for ref
function ChainRulesCore.ProjectTo(L::MatrixOperator)
    info = (;
            A = ProjectTo(L.A),
            update_func = ProjectTo(L.update_func),
           )
    ProjectTo{MatrixOperator}(info)
end

function (project::ChainRulesCore.ProjectTo{MatrixOperator})(dx::AbstractMatrix)
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

function ChainRulesCore.frule(
                              (_,ΔL,Δu),
                              ::typeof(Base.:*),
                              L::Diagonal,
                              u::AbstractVecOrMat,
                             )
    @show typeof(ΔL)
    @show typeof(Δu)
    v = L * u
    Δ = muladd(ΔL, u, L * Δu)

    v, Δ
end

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
