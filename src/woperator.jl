abstract type AbstractWOperator{T} <: AbstractSciMLOperator{T} 

struct StaticWOperator{isinv, T, F} <: AbstractWOperator{T}
    W::T
    F::F
    function StaticWOperator(W::T, callinv = true) where {T}
        n = size(W, 1)
        isinv = n <= 7 # cutoff where LU has noticable overhead

        F = if isinv && callinv
            # this should be in ArrayInterface but can't be for silly reasons
            # doing to how StaticArrays and StaticArraysCore are split up
            StaticArrays.LU(LowerTriangular(W), UpperTriangular(W), SVector{n}(1:n))
        else
            lu(W, check = false)
        end
        # when constructing W for the first time for the type
        # inv(W) can be singular
        _W = if isinv && callinv
            inv(W)
        else
            W
        end
        new{isinv, T, typeof(F)}(_W, F)
    end
end
Base.:\(W::StaticWOperator{isinv}, v::AbstractArray) where {isinv} = isinv ? W.W * v : W.F \ v

"""
    WOperator(mass_matrix,gamma,J)

A linear operator that represents the W matrix of an ODEProblem, defined as

```math
W = \\frac{1}{\\gamma}MM - J
```

where `MM` is the mass matrix (a regular `AbstractMatrix` or a `UniformScaling`),
`γ` is a real number proportional to the time step, and `J` is the Jacobian
operator (must be a `AbstractSciMLOperator`). A `WOperator` can also be
constructed using a `*DEFunction` directly as

    WOperator(f,gamma)

`f` needs to have a jacobian and `jac_prototype`, but the prototype does not need
to be a diffeq operator --- it will automatically be converted to one.

`WOperator` supports lazy `*` and `mul!` operations, the latter utilizing an
internal cache (can be specified in the constructor; default to regular `Vector`).
It supports all of `AbstractSciMLOperator`'s interface.
"""
mutable struct WOperator{IIP, T,
    MType,
    GType,
    JType,
    F,
    C,
    JV} <: AbstractWOperator{T}
    mass_matrix::MType
    gamma::GType
    J::JType
    _func_cache::F           # cache used in `mul!`
    _concrete_form::C        # non-lazy form (matrix/number) of the operator
    jacvec::JV

    function WOperator{IIP}(mass_matrix, gamma, J, u, jacvec = nothing) where {IIP}
        # TODO: there is definitely a missing interface.
        # Tentative interface: `has_concrete` and `concertize(A)`
        if J isa Union{Number, ScalarOperator}
            _concrete_form = -mass_matrix / gamma + convert(Number, J)
            _func_cache = nothing
        else
            AJ = J isa MatrixOperator ? convert(AbstractMatrix, J) : J
            if AJ isa AbstractMatrix
                mm = mass_matrix isa MatrixOperator ?
                     convert(AbstractMatrix, mass_matrix) : mass_matrix
                if is_sparse(AJ)

                    # If gamma is zero, then it's just an initialization and we want to make sure
                    # we get the right sparsity pattern. If gamma is not zero, then it's a case where
                    # a new W is created (as part of an out-of-place solve) and thus the actual
                    # values actually matter!
                    #
                    # Constant operators never refactorize so always use the correct values there
                    # as well
                    if gamma == 0 && !(J isa MatrixOperator && isconstant(J))
                        # Workaround https://github.com/JuliaSparse/SparseArrays.jl/issues/190
                        # Hopefully `rand()` does not match any value in the array (prob ~ 0, with a check)
                        # Then `one` is required since gamma is zero
                        # Otherwise this will not pick up the union sparsity pattern
                        # But instead drop the runtime zeros (i.e. all values) of the AJ pattern!
                        AJn = nonzeros(AJ)
                        x = rand()
                        @assert all(!isequal(x), AJn)

                        fill!(AJn, rand())
                        _concrete_form = -mm / one(gamma) + AJ
                        fill!(_concrete_form, false) # safety measure, throw singular error if not filled
                    else
                        _concrete_form = -mm / gamma + AJ
                    end
                else
                    _concrete_form = -mm / gamma + AJ
                end

            else
                _concrete_form = nothing
            end
            _func_cache = zero(u)
        end
        T = eltype(_concrete_form)
        MType = typeof(mass_matrix)
        GType = typeof(gamma)
        JType = typeof(J)
        F = typeof(_func_cache)
        C = typeof(_concrete_form)
        JV = typeof(jacvec)
        return new{IIP, T, MType, GType, JType, F, C, JV}(mass_matrix, gamma, J,
            _func_cache, _concrete_form,
            jacvec)
    end
    
    function Base.copy(W::WOperator{IIP, T, MType, GType, JType, F, C, JV}) where {IIP, T, MType, GType, JType, F, C, JV}
        return new{IIP, T, MType, GType, JType, F, C, JV}(
            W.mass_matrix, 
            W.gamma, 
            W.J, 
            W._func_cache === nothing ? nothing : copy(W._func_cache),
            W._concrete_form === nothing ? nothing : copy(W._concrete_form),
            W.jacvec
        )
    end
end
Base.eltype(W::WOperator) = eltype(W.J)

# In WOperator update_coefficients!, accept both missing u/p/t and missing dtgamma and don't update them in that case.
# This helps support partial updating logic used with Newton solvers.
function update_coefficients!(W::WOperator,
        u = nothing,
        p = nothing,
        t = nothing;
        dtgamma = nothing)
    if (u !== nothing) && (p !== nothing) && (t !== nothing)
        update_coefficients!(W.J, u, p, t)
        update_coefficients!(W.mass_matrix, u, p, t)
        !isnothing(W.jacvec) && update_coefficients!(W.jacvec, u, p, t)
    end
    dtgamma !== nothing && (W.gamma = dtgamma)
    W
end

function update_coefficients!(J::UJacobianWrapper, u, p, t)
    J.p = p
    J.t = t
end

function Base.convert(::Type{AbstractMatrix}, W::WOperator{IIP}) where {IIP}
    if !IIP
        # Allocating
        W._concrete_form = -W.mass_matrix / W.gamma + convert(AbstractMatrix, W.J)
    else
        # Non-allocating already updated
        #_W = W._concrete_form
        #jacobian2W!(_W, W.mass_matrix, W.gamma, W.J)
    end
    return W._concrete_form
end
function Base.convert(::Type{Number}, W::WOperator)
    W._concrete_form = -W.mass_matrix / W.gamma + convert(Number, W.J)
    return W._concrete_form
end
Base.size(W::WOperator) = size(W.J)
Base.size(W::WOperator, d::Integer) = d <= 2 ? size(W)[d] : 1
function Base.getindex(W::WOperator, i::Int)
    -W.mass_matrix[i] / W.gamma + W.J[i]
end
function Base.getindex(W::WOperator, I::Vararg{Int, N}) where {N}
    -W.mass_matrix[I...] / W.gamma + W.J[I...]
end
function Base.:*(W::WOperator, x::AbstractVecOrMat)
    (W.mass_matrix * x) / -W.gamma + W.J * x
end
function Base.:*(W::WOperator, x::Number)
    (W.mass_matrix * x) / -W.gamma + W.J * x
end
function Base.:\(W::WOperator, x::AbstractVecOrMat)
    if size(W) == () # scalar operator
        convert(Number, W) \ x
    else
        convert(AbstractMatrix, W) \ x
    end
end
function Base.:\(W::WOperator, x::Number)
    if size(W) == () # scalar operator
        convert(Number, W) \ x
    else
        convert(AbstractMatrix, W) \ x
    end
end

function LinearAlgebra.mul!(Y::AbstractVecOrMat, W::WOperator, B::AbstractVecOrMat)
    # Compute mass_matrix * B
    if isa(W.mass_matrix, UniformScaling)
        a = -W.mass_matrix.λ / W.gamma
        @.. broadcast=false Y=a*B
    else
        mul!(_vec(Y), W.mass_matrix, _vec(B))
        lmul!(-inv(W.gamma), Y)
    end
    # Compute J * B and add
    if W.jacvec !== nothing
        mul!(_vec(W._func_cache), W.jacvec, _vec(B))
    else
        mul!(_vec(W._func_cache), W.J, _vec(B))
    end
    _vec(Y) .+= _vec(W._func_cache)
end


has_concretization(::AbstractWOperator) = true
