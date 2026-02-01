abstract type AbstractWOperator{T} <: AbstractSciMLOperator{T} end

struct StaticWOperator{isinv, T, F} <: AbstractWOperator{T}
    W::T
    F::F
    function StaticWOperator(W::T, callinv = true) where {T}
        n = size(W, 1)
        isinv = n <= 7 # cutoff where LU has noticeable overhead

        F = if isinv && callinv
            ArrayInterface.lu_instance(W)
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
        return new{isinv, T, typeof(F)}(_W, F)
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
`γ` is a real number, and `J` is the Jacobian operator (must be a `AbstractSciMLOperator`).

`WOperator` supports lazy `*` and `mul!` operations, the latter utilizing an
internal cache (can be specified in the constructor; default to regular `Vector`).
"""
mutable struct WOperator{
        IIP, T,
        MType,
        GType,
        JType,
        F,
        C,
        JV,
    } <: AbstractWOperator{T}
    mass_matrix::MType
    gamma::GType
    J::JType
    _func_cache::F           # cache used in `mul!`
    _concrete_form::C        # non-lazy form (matrix/number) of the operator
    jacvec::JV

    function WOperator{IIP}(mass_matrix, gamma, J, u, jacvec = nothing) where {IIP}
        if J isa Union{Number, ScalarOperator}
            _concrete_form = -mass_matrix / gamma + convert(Number, J)
            _func_cache = nothing
        else
            AJ = J isa MatrixOperator ? convert(AbstractMatrix, J) : J
            mm = mass_matrix isa MatrixOperator ?
                convert(AbstractMatrix, mass_matrix) : mass_matrix
            _concrete_form = -mm / gamma + AJ
            _func_cache = zero(u)
        end
        T = eltype(_concrete_form)
        MType = typeof(mass_matrix)
        GType = typeof(gamma)
        JType = typeof(J)
        F = typeof(_func_cache)
        C = typeof(_concrete_form)
        JV = typeof(jacvec)
        return new{IIP, T, MType, GType, JType, F, C, JV}(
            mass_matrix, gamma, J,
            _func_cache, _concrete_form, jacvec
        )
    end

    function Base.copy(W::WOperator{IIP, T, MType, GType, JType, F, C, JV}) where {IIP, T, MType, GType, JType, F, C, JV}
        return new{IIP, T, MType, GType, JType, F, C, JV}(
            W.mass_matrix,
            W.gamma,
            W.J,
            copy(W._func_cache),
            copy(W._concrete_form),
            W.jacvec
        )
    end
end
Base.eltype(W::WOperator) = eltype(W.J)

# In WOperator update_coefficients!, accept both missing u/p/t and missing gamma and don't update them in that case.
# This helps support partial updating logic used with Newton solvers.
# Accept both `gamma` and deprecated `dtgamma` kwargs for backwards compatibility.
function update_coefficients!(
        W::WOperator,
        u = nothing,
        p = nothing,
        t = nothing;
        gamma = nothing,
        dtgamma = nothing
    )
    if dtgamma !== nothing
        Base.depwarn(
            "keyword argument `dtgamma` is deprecated, use `gamma` instead",
            :update_coefficients!
        )
        if gamma === nothing
            gamma = dtgamma
        end
    end
    if (u !== nothing) && (p !== nothing) && (t !== nothing)
        update_coefficients!(W.J, u, p, t)
        update_coefficients!(W.mass_matrix, u, p, t)
        !isnothing(W.jacvec) && update_coefficients!(W.jacvec, u, p, t)
    end
    gamma !== nothing && (W.gamma = gamma)
    return W
end

function Base.convert(::Type{AbstractMatrix}, W::WOperator{IIP}) where {IIP}
    if !IIP
        # Allocating
        W._concrete_form = -W.mass_matrix / W.gamma + convert(AbstractMatrix, W.J)
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
    return -W.mass_matrix[i] / W.gamma + W.J[i]
end
function Base.getindex(W::WOperator, I::Vararg{Int, N}) where {N}
    return -W.mass_matrix[I...] / W.gamma + W.J[I...]
end
function Base.:*(W::WOperator, x::AbstractVecOrMat)
    return (W.mass_matrix * x) / -W.gamma + W.J * x
end
function Base.:*(W::WOperator, x::Number)
    return (W.mass_matrix * x) / -W.gamma + W.J * x
end
function Base.:\(W::WOperator, x::AbstractVecOrMat)
    return if size(W) == () # scalar operator
        convert(Number, W) \ x
    else
        convert(AbstractMatrix, W) \ x
    end
end
function Base.:\(W::WOperator, x::Number)
    return if size(W) == () # scalar operator
        convert(Number, W) \ x
    else
        convert(AbstractMatrix, W) \ x
    end
end

function LinearAlgebra.mul!(Y::AbstractVector, W::WOperator, B::AbstractVector)
    # Compute mass_matrix * B
    if isa(W.mass_matrix, UniformScaling)
        a = -W.mass_matrix.λ / W.gamma
        @. Y = a * B
    else
        mul!(Y, W.mass_matrix, B)
        lmul!(-inv(W.gamma), Y)
    end
    # Compute J * B and add
    if isnothing(W.jacvec)
        mul!(W._func_cache, W.J, B)
    else
        mul!(W._func_cache, W.jacvec, B)
    end
    return Y .+= W._func_cache
end

function LinearAlgebra.mul!(Y::AbstractArray, W::WOperator, B::AbstractArray)
    # Compute mass_matrix * B
    if isa(W.mass_matrix, UniformScaling)
        a = -W.mass_matrix.λ / W.gamma
        @. Y = a * B
    else
        mul!(vec(Y), W.mass_matrix, vec(B))
        lmul!(-inv(W.gamma), Y)
    end
    # Compute J * B and add
    if isnothing(W.jacvec)
        mul!(vec(W._func_cache), W.J, vec(B))
    else
        mul!(vec(W._func_cache), W.jacvec, vec(B))
    end
    return vec(Y) .+= vec(W._func_cache)
end

has_concretization(::AbstractWOperator) = true
