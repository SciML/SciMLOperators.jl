abstract type AbstractWOperator{T} <: AbstractSciMLOperator{T} end

"""
$(TYPEDEF)

Small dense factorization helper for repeated solves with a fixed `W` matrix.

# Arguments

  - `W`: Concrete square matrix to solve against.
  - `callinv`: Whether very small matrices may store `inv(W)` for direct
    multiplication during `\\`.

# Fields

$(FIELDS)

# Interface Rules

`StaticWOperator` is a specialized helper for solver internals that need a
fixed W-operator solve. It supports `Wstatic \\ v`; it does not participate in
coefficient updates and should be reconstructed when the underlying matrix
changes.

# Examples

```julia
using SciMLOperators

W = StaticWOperator([2.0 0.0; 0.0 4.0])
W \\ [2.0, 8.0]
```
"""
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
$TYPEDEF

    WOperator{IIP}(mass_matrix, gamma, J, u[, jacvec])

A linear operator that represents the W matrix of an ODEProblem, defined as

```math
W = \\frac{1}{\\gamma}MM - J
```

where `MM` is the mass matrix, `γ` is a scalar, and `J` is the Jacobian
operator.

# Arguments

  - `mass_matrix`: A matrix-like object, `UniformScaling`, or `MatrixOperator`
    representing `MM`.
  - `gamma`: Scalar coefficient in the W-operator definition.
  - `J`: Jacobian represented as a number, matrix, or `AbstractSciMLOperator`.
  - `u`: Prototype state used to allocate the internal multiplication cache.
  - `jacvec`: Optional operator used for Jacobian-vector products in `mul!`.

# Fields

$(FIELDS)

# Interface Rules

`WOperator` is part of the public solver-developer interface used by implicit
ODE solvers. It supports matrix-like `*`, `\\`, `mul!`, indexing, sizing, and
concretization. Calling `update_coefficients!(W, u, p, t; gamma)` updates the
Jacobian, mass matrix, optional Jacobian-vector operator, and stored `gamma`.
Omitting `(u, p, t)` leaves those operators unchanged and only updates `gamma`
when it is supplied.

`IIP` controls whether conversion reuses the internally stored concrete form
as an in-place operator. The public contract is the mathematical action of
`W`; downstream code should not depend on `_func_cache` or `_concrete_form`.

# Examples

```julia
using LinearAlgebra, SciMLOperators

J = MatrixOperator([1.0 2.0; 3.0 4.0])
W = WOperator{true}(I, 0.5, J, zeros(2))

v = [1.0, 2.0]
W * v == (Matrix(W) * v)
```
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
    if !IIP || W.J isa AbstractSciMLOperator
        # Mirror the constructor: materialize a MatrixOperator mass matrix so the
        # result type matches `_concrete_form` (otherwise this becomes an
        # AddedOperator and the assignment back into the Matrix-typed slot fails).
        # The IIP case with a plain-matrix `J` is maintained externally via
        # `jacobian2W!` writing into `_concrete_form`; when `J` is an operator no
        # caller maintains it, so it must be rebuilt here with the current gamma.
        mm = W.mass_matrix isa MatrixOperator ?
            convert(AbstractMatrix, W.mass_matrix) : W.mass_matrix
        W._concrete_form = -mm / W.gamma + convert(AbstractMatrix, W.J)
    end
    return W._concrete_form
end
function Base.convert(::Type{Number}, W::WOperator)
    W._concrete_form = -W.mass_matrix / W.gamma + convert(Number, W.J)
    return W._concrete_form
end
# `convert(AbstractMatrix, W)` fuses `-M/γ` with `convert(AbstractMatrix, W.J)`, so `W` is
# convertible exactly when both its mass matrix `M` and its Jacobian `W.J` are. A matrix-free
# `W.J` (e.g. a Jacobian-vector-product operator) — or a matrix-free mass matrix — makes `W`
# matrix-free too; without this the default `all(isconvertible, getops(W)) ==
# all(isconvertible, ()) == true` would wrongly claim a matrix-free `W` is convertible.
isconvertible(W::WOperator) = isconvertible(W.mass_matrix) && isconvertible(W.J)
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
