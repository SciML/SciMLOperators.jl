# The AbstractSciMLOperator Interface

## Formal Properties of SciMLOperators

These are the formal properties that an `AbstractSciMLOperator` should obey
for it to work in the solvers.

1. An `AbstractSciMLOperator` represents a linear or nonlinear operator with input/output
   being `AbstractArray`s. Specifically, a SciMLOperator, `L`, of size `(M,N)` accepts
   input argument `u` with leading length `N`, i.e. `size(u, 1) == N`, and returns an
   `AbstractArray` of the same dimension with leading length `M`, i.e. `size(L * u, 1) == M`.
2. SciMLOperators can be applied to an `AbstractArray` via overloaded `Base.*`, or
   the in-place `LinearAlgebra.mul!`. Additionally, operators are allowed to be time,
   or parameter dependent. The state of a SciMLOperator can be updated by calling
   the mutating function `update_coefficients!(L, u, p, t)` where `p` representes
   parameters, and `t`, time.  Calling a SciMLOperator as `L(du, u, p, t)` or out-of-place
   `L(u, p, t)` will automatically update the state of `L` before applying it to `u`.
   `L(u, p, t)` is the same operation as `L(u, p, t) * u`.
3. To support the update functionality, we have lazily implemented a comprehensive operator
   algebra. That means a user can add, subtract, scale, compose and invert SciMLOperators,
   and the state of the resultant operator would be updated as expected upon calling
   `L(du, u, p, t)` or `L(u, p, t)` so long as an update function is provided for the
   component operators.

## AbstractSciMLOperator Interface Description

1. `AbstractSciMLLinearOperator <: AbstractSciMLOperator`
2. `AbstractSciMLScalarOperator <: AbstractSciMLLinearOperator`
3. `isconstant(A)` trait for whether the operator is constant or not.
4. Optional: `exp(A)`. Required for simple exponential integration.
5. Optional: `expv(A,u,t) = exp(t*A)*u` and `expv!(v,A::AbstractSciMLOperator,u,t)`
   Required for sparse-saving exponential integration.
6. Optional: factorizations. `ldiv!`, `factorize` et. al. This is only required
   for algorithms which use the factorization of the operator (Crank-Nicolson),
   and only for when the default linear solve is used.

## Note About Affine Operators

Affine operators are operators which have the action `Q*x = A*x + b`. These operators have
no matrix representation, since if there was it would be a linear operator instead of an 
affine operator. You can only represent an affine operator as a linear operator in a 
dimension of one larger via the operation: `[A b] * [u;1]`, so it would require something modified 
to the input as well. As such, affine operators are a distinct generalization of linear operators.

While it this seems like it might doom the idea of using matrix-free affine operators, it turns out 
that affine operators can be used in all cases where matrix-free linear solvers are used due to
an easy genearlization of the standard convergence proofs. If Q is the affine operator 
``Q(x) = Ax + b``, then solving ``Qx = c`` is equivalent to solving ``Ax + b = c`` or ``Ax = c-b``. 
If you know do this same "plug-and-chug" handling of the affine operator in into the GMRES/CG/etc. 
convergence proofs, move the affine part to the rhs residual, and show it converges to solving 
``Ax = c-b``, and thus GMRES/CG/etc. solves ``Q(x) = c`` for an affine operator properly. 

That same trick then can be used pretty much anywhere you would've had a linear operator to extend 
the proof to affine operators, so then ``exp(A*t)*v`` operations via Krylov methods work for A being 
affine as well, and all sorts of things. Thus affine operators have no matrix representation but they 
are still compatible with essentially any Krylov method which would otherwise be compatible with
matrix-free representations, hence their support in the SciMLOperators interface.

## Note about keyword arguments to `update_coefficients!`

In rare cases, an operator may be used in a context where additional state is expected to be provided
to `update_coefficients!` beyond `u`, `p`, and `t`. In this case, the operator may accept this additional
state through arbitrary keyword arguments to `update_coefficients!`. When the caller provides these, they will be recursively propagated downwards through composed operators just like `u`, `p`, and `t`, and provided to the operator.
For the [premade SciMLOperators](premade_operators.md), one can specify the keyword arguments used by an operator with an `accepted_kwargs` argument (by default, none are passed).

In the below example, we create an operator that gleefully ignores `u`, `p`, and `t` and uses its own special scaling.
```@example
using SciMLOperators

γ = ScalarOperator(0.0; update_func=(a, u, p, t; my_special_scaling) -> my_special_scaling,
                   accepted_kwargs=(:my_special_scaling,))

# Update coefficients, then apply operator
update_coefficients!(γ, nothing, nothing, nothing; my_special_scaling=7.0)
@show γ * [2.0]

# Use operator application form
@show γ([2.0], nothing, nothing; my_special_scaling = 5.0)
nothing # hide
```