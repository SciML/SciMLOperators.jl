# [The `AbstractSciMLOperator` Interface](@id operator_interface)

```@docs
SciMLOperators.AbstractSciMLOperator
SciMLOperators.AbstractSciMLScalarOperator
```

## Interface API Reference

```@docs
update_coefficients
update_coefficients!
cache_operator
concretize
SciMLOperators.DEFAULT_UPDATE_FUNC
```

## Traits

```@docs
isconstant
iscached
issquare
islinear
isconvertible
has_adjoint
has_expmv
has_expmv!
has_exp
has_mul
has_mul!
has_ldiv
has_ldiv!
SciMLOperators.has_concretization
SciMLOperators.NoKwargFilter
```

## Developer Extension Hooks

```@docs
SciMLOperators.has_tensor_outer_mul_fast
SciMLOperators.tensor_outer_mul_fast!
```

### Sharing scratch space between the summands of an `AddedOperator`

`mul!` applies the summands of an `AddedOperator` one after another, so no two of their
caches are ever live at the same time and a summand can reuse the scratch of an earlier
one. An operator type takes part by defining `getcache`, and can additionally avoid
allocating a cache it would only discard by defining `adopt_cache`. Both default to
declining, so a type that defines neither behaves exactly as it did before.

A wrapper holding no scratch of its own, such as `ScaledOperator`, forwards all three hooks
to the operator it wraps, so `2A + 3B` and `A - B` share as `A + B` does. A new wrapper of
that kind should do the same.

!!! warning "Concurrency"
    
    Applying a cached operator from several tasks at once was never safe, because they
    would write to the same scratch. Once its summands share scratch they are not
    independently safe either, so code that hand-parallelizes over the `ops` of an
    `AddedOperator` needs its own uncached copies.

```@docs
SciMLOperators.getcache
SciMLOperators.update_cache
SciMLOperators.adopt_cache
```

## Note About Affine Operators

Affine operators are operators that have the action `Q*x = A*x + b`. These operators have
no matrix representation, since if there was, it would be a linear operator instead of an
affine operator. You can only represent an affine operator as a linear operator in a
dimension of one larger via the operation: `[A b] * [u;1]`, so it would require something modified
to the input as well. As such, affine operators are a distinct generalization of linear operators.

While it seems like it might doom the idea of using matrix-free affine operators, it turns out
that affine operators can be used in all cases where matrix-free linear solvers are used due to
an easy generalization of the standard convergence proofs. If Q is the affine operator
``Q(x) = Ax + b``, then solving ``Qx = c`` is equivalent to solving ``Ax + b = c`` or ``Ax = c-b``.
If you now do this same “plug-and-chug” handling of the affine operator into the GMRES/CG/etc.
convergence proofs, move the affine part to the rhs residual, and show it converges to solving
``Ax = c-b``, and thus GMRES/CG/etc. solves ``Q(x) = c`` for an affine operator properly.

That same trick can be used mostly anywhere you would've had a linear operator to extend
the proof to affine operators, so then ``exp(A*t)*v`` operations via Krylov methods work for A being
affine as well, and all sorts of things. Thus, affine operators have no matrix representation, but they
are still compatible with essentially any Krylov method, which would otherwise be compatible with
matrix-free representations, hence their support in the SciMLOperators interface.

## Note about keyword arguments to `update_coefficients!`

In rare cases, an operator may be used in a context where additional state is expected to be provided
to `update_coefficients!` beyond `u`, `p`, and `t`. In this case, the operator may accept this additional
state through arbitrary keyword arguments to `update_coefficients!`. When the caller provides these, they will be recursively propagated downwards through composed operators just like `u`, `p`, and `t`, and provided to the operator.
For the [premade SciMLOperators](premade_operators.md), one can specify the keyword arguments used by an operator with an `accepted_kwargs` argument (by default, none are passed).

In the below example, we create an operator that gleefully ignores `u`, `p`, and `t` and uses its own special scaling.

```@example
using SciMLOperators

γ = ScalarOperator(0.0;
    update_func = (a, u, p, t; my_special_scaling) -> my_special_scaling,
    accepted_kwargs = Val((:my_special_scaling,)))

# Update coefficients, then apply operator
update_coefficients!(γ, nothing, nothing, nothing; my_special_scaling = 7.0)
@show γ * [2.0]

# Use operator application form
@show γ([2.0], nothing, nothing, nothing; my_special_scaling = 5.0)
nothing # hide
```
