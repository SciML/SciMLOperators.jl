# The AbstractSciMLOperator Interface

## Formal Properties of DiffEqOperators

These are the formal properties that an `AbstractSciMLOperator` should obey
for it to work in the solvers.

## AbstractDiffEqOperator Interface Description

1. Function call and multiplication: `L(du,u,p,t)` for inplace and `du = L(u,p,t)` for
   out-of-place, meaning `L*u` and `mul!`.
2. If the operator is not a constant, update it with `(u,p,t)`. A mutating form, i.e.
   `update_coefficients!(A,u,p,t)` that changes the internal coefficients, and a
   out-of-place form `B = update_coefficients(A,u,p,t)`.
3. `isconstant(A)` trait for whether the operator is constant or not.

## AbstractDiffEqLinearOperator Interface Description

1. `AbstractSciMLLinearOperator <: AbstractSciMLOperator`
2. Can absorb under multiplication by a scalar. In all algorithms things like
   `dt*L` show up all the time, so the linear operator must be able to absorb
   such constants.
4. `isconstant(A)` trait for whether the operator is constant or not.
5. Optional: `diagonal`, `symmetric`, etc traits from LinearMaps.jl.
6. Optional: `exp(A)`. Required for simple exponential integration.
7. Optional: `expv(A,u,t) = exp(t*A)*u` and `expv!(v,A::AbstractSciMLOperator,u,t)`
   Required for sparse-saving exponential integration.
8. Optional: factorizations. `ldiv!`, `factorize` et. al. This is only required
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