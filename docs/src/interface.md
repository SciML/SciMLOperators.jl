# The `AbstractSciMLOperator` Interface

## Formal Properties of SciMLOperators

These are the formal properties that an `AbstractSciMLOperator` should obey
for it to work in the solvers.

1. An `AbstractSciMLOperator` represents a linear or nonlinear operator with input/output
   being `AbstractArray`s. Specifically, a SciMLOperator, `L`, of size `(M, N)` accepts
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

## Overloaded Traits

Thanks to overloads defined for evaluation methods and traits in
`Base`, `LinearAlgebra`, the behaviour of a `SciMLOperator` is
indistinguishable from an `AbstractMatrix`. These operators can be
passed to linear solver packages, and even to ordinary differential
equation solvers. The list of overloads to the `AbstractMatrix`
interface include, but are not limited, the following:

- `Base: size, zero, one, +, -, *, /, \, ∘, inv, adjoint, transpose, convert`
- `LinearAlgebra: mul!, ldiv!, lmul!, rmul!, factorize, issymmetric, ishermitian, isposdef`
- `SparseArrays: sparse, issparse`

## Multidimension arrays and batching

SciMLOperator can also be applied to `AbstractMatrix` subtypes where
operator-evaluation is done column-wise.

```julia
K = 10
u_mat = rand(N, K)

v_mat = F(u_mat, p, t) # == mul!(v_mat, F, u_mat)
size(v_mat) == (N, K) # true
```

`L#` can also be applied to `AbstractArray`s that are not
`AbstractVecOrMat`s so long as their size in the first dimension is appropriate
for matrix-multiplication. Internally, `SciMLOperator`s reshapes an
`N`-dimensional array to an `AbstractMatrix`, and applies the operator via
matrix-multiplication.

## Operator update

This package can also be used to write time-dependent, and
parameter-dependent operators, whose state can be updated per
a user-defined function.
The updates can be done in-place, i.e. by mutating the object,
or out-of-place, i.e. in a non-mutating, `Zygote`-compatible way.

For example,

```julia
u = rand(N)
p = rand(N)
t = rand()

# out-of-place update
mat_update_func = (A, u, p, t) -> t * (p * p')
sca_update_func = (a, u, p, t) -> t * sum(p)

M = MatrixOperator(zero(N, N); update_func = mat_update_func)
α = ScalarOperator(zero(Float64); update_func = sca_update_func)

L = α * M
L = cache_operator(L, u)

# L is initialized with zero state
L * u == zeros(N) # true

# update operator state with `(u, p, t)`
L = update_coefficients(L, u, p, t)
# and multiply
L * u != zeros(N) # true

# updates state and evaluates L at (u, p, t)
L(u, p, t) != zeros(N) # true
```

The out-of-place evaluation function `L(u, p, t)` calls
`update_coefficients` under the hood, which recursively calls
the `update_func` for each component `SciMLOperator`.
Therefore the out-of-place evaluation function is equivalent to
calling `update_coefficients` followed by `Base.*`. Notice that
the out-of-place evaluation does not return the updated operator.

On the other hand,, the in-place evaluation function, `L(v, u, p, t)`,
mutates `L`, and is equivalent to calling `update_coefficients!`
followed by `mul!`. The in-place update behaviour works the same way
with a few `<!>`s appended here and there. For example,

```julia
v = rand(N)
u = rand(N)
p = rand(N)
t = rand()

# in-place update
_A = rand(N, N)
_d = rand(N)
mat_update_func!  = (A, u, p, t) -> (copy!(A, _A); lmul!(t, A); nothing)
diag_update_func! = (diag, u, p, t) -> copy!(diag, N)

M = MatrixOperator(zero(N, N); update_func! = mat_update_func!)
D = DiagonalOperator(zero(N); update_func! = diag_update_func!)

L = D * M
L = cache_operator(L, u)

# L is initialized with zero state
L * u == zeros(N) # true

# update L in-place
update_coefficients!(L, u, p, t)
# and multiply
mul!(v, u, p, t) != zero(N) # true

# updates L in-place, and evaluates at (u, p, t)
L(v, u, p, t) != zero(N) # true
```

The update behaviour makes this package flexible enough to be used
in `OrdianryDiffEq`. As the parameter object `p` is often reserved
for sensitivy computation via automatic-differentiation, a user may
prefer to pass in state information via other arguments. For that
reason, we allow for update functions with arbitrary keyword arguments.

```julia
mat_update_func = (A, u, p, t; scale = 0.0) -> scale * (p * p')

M = MatrixOperator(zero(N, N); update_func = mat_update_func,
                   accepted_kwargs = (:state,))

M(u, p, t) == zeros(N) # true
M(u, p, t; scale = 1.0) != zero(N)
```

## Interface API Reference

```@docs
update_coefficients
update_coefficients!
cache_operator
concretize
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
```

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
