# Usage with `SciML` and beyond

## Why `SciMLOperators`?

Many functions, from linear solvers to differential equations, require
the use of matrix-free operators in order to achieve maximum performance in
many scenarios. `SciMLOperators.jl` defines the abstract interface for how
operators in the SciML ecosystem are supposed to be defined. It gives the
common set of functions and traits which solvers can rely on for properly
performing their tasks. Along with that, `SciMLOperators.jl` provides
definitions for the basic standard operators which are used in building
blocks for most tasks, both simplifying the use of operators while also
demonstrating to users how such operators can be built and used in practice.

`SciMLOperators.jl` has the design that is required in order to be used in
all scenarios of equation solvers. For example, Magnus integrators for
differential equations require defining an operator ``u' = A(t) u``, while
Munthe-Kaas methods require defining operators of the form ``u' = A(u) u``.
Thus the operators need some form of time and state dependence which the
solvers can update and query when they are non-constant
(`update_coefficients!`). Additionally, the operators need the ability to
act like "normal" functions for equation solvers. For example, if `A(u,p,t)`
has the same operation as `update_coefficients(A, u, p, t); A * u`, then `A`
can be used in any place where a differential equation definition
`f(u, p, t)` is used without requring the user or solver to do any extra
work. Thus while previous good efforts for matrix-free operators have existed
in the Julia ecosystem, such as
[LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl), those
operator interfaces lack these aspects in order to actually be fully seamless
with downstream equation solvers. This necessitates the definition and use of
an extended operator interface with all of these properties, hence the
`AbstractSciMLOperator` interface.

Some packages providing similar functionality are
* [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl)
* [`DiffEqOperators.jl`](https://github.com/SciML/DiffEqOperators.jl/tree/master) (deprecated)

## Interoperability and extended Julia ecosystem

`SciMLOperator.jl` overloads the `AbstractMatrix` interface for
`AbstractSciMLOperator`s, allowing seamless compatibility with
linear solves, and nonlinear solvers. Further, due to the update functionality,
`AbstractSciMLOperator`s can represent an `ODEFunction` in `OrdinaryDiffEq.jl`,
and downstream packages. See tutorials for example of usage with
`OrdinaryDiffEq.jl`, `LinearSolve.jl`, `NonlinearSolve.jl`.

Further, the nonmutating update functionality allows gradient propogation
through `AbstractSciMLOperator`s, and is compatible with
automatic-differentiation libraries like
[`Zygote.jl`](https://github.com/SciML/DiffEqOperators.jl/tree/master).
An example of `Zygote.jl` usage with
[`Lux.jl`](https://github.com/LuxDL/Lux.jl) is also provided in the tutorials.

Please make an issue [here](https://github.com/SciML/SciMLOperators.jl/issues)
if you come across an unexpected issue while using `SciMLOperators`.

We provide below a list of packages that make use of `SciMLOperators`.
If you are using `SciMLOperators` in your work, feel free to create a PR
and add your package to this list.

* [`SciML.ai`](https://sciml.ai/) ecosystem: `SciMLOperators` is compatible with, and utilized by every `SciML` package.
* [`CalculustJL`](https://github.com/CalculustJL) packages use `SciMLOperators` to define matrix-free vector-calculus operators for solving partial differential equations.
    * [`CalculustCore.jl`](https://github.com/CalculustJL/CalculustCore.jl)
    * [`FourierSpaces.jl`](https://github.com/CalculustJL/FourierSpaces.jl)
    * [`NodalPolynomialSpaces.jl`](https://github.com/CalculustJL/NodalPolynomialSpaces.jl)
* `SparseDiffTools.jl`

