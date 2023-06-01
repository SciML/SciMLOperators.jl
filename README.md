# SciMLOperators.jl

*Unified operator interface for `SciML.ai` and beyond*

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/SciMLOperators/stable)

[![codecov](https://codecov.io/gh/SciML/SciMLOperators.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/SciMLOperators.jl)
[![Build Status](https://github.com/SciML/SciMLOperators.jl/workflows/CI/badge.svg)](https://github.com/SciML/SciMLOperators.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

`SciMLOperators` is a package for managing linear, nonlinear, and
time-dependent operators acting on vectors, (or column-vectors of matrices).
We provide wrappers for matrix-free operators, fast tensor-product
evaluations, pre-cached mutating evaluations, as well as `Zygote`-compatible
non-mutating evaluations.

The lazily implemented operator algebra allows the user to update the
operator state by passing in an update function that accepts arbirary
parameter objects. Further, our operators behave like `AbstractMatrix` types
thanks to  overloads defined for methods in `Base`, and `LinearAlgebra`.

Therefore, an `AbstractSciMLOperator` can be passed to `LinearSolve.jl`,
or `NonlinearSolve.jl` as a linear/nonlinear operator, or to
`OrdinaryDiffEq.jl` as an `ODEFunction`. Examples of usage within the
`SciML` ecosystem are provided in the documentation.

## Installation
`SciMLOperators.jl` is a registerd package and can be installed via

```
julia> import Pkg
julia> Pkg.add("SciMLOperators")
```

## Examples

Let `M`, `D`, `F` be matrix, diagonal, and function-based `SciMLOperators`
respectively.

```julia
N = 4
f = (u, p, t) -> u .* u

M = MatrixOperator(rand(N, N))
D = DiagonalOperator(rand(N))
F = FunctionOperator(f, zeros(N), zeros(N))
```

Then, the following codes just work.

```julia
L1 = 2M + 3F + LinearAlgebra.I
L2 = D * F * M'
L3 = kron(M, D, F)
L4 = M \ D
L5 = [M; D]' * [M F; F D] * [F; D]
```

Each `L#` can be applied to vectors of appropriate sizes:

```julia
u = rand(N)
v = zeros(N)
u_kron = rand(N ^ 3)

v = L1 * u
mul!(v, L2, u)
v_kron = L3(u_kron, p, t)
L4(v, u, p, t)
```

Thanks to overloads defined for evaluation methods and traits in
`Base`, `LinearAlgebra`, the behaviour of a `SciMLOperator` is
indistinguishable from an `AbstractMatrix`. These operators can be
passed to linear solver packages, and even to ordinary differential
equation solvers.

## Operator update

## Features

* Matrix-free operators with `FunctionOperator`
* Fast tensor product evaluation
* Lazy algebra: addition, subtraction, multiplication, inverse, adjoint
* Mutating, nonmutating update behaviour (Zygote compatible)
* `InvertibleOperator` - pair fwd, bwd operators

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
`AbstractSciMLOperator`s, here allowing seamless compatibility with
linear solves, and nonlinear solvers. Further, due to the update functionality,
`AbstractSciMLOperato`s can represent an `ODEFunction` in `OrdinaryDiffEq.jl`,
and downstream packages. See tutorials for example of usage with
`OrdinaryDiffEq.jl`, `LinearSolve.jl`, `NonlinearSolve.jl`.

Further, the nonmutating update functionality allows gradient propogation
through `AbstractSciMLOperator`s, and is compatible with
automatic-differentiation libraries like
[`Zygote.jl`](https://github.com/SciML/DiffEqOperators.jl/tree/master).
An example of `Zygote.jl` usage with
[`Lux.jl`](https://github.com/LuxDL/Lux.jl) is also provided in the tutorials.

Please make an issue [here](https://github.com/SciML/SciMLOperators.jl/issues)
if you come across an unexpected issue using `SciMLOperator`

We provide below a list of packages that make use of `SciMLOperators`.
If you are using `SciMLOperators` in your work, feel free to create a PR
and add your package to this list.

* [`SciML.ai`](https://sciml.ai/) ecosystem: `SciMLOperators` is compatible
with, and utilized by every `SciML` package.
* [`CalculustJL`](https://github.com/CalculustJL) packages use
`SciMLOperators` to define matrix-free vector-calculus operators for solving
partial differential equations.
    * [`CalculustCore.jl`](https://github.com/CalculustJL/CalculustCore.jl)
    * [`FourierSpaces.jl`](https://github.com/CalculustJL/FourierSpaces.jl)
    * [`NodalPolynomialSpaces.jl`](https://github.com/CalculustJL/NodalPolynomialSpaces.jl)
* `SparseDiffTools.jl`

## Roadmap

- [ ] [Complete integration with `SciML` ecosystem](https://github.com/SciML/SciMLOperators.jl/issues/142)
- [ ] [Block-matrices](https://github.com/SciML/SciMLOperators.jl/issues/161)
- [x] [Benchmark and speed-up tensorbproduct evaluations](https://github.com/SciML/SciMLOperators.jl/issues/58)
- [ ] [Fast tensor-sum (`kronsum`) evaluation](https://github.com/SciML/SciMLOperators.jl/issues/53)
- [ ] Fully flesh out operator array algebra
- [ ] [Operator fusion/matrix chain multiplication at constant `(u, p, t)`-slices](https://github.com/SciML/SciMLOperators.jl/issues/51)

## Contributing

- Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing to SciML.
- There are a few community forums:
    - The #diffeq-bridged and #sciml-bridged channels in the
      [Julia Slack](https://julialang.org/slack/)
    - [JuliaDiffEq](https://gitter.im/JuliaDiffEq/Lobby) on Gitter
    - On the Julia Discourse forums (look for the [modelingtoolkit tag](https://discourse.julialang.org/tag/modelingtoolkit)
    - See also [SciML Community page](https://sciml.ai/community/)

