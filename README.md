# SciMLOperators.jl

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

```
N = 4
f = (u, p, t) -> u .* u

M = MatrixOperator(rand(N, N))
D = DiagonalOperator(rand(N))
F = FunctionOperator(f, zeros(N), zeros(N))
```

Then, the following codes just work.

```
L = 2M + 3F + LinearAlgebra.I
L = D * F * M'
L = kron(M, F)
LL = [M F; F D]
```

